"""
embeddings_faiss_pipeline.py (vFINAL - Production Embedding + FAISS Store)

Purpose
-------
- Reads every chunk produced by chunker_runner_saver.py
  (backend/data/chunks/*.json).
- Embeds chunk text with a FREE, LOCAL sentence-transformers model
  (no API key, no cost, runs fine on a laptop CPU).
- Stores vectors in a FAISS index that is PERSISTED TO DISK and only ever
  grows: re-running this script (e.g. on every app refresh / restart) does
  NOT wipe the index. Already-embedded chunks are skipped (incremental
  upsert), so re-runs after adding a handful of new PDFs are fast.
- Ships a small search() helper for low-latency semantic retrieval, with
  the model and index both cached in memory after first load - exactly the
  pattern you want behind an API endpoint / chat backend.

Model choice
------------
Default: "sentence-transformers/all-MiniLM-L6-v2"
  - 384-dim, ~80MB, very fast on CPU -> good default for "low latency".
Alternative (slightly better retrieval quality, still CPU-friendly,
still free, still ~130MB): "BAAI/bge-small-en-v1.5"
  - If you switch to this, also wrap queries with the instruction prefix
    BGE expects ("Represent this sentence for searching relevant passages: ")
    for best results - see EMBED_QUERY_PREFIX below.
Both download once from Hugging Face on first run and are cached locally
afterwards (~/.cache/huggingface by default) - no further network needed.

Run
---
    python embeddings_faiss_pipeline.py build            # embed + index everything new
    python embeddings_faiss_pipeline.py search "query"   # quick retrieval test
    python embeddings_faiss_pipeline.py stats             # show index size
    python embeddings_faiss_pipeline.py rebuild            # wipe + rebuild from scratch (explicit, opt-in only)
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import faiss
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# If you switch MODEL_NAME to a BGE model, set this so queries get the
# instruction prefix BGE was trained with. Leave "" for MiniLM.
EMBED_QUERY_PREFIX = ""

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHUNK_PATH = str(PROJECT_ROOT / "backend" / "data" / "chunks")
INDEX_DIR = str(PROJECT_ROOT / "backend" / "data" / "faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "insurance.index")
META_FILE = os.path.join(INDEX_DIR, "metadata_store.json")

BATCH_SIZE = 64

os.makedirs(INDEX_DIR, exist_ok=True)


# =========================================================
# MODEL (LAZY, CACHED - LOADED ONCE PER PROCESS)
# =========================================================

_model = None


def get_model():
    """Loads the embedding model once and reuses it (low latency for
    repeated calls e.g. inside a server process)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Batch-encodes texts to L2-normalized float32 vectors (so FAISS inner
    product == cosine similarity)."""
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype("float32")


def get_embedding_dim() -> int:
    return get_model().get_sentence_embedding_dimension()


def infer_chunk_type(chunk: Dict[str, Any]) -> str:
    """Heuristic fallback for missing chunk_type values."""
    text = " ".join(str(chunk.get(k, "")) for k in ("text", "embedding_text", "content", "raw_text")).lower()

    if chunk.get("table_id") or chunk.get("row_number") is not None:
        return "table_row"
    if any(k in text for k in ("not covered", "excluded", "exclusion", "waiting period", "pre-existing")):
        return "exclusion"
    if any(k in text for k in ("cover", "covered", "hospitalization", "benefit", "eligibility")):
        return "clause"
    return "general_info"


def normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure each chunk has the minimum fields needed by the indexer."""
    normalized = dict(chunk)

    if not normalized.get("chunk_id"):
        normalized["chunk_id"] = f"C_{uuid.uuid4().hex[:8]}"

    text_value = (
        normalized.get("embedding_text")
        or normalized.get("text")
        or normalized.get("content")
        or normalized.get("raw_text")
        or ""
    )
    if not normalized.get("text"):
        normalized["text"] = text_value

    if not normalized.get("embedding_text") and normalized.get("text"):
        normalized["embedding_text"] = normalized["text"]

    if not normalized.get("chunk_type"):
        normalized["chunk_type"] = infer_chunk_type(normalized)

    return normalized


# =========================================================
# DETERMINISTIC chunk_id -> int64 FAISS id
# =========================================================

def chunk_id_to_faiss_id(chunk_id: str) -> int:
    """
    FAISS needs int64 ids. chunk_id is normally a sha256 hex string (from
    chunker.py) - we take the first 15 hex chars (60 bits), which keeps us
    safely inside the positive int64 range with a negligible collision risk
    for any realistic corpus size. If chunk_id ever isn't valid hex (e.g.
    hand-built test data or a future non-chunker source), we re-hash it
    first so this never raises on unexpected input.
    """
    try:
        return int(chunk_id[:15], 16)
    except (ValueError, TypeError):
        import hashlib
        rehashed = hashlib.sha256(str(chunk_id).encode("utf-8", errors="ignore")).hexdigest()
        return int(rehashed[:15], 16)


# =========================================================
# METADATA STORE (PERSISTED ALONGSIDE THE FAISS INDEX)
# =========================================================

def _atomic_write_json(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_metadata_store() -> Dict[str, Any]:
    """
    Structure:
    {
      "model_name": "...",
      "dim": 384,
      "by_id": { "<faiss_id_str>": {chunk payload...} },
      "chunk_to_id": { "<chunk_id>": "<faiss_id_str>" }
    }
    """
    if not os.path.exists(META_FILE):
        return {"model_name": MODEL_NAME, "dim": None, "by_id": {}, "chunk_to_id": {}}

    with open(META_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata_store(store: Dict[str, Any]) -> None:
    _atomic_write_json(META_FILE, store)


# =========================================================
# FAISS INDEX LOAD / CREATE (NEVER ERASES ON A NORMAL RUN)
# =========================================================

def load_or_create_index(dim: int) -> faiss.Index:
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        if index.d != dim:
            raise RuntimeError(
                f"Existing FAISS index dim ({index.d}) does not match the "
                f"current embedding model's dim ({dim}). You likely changed "
                f"MODEL_NAME after already building an index. Run with "
                f"'rebuild' to start a fresh index for the new model - this "
                f"is destructive and intentional, so it's a separate command."
            )
        return index

    # Inner product on L2-normalized vectors == cosine similarity.
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)


def save_index(index: faiss.Index) -> None:
    faiss.write_index(index, INDEX_FILE)


# =========================================================
# GATHER CHUNKS FROM DISK
# =========================================================

def iter_chunk_files(chunk_dir: str = CHUNK_PATH):
    for path in sorted(glob.glob(os.path.join(chunk_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield path, json.load(f)
        except Exception as e:
            print(f"⚠️  Skipping unreadable chunk file {path}: {e}")


def iter_all_chunks(chunk_dir: str = CHUNK_PATH):
    for path, doc in iter_chunk_files(chunk_dir):
        for c in doc.get("chunks", []):
            yield normalize_chunk(c)


# =========================================================
# BUILD / UPDATE INDEX (INCREMENTAL UPSERT - THE CORE OF "DON'T ERASE")
# =========================================================

def build_or_update_index(batch_size: int = BATCH_SIZE) -> None:
    store = load_metadata_store()

    if store.get("model_name") and store["model_name"] != MODEL_NAME and store.get("by_id"):
        raise RuntimeError(
            f"The existing index was built with model '{store['model_name']}' "
            f"but MODEL_NAME is now '{MODEL_NAME}'. Mixing embeddings from two "
            f"different models in one index silently corrupts similarity "
            f"search. Either set MODEL_NAME back, or run the explicit "
            f"'rebuild' command to start over."
        )

    dim = get_embedding_dim()
    index = load_or_create_index(dim)

    chunk_to_id: Dict[str, str] = store.setdefault("chunk_to_id", {})
    by_id: Dict[str, Any] = store.setdefault("by_id", {})
    store["model_name"] = MODEL_NAME
    store["dim"] = dim

    new_chunks: List[Dict[str, Any]] = []
    seen_in_this_run = set()

    for c in iter_all_chunks():
        chunk_id = c.get("chunk_id")
        if not chunk_id or chunk_id in chunk_to_id or chunk_id in seen_in_this_run:
            continue
        seen_in_this_run.add(chunk_id)
        new_chunks.append(c)

    print(f"Existing vectors in index : {index.ntotal}")
    print(f"New chunks to embed       : {len(new_chunks)}")

    if not new_chunks:
        print("Nothing new to embed. Index is already up to date.")
        save_metadata_store(store)  # still persist model_name/dim if first run
        return

    added = 0
    for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding + indexing"):
        batch = new_chunks[i:i + batch_size]
        texts = [c.get("embedding_text") or c.get("text") or "" for c in batch]

        vectors = embed_texts(texts)
        faiss_ids = np.array([chunk_id_to_faiss_id(c["chunk_id"]) for c in batch], dtype="int64")

        index.add_with_ids(vectors, faiss_ids)

        for c, fid in zip(batch, faiss_ids):
            fid_str = str(int(fid))
            chunk_to_id[c["chunk_id"]] = fid_str
            by_id[fid_str] = {
                "chunk_id": c.get("chunk_id"),
                "text": c.get("text"),
                "chunk_type": c.get("chunk_type"),
                "section": c.get("section"),
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "table_id": c.get("table_id"),
                "row_number": c.get("row_number"),
                "row_part": c.get("row_part"),
                "document_id": c.get("document_id"),
                "insurer": c.get("insurer"),
                "file_name": c.get("file_name"),
                "document_type": c.get("document_type"),
            }
        added += len(batch)

        # Persist after every batch, not just at the very end - if the
        # process is killed mid-run (laptop sleeps, OOM, etc.) you keep
        # everything embedded so far instead of losing the whole run.
        save_index(index)
        save_metadata_store(store)

    print(f"✅ Added {added} new vectors. Index now has {index.ntotal} total vectors.")


def rebuild_index() -> None:
    """Explicit, destructive, opt-in full rebuild (e.g. after switching
    embedding models). Never called automatically."""
    confirm = input(
        f"This will DELETE {INDEX_FILE} and {META_FILE} and re-embed every "
        f"chunk from scratch. Type 'yes' to continue: "
    )
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return
    for f in (INDEX_FILE, META_FILE):
        if os.path.exists(f):
            os.remove(f)
    build_or_update_index()


# =========================================================
# SEARCH (LOW LATENCY - MODEL + INDEX BOTH CACHED IN MEMORY)
# =========================================================

_cached_index: Optional[faiss.Index] = None
_cached_store: Optional[Dict[str, Any]] = None


def _get_cached_index_and_store() -> Tuple[faiss.Index, Dict[str, Any]]:
    global _cached_index, _cached_store
    if _cached_index is None or _cached_store is None:
        store = load_metadata_store()
        dim = store.get("dim") or get_embedding_dim()
        _cached_index = load_or_create_index(dim)
        _cached_store = store
    return _cached_index, _cached_store


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:

    index, store = _get_cached_index_and_store()

    if index.ntotal == 0:
        return []

    q_text = f"{EMBED_QUERY_PREFIX}{query}" if EMBED_QUERY_PREFIX else query
    q_vec = embed_texts([q_text])

    k = min(top_k, index.ntotal)
    scores, ids = index.search(q_vec, k)

    results = []
    by_id = store.get("by_id", {})

    for score, fid in zip(scores[0], ids[0]):

        if fid == -1:
            continue

        payload = by_id.get(str(int(fid)))

        # =========================================
        # FIX: fallback instead of skipping result
        # =========================================
        if not payload:
            results.append({
                "score": float(score),
                "chunk_id": str(int(fid)),
                "text": "",
                "chunk_type": "unknown",
                "page_start": 0,
                "page_end": 0,
                "table_id": None,
                "row_number": None
            })
            continue

        # =========================================
        # FULL PAYLOAD RETURN (FIX 2 IMPLEMENTED)
        # =========================================
        results.append({
            "score": float(score),

            "chunk_id": payload.get("chunk_id"),
            "text": payload.get("text", ""),
            "chunk_type": payload.get("chunk_type", "unknown"),
            "page_start": payload.get("page_start", 0),
            "page_end": payload.get("page_end", 0),
            "table_id": payload.get("table_id"),
            "row_number": payload.get("row_number"),

            # optional metadata (kept for debugging)
            "section": payload.get("section"),
            "document_id": payload.get("document_id"),
            "insurer": payload.get("insurer"),
        })

    return results


def index_stats() -> Dict[str, Any]:
    store = load_metadata_store()
    index_exists = os.path.exists(INDEX_FILE)
    ntotal = 0
    if index_exists:
        index = faiss.read_index(INDEX_FILE)
        ntotal = index.ntotal
    return {
        "index_file": INDEX_FILE,
        "index_exists": index_exists,
        "total_vectors": ntotal,
        "model_name": store.get("model_name"),
        "dim": store.get("dim"),
        "chunks_indexed": len(store.get("chunk_to_id", {})),
    }


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and maintain the FAISS index.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="Embed any new chunks and add them to the persisted index (default).")
    sub.add_parser("rebuild", help="DESTRUCTIVE: wipe the index and re-embed everything.")
    sub.add_parser("stats", help="Show current index stats.")
    search_parser = sub.add_parser("search", help="Run a quick test query against the index.")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "rebuild":
        rebuild_index()
    elif args.command == "stats":
        print(json.dumps(index_stats(), indent=2))
    elif args.command == "search":
        for r in search(args.query, top_k=args.top_k):
            print(f"[{r['score']:.3f}] ({r.get('chunk_type')}, page {r.get('page_start')}) {r['text'][:120]}")
    else:
        build_or_update_index()