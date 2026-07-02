"""
chunker_runner_saver.py (vFINAL - Production Runner)

Purpose
-------
- Runs chunker.py safely on every extracted insurance document JSON.
- Zero crash on a bad document: errors are isolated per-file, logged with a
  full traceback, and saved to backend/data/failed_chunks/ for inspection -
  one bad PDF never takes down the whole batch.
- RESUMABLE / LOW LATENCY: if a document's chunk output already exists on
  disk, it is skipped by default, so re-running this script after adding a
  handful of new PDFs only does work for the new ones. Pass --force (or set
  FORCE_REPROCESS=1) to rebuild everything from scratch.
"""

from __future__ import annotations

import os
import json
import sys
import traceback
from typing import Dict, Any, List

from tqdm import tqdm

from backend.app.utils.chunker import build_chunks

# =========================================================
# PATHS
# =========================================================

EXTRACTED_PATH = "backend/data/extracted"
CHUNK_PATH = "backend/data/chunks"
FAILED_PATH = "backend/data/failed_chunks"

os.makedirs(CHUNK_PATH, exist_ok=True)
os.makedirs(FAILED_PATH, exist_ok=True)


# =========================================================
# LOAD
# =========================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# VALIDATION (STRUCTURAL - CATCHES SILENT CHUNKER BUGS EARLY)
# =========================================================

def validate_output(output: Dict[str, Any]) -> bool:
    """
    Ensures chunker output is structurally valid before it's trusted enough
    to feed the embeddings pipeline.
    """
    if not output:
        return False
    if "chunks" not in output or not isinstance(output["chunks"], list):
        return False
    if len(output["chunks"]) == 0:
        return False

    # Check every chunk (cheap - chunk_count is small per document) instead
    # of only the first few, since a single malformed chunk later in the
    # list would otherwise slip through unnoticed.
    required_keys = ("chunk_id", "text", "embedding_text", "chunk_type", "page_start", "page_end")
    for c in output["chunks"]:
        if not isinstance(c, dict):
            return False
        if any(not c.get(k) and c.get(k) != 0 for k in required_keys):
            return False
        # table rows must always carry their table_id - this is the whole
        # point of table-aware chunking, so treat its absence as a hard
        # validation failure rather than a silent gap.
        if c.get("chunk_type") in ("table_row", "table_row_part", "table_meta") and not c.get("table_id"):
            return False

    return True


# =========================================================
# SAVE OUTPUT (ATOMIC - NEVER LEAVES A HALF-WRITTEN FILE)
# =========================================================

def _atomic_write_json(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def save_chunks(doc_id: str, data: Dict[str, Any]) -> None:
    path = os.path.join(CHUNK_PATH, f"{doc_id}.json")
    _atomic_write_json(path, data)


def save_failed(doc_id: str, doc: Dict[str, Any], error: str) -> None:
    path = os.path.join(FAILED_PATH, f"{doc_id}_failed.json")
    payload = {
        "document_id": doc_id,
        "error": error,
        "document": doc,
    }
    _atomic_write_json(path, payload)


# =========================================================
# STATS
# =========================================================

def print_stats(doc_id: str, output: Dict[str, Any]) -> None:
    type_count = output.get("chunk_type_counts") or {}
    print(f"\n📄 Document: {doc_id}")
    print(f"Total chunks: {output.get('chunk_count', len(output.get('chunks', [])))}")
    for k, v in type_count.items():
        print(f" - {k}: {v}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_all(force: bool = False) -> None:
    files: List[str] = [
        os.path.join(EXTRACTED_PATH, f)
        for f in os.listdir(EXTRACTED_PATH)
        if f.endswith(".json")
    ]

    print(f"Found {len(files)} extracted documents")

    success = 0
    failed = 0
    skipped = 0

    for file_path in tqdm(files, desc="Chunking documents"):

        doc_id = "unknown"
        document = None

        try:
            document = load_json(file_path)

            doc_id = (
                document.get("metadata", {}).get("document_id")
                or os.path.basename(file_path).replace(".json", "")
            )

            out_path = os.path.join(CHUNK_PATH, f"{doc_id}.json")

            # ==============================
            # RESUME / SKIP IF ALREADY DONE
            # ==============================
            if not force and os.path.exists(out_path):
                skipped += 1
                continue

            # ==============================
            # CORE CHUNKING CALL
            # ==============================
            output = build_chunks(document)

            # ==============================
            # VALIDATION
            # ==============================
            if not validate_output(output):
                raise ValueError("Invalid chunk output structure (failed validate_output)")

            # ==============================
            # SAVE OUTPUT
            # ==============================
            save_chunks(doc_id, output)
            print_stats(doc_id, output)

            success += 1

        except Exception as e:
            failed += 1
            tb = traceback.format_exc()

            print(f"\n❌ ERROR in file: {file_path}")
            print(f"Reason: {e}")

            save_failed(doc_id, document or {}, tb)

    print("\n==============================")
    print(f"✅ Success : {success}")
    print(f"⏭️  Skipped : {skipped} (already chunked - use --force to redo)")
    print(f"❌ Failed  : {failed}")
    print("==============================")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    force_flag = "--force" in sys.argv or os.environ.get("FORCE_REPROCESS") == "1"
    process_all(force=force_flag)