"""
backend/api/pdf_pipeline.py
============================
Ingestion pipeline for user-uploaded PDFs.

Pipeline:
  upload bytes → save to temp dir
  → extract_pdf()          (pdf_extractor)
  → normalize_document()   (chunker)
  → process_document()     (chunker)
  → save_pdf_chunks()      (chunker)
  → build_or_update_index() (embeddings — incremental FAISS upsert)
  → _reload_retrieval_pipeline() — hot-reload orchestrator singleton

Design notes:
  - Does NOT modify any existing file.
  - Monkey-patches chunker module-level path constants so hardcoded Windows
    paths are replaced with dynamic PROJECT_ROOT-relative paths at import time.
    Python resolves module globals at call time, so patching before any call
    works correctly.
  - Caller is responsible for providing a temp file path that stays alive
    during the entire call (use tempfile.TemporaryDirectory as context manager).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# ── PROJECT ROOT ──────────────────────────────────────────────────────────────
# backend/api/pdf_pipeline.py → parents: [api, backend, project_root]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── CANONICAL DATA PATHS ──────────────────────────────────────────────────────
EXTRACTED_PATH = str(PROJECT_ROOT / "backend" / "data" / "extracted")
CHUNKS_PATH    = str(PROJECT_ROOT / "backend" / "data" / "chunks")

# ── MONKEY-PATCH CHUNKER PATHS BEFORE ANY IMPORT OF ITS FUNCTIONS ────────────
# chunker.py has Windows-hardcoded DATA_FOLDER / CHUNK_OUTPUT_FOLDER at module
# level. We patch them here so every subsequent call inside chunker uses the
# correct paths regardless of OS.
import backend.app.utils.chunker as _chunker_mod  # noqa: E402
_chunker_mod.DATA_FOLDER         = EXTRACTED_PATH
_chunker_mod.CHUNK_OUTPUT_FOLDER = CHUNKS_PATH

from backend.app.utils.chunker import (   # noqa: E402  (after patch)
    normalize_document,
    process_document,
    save_pdf_chunks,
    get_file_signature,
)
from backend.app.utils.pdf_extractor import extract_pdf   # noqa: E402
from backend.app.utils.embeddings import build_or_update_index   # noqa: E402

# ── DOCUMENT TYPE MAPPING ─────────────────────────────────────────────────────
_DOC_TYPE_MAP: Dict[str, str] = {
    "Policy":            "policy",
    "Claim":             "claim",
    "CIS":               "cis",
    "Coverage":          "coverage",
    "Exclusions":        "exclusions",
    "Brochure":          "brochure",
    "PreAuth":           "preauth",
    "Proposal":          "proposal",
    "Policy Usage Guide":"usage_guide",
}


def _map_doc_type(folder_type: str) -> str:
    return _DOC_TYPE_MAP.get(folder_type, "other")


# ── MAIN INGESTION FUNCTION ───────────────────────────────────────────────────

def _find_existing_chunk_payload(document_id: str) -> Dict[str, Any] | None:
    """Return an existing chunk payload if the same document has already been indexed."""
    for path in sorted(glob.glob(os.path.join(CHUNKS_PATH, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                continue
            for chunk in data.get("chunks", []):
                metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                if metadata.get("document_id") == document_id:
                    return data
        except Exception as exc:
            print(f"[pdf_pipeline] Warning: could not inspect chunk file {path}: {exc}")
    return None


def ingest_uploaded_pdf(
    file_path: str,
    insurer: str = "Uploaded",
    doc_type_folder: str = "Policy",
) -> Dict[str, Any]:
    """
    Full ingestion pipeline for one uploaded PDF.

    Parameters
    ----------
    file_path     : absolute path to the PDF (caller manages temp lifecycle)
    insurer       : e.g. "HDFC ERGO", "Uploaded"
    doc_type_folder: one of the SUPPORTED_DOC_TYPES keys, e.g. "Policy"

    Returns
    -------
    Summary dict with page_count, chunk_count, status, etc.
    """

    # ── STEP 1: Hash + metadata ───────────────────────────────────────────────
    with open(file_path, "rb") as fh:
        file_bytes = fh.read()

    document_id = hashlib.sha256(file_bytes).hexdigest()
    file_name   = os.path.basename(file_path)

    metadata: Dict[str, Any] = {
        "insurer":       insurer,
        "document_type": _map_doc_type(doc_type_folder),
        "folder_type":   doc_type_folder,
        "file_name":     file_name,
        "file_path":     file_path,
        "document_id":   document_id,
        "file_size":     len(file_bytes),
        "last_modified": os.path.getmtime(file_path),
    }

    existing_payload = _find_existing_chunk_payload(document_id)
    if existing_payload is not None:
        print(f"[pdf_pipeline] Reusing existing chunks for document {document_id[:12]}.")
        build_or_update_index()
        _reload_retrieval_pipeline()
        return {
            "file_name":    file_name,
            "document_id":  document_id,
            "insurer":      insurer,
            "document_type": metadata["document_type"],
            "page_count":   metadata.get("page_count", 0),
            "table_count":  0,
            "chunk_count":  existing_payload.get("total_chunks", len(existing_payload.get("chunks", []))),
            "status":       "already_indexed",
        }

    # ── STEP 2: Extract PDF ───────────────────────────────────────────────────
    extracted = extract_pdf(file_path)

    document: Dict[str, Any] = {
        "metadata":        metadata,
        "raw_text":        extracted["raw_text"],
        "clean_text":      extracted["cleaned_text"],
        "pages":           extracted["pages"],
        "tables":          extracted["tables"],
        "raw_char_count":  extracted["raw_char_count"],
        "clean_char_count":extracted["clean_char_count"],
        "page_count":      extracted["page_count"],
        "table_count":     extracted["table_count"],
        "used_ocr_pages":  extracted["used_ocr_pages"],
        "enrichment":      extracted["enrichment"],
    }

    # ── STEP 3: Save extracted JSON ───────────────────────────────────────────
    os.makedirs(EXTRACTED_PATH, exist_ok=True)
    insurer_slug = insurer.replace(" ", "_")
    json_name    = f"{insurer_slug}_{metadata['document_type']}_{document_id}.json"
    extracted_out = os.path.join(EXTRACTED_PATH, json_name)

    with open(extracted_out, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)

    # ── STEP 4: Chunk ─────────────────────────────────────────────────────────
    normalized = normalize_document(document)
    # process_document() uses doc.get("enrichment") which normalize_document
    # drops — re-attach it manually.
    normalized["enrichment"] = document.get("enrichment", {})

    chunks    = process_document(normalized)
    signature = get_file_signature(normalized)

    save_pdf_chunks(
        file_name=file_name.replace(".pdf", ""),
        signature=signature,
        chunks=chunks,
    )

    # ── STEP 5: Embed → incremental FAISS upsert ──────────────────────────────
    build_or_update_index()

    # ── STEP 6: Hot-reload orchestrator retrieval singleton ───────────────────
    _reload_retrieval_pipeline()

    return {
        "file_name":    file_name,
        "document_id":  document_id,
        "insurer":      insurer,
        "document_type":metadata["document_type"],
        "page_count":   extracted["page_count"],
        "table_count":  extracted["table_count"],
        "chunk_count":  len(chunks),
        "status":       "indexed",
    }


# ── CHUNK LOADER ──────────────────────────────────────────────────────────────

def load_all_chunks_from_folder() -> List[Dict[str, Any]]:
    """
    Load every *.json chunk file from backend/data/chunks/ into a flat list.
    Called at API startup and after each upload.
    """
    all_chunks: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(CHUNKS_PATH, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            all_chunks.extend(data.get("chunks", []))
        except Exception as exc:
            print(f"[pdf_pipeline] Warning: could not load chunk file {path}: {exc}")
    return all_chunks


# ── RETRIEVAL PIPELINE RELOAD ─────────────────────────────────────────────────

def _reload_retrieval_pipeline() -> None:
    """
    Force-reset the orchestrator's module-level singletons and rebuild
    the retrieval pipeline with the latest chunks (including newly uploaded).

    This is safe to call multiple times — it re-reads all chunk files from
    disk each time so newly added documents are always picked up.
    """
    try:
        import backend.agents_claude.orchestrator as orch

        chunks = load_all_chunks_from_folder()

        from backend.app.retrieval.final import build_retrieval_pipeline
        pipeline = build_retrieval_pipeline(chunks)

        # Overwrite singletons
        orch._retrieval_pipeline = pipeline
        orch._graph = None          # graph will be rebuilt on next invocation

        from backend.agents_claude.retreival_agent import get_retrieval_agent
        agent = get_retrieval_agent()
        agent.set_pipeline(chunks)

        print(f"[pdf_pipeline] Retrieval pipeline reloaded — {len(chunks)} chunks.")
    except Exception as exc:
        print(f"[pdf_pipeline] Warning: retrieval pipeline reload failed: {exc}")