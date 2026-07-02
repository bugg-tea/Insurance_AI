"""
backend/api/main.py
====================
FastAPI production entry point.

Endpoints
---------
GET  /health                         liveness + readiness check
POST /query                          run full LangGraph pipeline
POST /upload                         ingest 1-5 PDFs, update FAISS
GET  /session/{session_id}/history   fetch Redis chat history
DELETE /session/{session_id}         clear Redis chat history

Architecture notes
------------------
- Single uvicorn worker (workers=1) because FAISS index and the retrieval
  pipeline live as in-process singletons.  Multi-worker would give each
  worker its own divergent index copy — wrong behaviour.
- PDF ingestion is synchronous inside the request (not background) so the
  caller knows when the index is actually ready before querying.
- Redis is used for short-term memory only; it gracefully degrades to
  stateless mode if unavailable.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── PYTHON PATH ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.session import (
    append_turn,
    clear_session,
    get_history,
    refresh_ttl,
)
from backend.api.pdf_pipeline import (
    ingest_uploaded_pdf,
    load_all_chunks_from_folder,
)

# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="InsureAI — Insurance RAG API",
    version="1.0.0",
    description="Multi-agent insurance policy Q&A with retrieval-augmented generation.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── STARTUP ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    """
    Pre-warm: load all existing chunk files → build retrieval pipeline →
    init LangGraph.  The graph itself is lazy (built on first /query call)
    but the retrieval pipeline is eager so the first query has no cold-start
    penalty beyond LLM latency.
    """
    print("[startup] Loading chunks and building retrieval pipeline …")
    try:
        import backend.agents_claude.orchestrator as orch
        from backend.app.retrieval.final import build_retrieval_pipeline
        from backend.agents_claude.retreival_agent import get_retrieval_agent

        chunks = load_all_chunks_from_folder()
        pipeline = build_retrieval_pipeline(chunks)

        orch._retrieval_pipeline = pipeline
        orch._graph = None           # build on first request

        agent = get_retrieval_agent()
        agent.set_pipeline(chunks)

        print(f"[startup] Ready — {len(chunks)} chunks loaded.")
    except Exception as exc:
        print(f"[startup] Warning: pipeline init failed ({exc}). "
              "The first query will trigger a cold start.")


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str       = Field(..., min_length=1, max_length=2000)
    session_id: str  = Field(..., min_length=1, max_length=128)
    user_id: str     = Field(default="anonymous", max_length=128)
    human_response: Optional[str] = Field(default=None, max_length=4000)


class QueryResponse(BaseModel):
    session_id: str
    markdown: str
    confidence: float
    confidence_label: str
    intent: Optional[str]         = None
    synthesized_answer: Optional[str] = None
    sources: List[str]            = []
    diagnostics: Optional[Dict[str, Any]] = None


class UploadSummary(BaseModel):
    file_name: str
    document_id: str
    insurer: str
    document_type: str
    page_count: int
    table_count: int
    chunk_count: int
    status: str


class UploadResponse(BaseModel):
    ingested: List[UploadSummary]
    errors: List[Dict[str, str]]
    total_indexed: int


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

MAX_UPLOAD_FILES = 5

ALLOWED_DOC_TYPES = [
    "Policy", "Claim", "CIS", "Coverage",
    "Exclusions", "Brochure", "PreAuth", "Proposal", "Policy Usage Guide",
]


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    """Liveness and basic readiness check."""
    import backend.agents_claude.orchestrator as orch
    pipeline_ready = orch._retrieval_pipeline is not None
    return {
        "status": "ok",
        "version": "1.0.0",
        "pipeline_ready": pipeline_ready,
    }


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Run the full LangGraph multi-agent pipeline for a user query.

    Flow:
      Redis history fetch → run_pipeline() → append to Redis → return response.
    """
    from backend.agents_claude.orchestrator import run_pipeline

    # Refresh session TTL on each interaction
    refresh_ttl(req.session_id)

    try:
        result: Dict[str, Any] = run_pipeline(
            raw_query=req.query,
            session_id=req.session_id,
            user_id=req.user_id,
            human_response=req.human_response,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    if result.get("status") == "awaiting_human_input":
        return QueryResponse(
            session_id=req.session_id,
            markdown=result.get("question") or "Please provide the missing information.",
            confidence=0.0,
            confidence_label="🟡 NEEDS INPUT",
            intent=None,
            synthesized_answer=None,
            sources=[],
            diagnostics={"status": "awaiting_human_input", "question": result.get("question"), "attempt": result.get("attempt")},
        )

    # Unpack report fields with safe defaults
    markdown          = result.get("markdown") or "I could not generate a response. Please try again."
    confidence        = float(result.get("confidence", 0.0))
    confidence_label  = result.get("confidence_label", "🟡 UNKNOWN")
    sources: List[str] = [s for s in (result.get("sources") or []) if s]

    # Synthesised answer lives inside result["json"] in the orchestrator
    synthesized_answer: Optional[str] = None
    if isinstance(result.get("json"), dict):
        synthesized_answer = result["json"].get("synthesized_answer")

    intent = None
    if isinstance(result.get("diagnostics"), dict):
        steps = result["diagnostics"].get("steps", [])
        for step in steps:
            if step.get("node") == "query_normalizer":
                intent = step.get("snapshot", {}).get("intent")
                break

    # Persist to Redis
    append_turn(req.session_id, "user",      req.query)
    append_turn(req.session_id, "assistant", markdown)

    return QueryResponse(
        session_id=req.session_id,
        markdown=markdown,
        confidence=confidence,
        confidence_label=confidence_label,
        intent=intent,
        synthesized_answer=synthesized_answer,
        sources=sources,
    )


@app.post("/upload", response_model=UploadResponse, tags=["upload"])
async def upload_pdfs(
    files: List[UploadFile] = File(..., description="Up to 5 PDF files"),
    insurer:  str = Form(default="Uploaded",
                         description="Insurer name, e.g. 'HDFC ERGO'"),
    doc_type: str = Form(default="Policy",
                         description="Document type folder name"),
) -> UploadResponse:
    """
    Upload 1–5 PDFs → extract → chunk → embed → update FAISS.

    The response is returned only after ingestion is fully complete so the
    caller can immediately query the newly uploaded content.
    """
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_UPLOAD_FILES} files per upload. Got {len(files)}.",
        )

    if doc_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"doc_type must be one of {ALLOWED_DOC_TYPES}. Got '{doc_type}'.",
        )

    results: List[UploadSummary] = []
    errors:  List[Dict[str, str]] = []

    # Use a single temp directory — files are processed one by one but stay
    # alive for the entire batch so paths remain valid.
    with tempfile.TemporaryDirectory() as tmp_dir:
        for upload_file in files:
            fname = upload_file.filename or "unknown.pdf"

            if not fname.lower().endswith(".pdf"):
                errors.append({"file": fname, "error": "Not a PDF file."})
                continue

            tmp_path = os.path.join(tmp_dir, fname)
            content  = await upload_file.read()

            with open(tmp_path, "wb") as fh:
                fh.write(content)

            try:
                summary = ingest_uploaded_pdf(
                    file_path=tmp_path,
                    insurer=insurer,
                    doc_type_folder=doc_type,
                )
                results.append(UploadSummary(**summary))
            except Exception as exc:
                errors.append({"file": fname, "error": str(exc)})

    return UploadResponse(
        ingested=results,
        errors=errors,
        total_indexed=len(results),
    )


@app.get("/session/{session_id}/history", tags=["session"])
def get_session_history(session_id: str) -> Dict[str, Any]:
    """Return the chat history stored in Redis for this session."""
    return {
        "session_id": session_id,
        "history":    get_history(session_id),
    }


@app.delete("/session/{session_id}", tags=["session"])
def delete_session(session_id: str) -> Dict[str, str]:
    """Clear all history for this session from Redis."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ── ENTRYPOINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
        workers=1,      # must stay 1 — FAISS index is an in-process singleton
        log_level="info",
    )