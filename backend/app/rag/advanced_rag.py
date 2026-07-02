"""
Advanced RAG Pipeline (full orchestration of the diagram you described)
===========================================================================

    User Query
      -> Query Rewriting            (query_normalizer.QueryNormalizerAgent)
      -> Hybrid Retrieval           (retrieval.final.RetrievalPipeline)
      -> Reranking                  (already inside RetrievalPipeline)
      -> Context Compression        (context_compressor.compress)
      -> Grounded Prompt -> LLM     (answer_generator.generate_answer)
      -> Self Reflection (Self-RAG) (self_rag.reflect)
      -> Response Validator         (response_validator.validate)
      -> Citation Checker           (citation_checker — used inside self_rag + validator)
      -> JSON Output
         |
         if faithfulness low -> Corrective RAG -> Retrieve Again -> Regenerate
                                 (corrective_rag.run_crag_loop)

    + Observability (observability.traced_node / log_event on every stage)
    + Evaluation (evaluation.evaluate — RAGAS-style + DeepEval-style, free)

Setup
-----
No new dependencies beyond what the individual modules already declare.
This file is pure orchestration glue. It works standalone with a mock
retrieval pipeline (see `__main__` below) and is also what
`orchestrator.py` should call for the answer-synthesis stage — see
ORCHESTRATOR_INTEGRATION.md for the exact wiring.

Run standalone:
    python -m backend.agents_claude.advanced_rag_pipeline
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Callable

from backend.app.rag.context_compression import compress
from backend.app.rag.answer_generator import generate_answer
from backend.app.rag.self_rag import reflect
from backend.app.rag.response_validator import validate
from backend.app.rag.citation_checeker import check_citations
from backend.app.rag.corrective_rag import run_crag_loop
from backend.app.rag.evaluation import evaluate
from backend.app.rag.observability import log_event

DEFAULT_TOP_K = 8


def _single_pass(
    query: str,
    chunks: List[Dict[str, Any]],
    token_budget: int = 1800,
) -> Dict[str, Any]:
    """One full pass: compress -> generate -> reflect -> validate. No retry
    logic here — that's corrective_rag.run_crag_loop's job."""

    timings: Dict[str, float] = {}

    t0 = time.time()
    compressed = compress(query, chunks, token_budget=token_budget)
    timings["compression_ms"] = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    gen = generate_answer(query, compressed["context_block"], compressed["used_chunks"])
    timings["generation_ms"] = round((time.time() - t1) * 1000, 1)

    t2 = time.time()
    reflection = reflect(query, compressed["context_block"], gen["answer"], compressed["used_chunks"])
    timings["reflection_ms"] = round((time.time() - t2) * 1000, 1)

    t3 = time.time()
    validation = validate(query, gen["answer"], reflection, compressed["used_chunks"])
    timings["validation_ms"] = round((time.time() - t3) * 1000, 1)

    timings["pass_ms"] = round(sum(timings.values()), 1)

    return {
        "answer": gen["answer"],
        "used_llm": gen["used_llm"],
        "used_chunks": compressed["used_chunks"],
        "dropped_chunk_count": compressed["dropped_count"],
        "tokens_used": compressed.get("tokens_used", 0),
        "reflection": reflection,
        "validation": validation,
        "all_candidate_chunks": chunks,
        "timings": timings,
    }


def run_advanced_rag(
    query: str,
    retrieval_pipeline: Any,
    session_id: str = "default",
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = 1800,
) -> Dict[str, Any]:
    """
    Main entry point.

    `retrieval_pipeline` must expose `.search(query: str, top_k: int) ->
    List[chunk dicts]` — i.e. the same interface as
    backend.app.retrieval.final.RetrievalPipeline (or final2.RetrievalPipeline).

    Returns a fully-formed JSON-serializable result with the generated
    answer, citations, faithfulness/relevance scores, the CRAG retry trail
    (if triggered), free evaluation metrics, and a trace summary.
    """

    start = time.time()
    log_event(session_id, {"node": "advanced_rag_pipeline", "phase": "start", "query": query})

    last_timings: Dict[str, float] = {}

    def _retrieve(q: str, k: int) -> List[Dict[str, Any]]:
        t0 = time.time()
        result = retrieval_pipeline.search(q, top_k=k)
        log_event(session_id, {
            "node": "hybrid_retrieval_plus_rerank",
            "query": q,
            "top_k": k,
            "result_count": len(result),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
        return result

    def _generate_and_validate(q: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.time()
        out = _single_pass(q, chunks, token_budget=token_budget)
        # capture timings from the last single pass so we can report a
        # stable latency breakdown even if CRAG retries.
        nonlocal last_timings
        last_timings = out.get("timings", {}) or last_timings
        log_event(session_id, {
            "node": "generate_reflect_validate",
            "query": q,
            "passed": out["validation"]["passed"],
            "faithfulness_score": out["validation"]["faithfulness_score"],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
        return out

    crag_result = run_crag_loop(
        query=query,
        retrieve_fn=_retrieve,
        generate_and_validate_fn=_generate_and_validate,
        initial_top_k=top_k,
    )
    # Compute retrieval statistics from the final attempt's candidate chunks
    eval_metrics = evaluate(
        query=query,
        answer=crag_result["answer"],
        used_chunks=crag_result["used_chunks"],
        citation_report=crag_result["reflection"]["citation_report"],
        all_candidate_chunks=crag_result.get("all_candidate_chunks", crag_result["used_chunks"]),
    )

    # Retrieval stats
    all_candidates = crag_result.get("all_candidate_chunks") or []
    scores = [float(c.get("score", 0.0)) for c in all_candidates]
    retrieved_count = len(all_candidates)
    top_score = round(max(scores), 3) if scores else 0.0
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    min_score = round(min(scores), 3) if scores else 0.0
    score_gap = 0.0
    if scores:
        if len(scores) > 1:
            score_gap = round(scores[0] - scores[1], 3)
        else:
            score_gap = round(scores[0], 3)

    retrieval_stats = {
        "retrieved": retrieved_count,
        "top_score": top_score,
        "avg_score": avg_score,
        "min_score": min_score,
        "score_gap": score_gap,
    }

    # Latency breakdown: prefer timings from the last pass if available
    timings = last_timings or crag_result.get("timings") or {}

    # Confidence blend: faithfulness (0.4), relevance (0.3), retrieval_confidence (0.3)
    faith = float(crag_result.get("validation", {}).get("faithfulness_score", 0.0) or 0.0)
    rel = float(crag_result.get("validation", {}).get("relevance_score", 0.0) or 0.0)
    retrieval_top = retrieval_stats.get("top_score", 0.0)
    retrieval_confidence = min(1.0, retrieval_top)
    confidence = round(max(0.0, min(1.0, 0.4 * faith + 0.3 * rel + 0.3 * retrieval_confidence)), 3)

    total_latency_ms = round((time.time() - start) * 1000, 1)
    log_event(session_id, {"node": "advanced_rag_pipeline", "phase": "end", "total_latency_ms": total_latency_ms})

    final_query_used = crag_result["crag_attempts"][-1]["query_used"] if crag_result.get("crag_attempts") else query

    return {
        "query": query,
        "final_query_used": final_query_used,
        "answer": crag_result["answer"],
        "used_llm": crag_result["used_llm"],
        "citations": crag_result["reflection"]["citation_report"]["valid_tags"],
        "sources": [
            {
                "chunk_id": c.get("chunk_id"),
                "tag": c.get("tag"),
                "page_start": c.get("page_start"),
                "chunk_type": c.get("chunk_type"),
                "document_id": c.get("document_id"),
                "section": c.get("section"),
                "score": round(float(c.get("score", 0.0)), 3),
            }
            for c in crag_result["used_chunks"]
        ],
        "faithfulness_score": crag_result["validation"]["faithfulness_score"],
        "relevance_score": crag_result["validation"]["relevance_score"],
        "passed_validation": crag_result["validation"]["passed"],
        "validation_reasons": crag_result["validation"]["reasons"],
        "validation": crag_result["validation"],
        "crag_triggered": crag_result["crag_triggered"],
        "crag_attempts": crag_result["crag_attempts"],
        "evaluation": eval_metrics,
        "retrieval_stats": retrieval_stats,
        "retrieval": {
            "compressed": len(crag_result.get("used_chunks", [])),
            "dropped": crag_result.get("dropped_chunk_count", 0),
            "compression_ratio": round(len(crag_result.get("used_chunks", [])) / max(1, len(all_candidates)), 2),
        },
        "latency_ms": {**timings, "total": total_latency_ms},
        "confidence": confidence,
        "total_latency_ms": total_latency_ms,
    }


# =========================================================
# TEST (standalone, mock retrieval pipeline — no real index needed)
# =========================================================

if __name__ == "__main__":
    import json

    class MockRetrievalPipeline:
        """Mimics backend.app.retrieval.final.RetrievalPipeline.search()."""

        CORPUS = [
            {"chunk_id": "wp001", "text": "Waiting period for pre-existing diseases is 48 months from policy inception.", "chunk_type": "clause", "score": 0.92, "page_start": 5},
            {"chunk_id": "mat002", "text": "Maternity benefits have a waiting period of 24 months and cover normal delivery up to 50000.", "chunk_type": "clause", "score": 0.81, "page_start": 9},
            {"chunk_id": "exc003", "text": "Cosmetic surgery and dental treatment are not covered under this policy unless arising from an accident.", "chunk_type": "exclusion", "score": 0.65, "page_start": 12},
            {"chunk_id": "cat004", "text": "Cataract surgery is covered after a 2 year waiting period subject to a sub-limit of 40000 per eye.", "chunk_type": "table_row", "score": 0.58, "page_start": 14},
        ]

        def search(self, query: str, top_k: int = 8):
            q = query.lower()
            scored = []
            for c in self.CORPUS:
                overlap = len(set(q.split()) & set(c["text"].lower().split()))
                if overlap > 0:
                    scored.append({**c, "score": c["score"] + 0.05 * overlap})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

    pipeline = MockRetrievalPipeline()

    TEST_QUERIES = [
        "What is the waiting period for pre-existing diseases?",
        "Is cosmetic surgery covered?",
        "What is the sub-limit for cataract surgery?",
        "Does this policy cover alien abduction?",  # should fail validation gracefully
    ]

    for q in TEST_QUERIES:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        result = run_advanced_rag(q, pipeline, session_id="advanced-rag-test")
        print(json.dumps(result, indent=2))