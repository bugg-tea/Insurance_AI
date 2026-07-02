"""
MEGA TEST — Full End-to-End Pipeline Runner
==============================================
Sends a batch of realistic insurance queries through the WHOLE stack:

    raw_query
      -> QueryNormalizerAgent (spelling/OCR fix, intent, entities)
      -> RetrievalPipeline (hybrid search + reranker, from final.py)
      -> Advanced RAG Pipeline (compression -> grounded LLM -> self-RAG ->
         validation -> CRAG retries -> RAGAS/DeepEval-style evaluation)
      -> observability trace summary

...and prints a single consolidated report per query, plus an aggregate
scorecard across the whole batch at the end.

Setup
-----
By default this runs against a small built-in MOCK_CHUNKS corpus so you
can verify the whole architecture works with ZERO setup, no PDFs, no FAISS
index, no API keys (everything degrades to free heuristics if an LLM /
embedding model isn't available — see each module's own docstring).

To test against your REAL corpus instead:
    1. Build your chunks + FAISS index as usual:
         python -m backend.app.chunker.chunker_runner_saver
         python -m backend.app.utils.embeddings_faiss_pipeline build
    2. Run this file with --real:
         python -m backend.agents_claude.mega_test --real
       It will load backend/data/chunks/*.json and build a real
       RetrievalPipeline via build_retrieval_pipeline(...).

Run:
    python -m backend.agents_claude.mega_test
    python -m backend.agents_claude.mega_test --real
"""

from __future__ import annotations

import sys
import json
import glob
import os
from typing import List, Dict, Any

from backend.agents_claude.query_normalizer import QueryNormalizerAgent
from backend.app.rag.advanced_rag import run_advanced_rag
from backend.app.rag.observability import summarize_trace


# =========================================================
# MOCK CORPUS (zero-setup path)
# =========================================================

class MockRetrievalPipeline:
    """Mimics backend.app.retrieval.final.RetrievalPipeline.search()."""

    CORPUS = [
        {"chunk_id": "wp001", "text": "Waiting period for pre-existing diseases is 48 months from policy inception.", "chunk_type": "clause", "score": 0.92, "page_start": 5},
        {"chunk_id": "mat002", "text": "Maternity benefits have a waiting period of 24 months and cover normal delivery up to 50000.", "chunk_type": "clause", "score": 0.81, "page_start": 9},
        {"chunk_id": "exc003", "text": "Cosmetic surgery and dental treatment are not covered under this policy unless arising from an accident.", "chunk_type": "exclusion", "score": 0.65, "page_start": 12},
        {"chunk_id": "cat004", "text": "Cataract surgery is covered after a 2 year waiting period subject to a sub-limit of 40000 per eye.", "chunk_type": "table_row", "score": 0.58, "page_start": 14},
        {"chunk_id": "prem005", "text": "Annual premium for the Gold plan is 18500 for a 35 year old individual with no co-payment.", "chunk_type": "table_row", "score": 0.55, "page_start": 3},
        {"chunk_id": "claim006", "text": "To file a claim, submit the reimbursement form within 30 days of discharge along with original bills.", "chunk_type": "claim_rule", "score": 0.6, "page_start": 20},
    ]

    def search(self, query: str, top_k: int = 8):
        q_tokens = set(query.lower().split())
        scored = []
        for c in self.CORPUS:
            overlap = len(q_tokens & set(c["text"].lower().split()))
            if overlap > 0:
                scored.append({**c, "score": c["score"] + 0.05 * overlap})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def build_real_pipeline():
    """Loads real chunks + builds the real hybrid+rerank+intent pipeline."""
    from backend.app.retrieval.final import build_retrieval_pipeline

    chunk_dir = "backend/data/chunks"
    files = glob.glob(os.path.join(chunk_dir, "*.json"))
    if not files:
        print(f"⚠️  No chunk files found in {chunk_dir}. Falling back to mock corpus.")
        return MockRetrievalPipeline()

    all_chunks: List[Dict[str, Any]] = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
            all_chunks.extend(payload.get("chunks", []))

    print(f"✅ Loaded {len(all_chunks)} real chunks from {len(files)} files.")
    return build_retrieval_pipeline(all_chunks)


# =========================================================
# TEST QUERIES (intentionally messy — spelling errors, OCR noise, etc.)
# =========================================================

TEST_QUERIES = [
    "waht is waitng perod for catarct",
    "is cosmetic surgery covered",
    "matrnity benifit waiting period",
    "how to file a claim after discharge",
    "what is the premium for gold plan",
    "does this policy cover alien abduction",   # expect low relevance, graceful answer
]


# =========================================================
# RUNNER
# =========================================================

def run_mega_test(use_real: bool = False) -> Dict[str, Any]:
    normalizer = QueryNormalizerAgent()
    pipeline = build_real_pipeline() if use_real else MockRetrievalPipeline()

    session_id = "mega-test-session"
    all_results = []

    for i, raw_query in enumerate(TEST_QUERIES, 1):
        print("\n" + "═" * 78)
        print(f"  [{i}/{len(TEST_QUERIES)}] RAW QUERY: {raw_query!r}")
        print("═" * 78)

        # ---- Stage 1: Query Normalization / Rewriting ----
        norm = normalizer.run({"raw_query": raw_query})
        normalized_query = norm["normalized_query"]
        print(f"  -> normalized_query : {normalized_query}")
        print(f"  -> intent           : {norm['intent']} (confidence={norm['intent_confidence']})")
        print(f"  -> comparison       : {norm['comparison']}")
        print(f"  -> entities         : {norm['entities']}")

        # ---- Stage 2-9: Advanced RAG pipeline (retrieval -> ... -> eval) ----
        rag_result = run_advanced_rag(
            query=normalized_query,
            retrieval_pipeline=pipeline,
            session_id=session_id,
        )
        print("\nVALIDATION REPORT")
        print(json.dumps(rag_result["validation"], indent=2))

        print("\nRETRIEVAL STATS")
        print(json.dumps(rag_result.get("retrieval_stats", {}), indent=2))

        print("\nRETRIEVAL")
        print(json.dumps(rag_result.get("retrieval", {}), indent=2))
        print(f"\n  ANSWER: {rag_result['answer']}")
        print(f"  CITATIONS: {rag_result['citations']}")
        print(f"  FAITHFULNESS: {rag_result['faithfulness_score']}  "
              f"RELEVANCE: {rag_result['relevance_score']}  "
              f"PASSED: {rag_result['passed_validation']}  "
              f"CRAG_TRIGGERED: {rag_result['crag_triggered']}")
        print(f"  RAGAS    : {rag_result['evaluation']['ragas']}")
        print(f"  DEEPEVAL : {rag_result['evaluation']['deepeval']}")

        all_results.append({
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "intent": norm["intent"],
            **rag_result,
        })

    # ---- Aggregate scorecard ----
    print("\n\n" + "█" * 78)
    print("  AGGREGATE SCORECARD")
    print("█" * 78)

    n = len(all_results)
    avg = lambda key_path: round(
        sum(_dig(r, key_path) for r in all_results) / n, 3
    ) if n else 0.0

    def _dig(d, path):
        for p in path.split("."):
            d = d.get(p, {}) if isinstance(d, dict) else 0
        return d if isinstance(d, (int, float)) else 0

    pass_rate = round(sum(1 for r in all_results if r["passed_validation"]) / n, 3) if n else 0.0
    crag_rate = round(sum(1 for r in all_results if r["crag_triggered"]) / n, 3) if n else 0.0

    scorecard = {
        "total_queries": n,
        "validation_pass_rate": pass_rate,
        "crag_trigger_rate": crag_rate,
        "avg_faithfulness": avg("faithfulness_score"),
        "avg_relevance": avg("relevance_score"),
        "avg_ragas_answer_relevancy": avg("evaluation.ragas.answer_relevancy"),
        "avg_ragas_context_precision": avg("evaluation.ragas.context_precision"),
        "avg_deepeval_hallucination": avg("evaluation.deepeval.hallucination_score"),
        "avg_deepeval_correctness": avg("evaluation.deepeval.correctness"),
    }
    print(json.dumps(scorecard, indent=2))

    print("\n  TRACE SUMMARY (observability.py):")
    print(json.dumps(summarize_trace(session_id), indent=2))

    return {"results": all_results, "scorecard": scorecard}


if __name__ == "__main__":
    use_real = "--real" in sys.argv
    run_mega_test(use_real=use_real)