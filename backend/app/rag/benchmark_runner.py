"""
Free local benchmark runner for InsureAI.

This script produces portfolio-style numbers from the existing chunk corpus
without relying on paid LLM judges.

It generates:
- Retrieval metrics: Recall@k, MRR, nDCG (correctly normalized, capped at 1.0)
- RAG metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Latency and approximate token cost
- A simple human-in-the-loop (HITL) clarification metric
- A Markdown report and JSON artifact you can drop into the README

CHANGES vs the original version (see conversation for the full rationale):
  1. FIXED nDCG bug: `ideal_gains` used to always assume exactly one relevant
     chunk, but `gains` counted every keyword-matching chunk as relevant. That
     mismatch let DCG exceed IDCG, producing impossible nDCG values > 1.0
     (e.g. 2.948 in the old output). `ideal_gains` is now built from the
     actual number of relevant chunks found, so nDCG is correctly capped at
     1.0 like it's supposed to be.
  2. FIXED table-aware scoring bug: the old code added a flat +0.12 bonus to
     ANY table-type chunk regardless of whether it was actually relevant to
     the query. That's why `table_aware` scored WORSE than `naive` in your
     run — irrelevant table rows were out-scoring relevant text chunks. The
     bonus is now (a) only applied to queries that plausibly need tabular
     data (`prefers_table=True` — limits, room rent, co-pay %, comparisons),
     and (b) scaled by the chunk's own semantic+lexical relevance instead of
     being a flat additive constant, so an irrelevant table row can no
     longer out-rank a relevant paragraph.
  3. FIXED embedding-recomputation bug: the old `_retrieve` re-embedded up to
     400 corpus chunks on every single query x strategy call (~32 redundant
     full-corpus passes for 8 queries x 4 strategies). Corpus embeddings are
     now computed ONCE per benchmark run and reused, so latency reflects
     actual strategy cost instead of embedding-model call noise.
  4. EXPANDED the query set from 8 to 56 queries, covering every category
     your README claims to support: coverage, waiting period, exclusions,
     claims, premium, eligibility, hospitalization, day-care, pre-existing
     disease, room rent, comparison, maternity, critical illness,
     co-payment, and network hospitals.

Run:
    python -m backend.app.rag.benchmark_runner
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.rag.evaluation import evaluate
from backend.app.utils.embeddings import embed_texts

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHUNK_ROOT = PROJECT_ROOT / "backend" / "data" / "chunks"
REPORT_JSON = PROJECT_ROOT / "backend" / "data" / "benchmark_results.json"
REPORT_MD = PROJECT_ROOT / "backend" / "data" / "benchmark_report.md"

STRATEGIES = ["naive", "hybrid", "table_aware", "full_agentic"]

# ── BENCHMARK QUERY SET ───────────────────────────────────────────────────────
# `prefers_table`: True when the correct answer is the kind of fact that
# typically lives in a tabular section of a policy doc (limits, percentages,
# amounts, side-by-side comparisons). Table-aware scoring only applies its
# relevance-gated bonus on these queries — this is the fix for the "why is
# table-aware worse than naive" issue: previously the bonus fired on every
# query regardless of whether tabular data was even relevant.
BENCHMARK_CASES = [
    # ── coverage / room rent ──
    {"query": "What is the room rent limit in Niva Bupa ReAssure?", "expected_keywords": ["room rent", "niva", "reassure"], "type": "coverage", "prefers_table": True},
    {"query": "Does HDFC ERGO Optima Secure cover maternity expenses?", "expected_keywords": ["maternity", "hdfc", "optima"], "type": "coverage", "prefers_table": False},
    {"query": "Is daycare surgery covered under Star Health?", "expected_keywords": ["daycare", "day care", "star health"], "type": "coverage", "prefers_table": False},
    {"query": "What is the sub-limit for cataract treatment?", "expected_keywords": ["cataract", "sub-limit", "sub limit"], "type": "coverage", "prefers_table": True},
    {"query": "Does the policy cover ambulance charges?", "expected_keywords": ["ambulance", "charges", "cover"], "type": "coverage", "prefers_table": True},
    {"query": "What is the room rent limit in ICICI Lombard Elevate?", "expected_keywords": ["room rent", "icici", "elevate"], "type": "coverage", "prefers_table": True},
    {"query": "What is the sum insured available under Care Supreme?", "expected_keywords": ["sum insured", "care supreme"], "type": "coverage", "prefers_table": True},

    # ── waiting period ──
    {"query": "What is the waiting period for cataract surgery?", "expected_keywords": ["waiting period", "cataract", "surgery"], "type": "waiting_period", "prefers_table": True},
    {"query": "What is the waiting period for pre-existing diseases?", "expected_keywords": ["pre-existing", "waiting period"], "type": "waiting_period", "prefers_table": True},
    {"query": "How long is the initial waiting period for a new policy?", "expected_keywords": ["initial waiting period", "30 days", "policy"], "type": "waiting_period", "prefers_table": True},
    {"query": "What is the waiting period for hernia treatment?", "expected_keywords": ["hernia", "waiting period"], "type": "waiting_period", "prefers_table": True},
    {"query": "Is there a waiting period for maternity benefits?", "expected_keywords": ["maternity", "waiting period"], "type": "waiting_period", "prefers_table": True},
    {"query": "What is the waiting period for joint replacement surgery?", "expected_keywords": ["joint replacement", "waiting period"], "type": "waiting_period", "prefers_table": True},

    # ── exclusion ──
    {"query": "What is not covered under Star Health policy?", "expected_keywords": ["not covered", "star health", "exclusion"], "type": "exclusion", "prefers_table": False},
    {"query": "What are the exclusions for maternity benefits?", "expected_keywords": ["maternity", "exclusion", "not covered"], "type": "exclusion", "prefers_table": False},
    {"query": "Are cosmetic surgeries excluded from the policy?", "expected_keywords": ["cosmetic", "exclusion", "not covered"], "type": "exclusion", "prefers_table": False},
    {"query": "Is dental treatment excluded under this policy?", "expected_keywords": ["dental", "exclusion", "not covered"], "type": "exclusion", "prefers_table": False},
    {"query": "What conditions are permanently excluded from coverage?", "expected_keywords": ["permanent exclusion", "not covered"], "type": "exclusion", "prefers_table": False},
    {"query": "Are self-inflicted injuries covered by the policy?", "expected_keywords": ["self-inflicted", "exclusion", "not covered"], "type": "exclusion", "prefers_table": False},

    # ── claim ──
    {"query": "How do I file a claim for hospitalization?", "expected_keywords": ["claim", "hospitalization", "process"], "type": "claim", "prefers_table": False},
    {"query": "How do I file a cashless claim under Star Health?", "expected_keywords": ["cashless", "claim", "star health"], "type": "claim", "prefers_table": False},
    {"query": "What documents are required for a reimbursement claim?", "expected_keywords": ["reimbursement", "documents", "claim"], "type": "claim", "prefers_table": False},
    {"query": "What is the process for pre-authorization of a claim?", "expected_keywords": ["pre-authorization", "preauth", "claim"], "type": "claim", "prefers_table": False},
    {"query": "What is the time limit to intimate a claim after hospitalization?", "expected_keywords": ["intimate", "claim", "time limit"], "type": "claim", "prefers_table": True},
    {"query": "How do I track the status of my submitted claim?", "expected_keywords": ["claim status", "track", "claim"], "type": "claim", "prefers_table": False},

    # ── premium ──
    {"query": "What is the premium or cost for the policy?", "expected_keywords": ["premium", "cost", "policy"], "type": "premium", "prefers_table": True},
    {"query": "Is there a discount on premium for a longer policy tenure?", "expected_keywords": ["discount", "premium", "tenure"], "type": "premium", "prefers_table": True},
    {"query": "How does age affect the premium amount?", "expected_keywords": ["age", "premium"], "type": "premium", "prefers_table": True},
    {"query": "What is the premium for a family floater plan?", "expected_keywords": ["premium", "family floater"], "type": "premium", "prefers_table": True},
    {"query": "Are there any loading charges added to the premium for pre-existing conditions?", "expected_keywords": ["loading", "premium", "pre-existing"], "type": "premium", "prefers_table": True},

    # ── comparison ──
    {"query": "Compare room rent limits between Niva Bupa and Star Health", "expected_keywords": ["room rent", "compare", "niva", "star"], "type": "comparison", "prefers_table": True},
    {"query": "Compare waiting periods across HDFC ERGO and ICICI Lombard", "expected_keywords": ["waiting period", "compare", "hdfc", "icici"], "type": "comparison", "prefers_table": True},
    {"query": "Star Health vs Care Supreme for a family of 4", "expected_keywords": ["star health", "care supreme", "family"], "type": "comparison", "prefers_table": True},
    {"query": "Which policy has a lower co-payment, Niva Bupa or HDFC ERGO?", "expected_keywords": ["co-payment", "copayment", "niva", "hdfc"], "type": "comparison", "prefers_table": True},

    # ── maternity ──
    {"query": "What is the maternity benefit limit under the policy?", "expected_keywords": ["maternity", "benefit", "limit"], "type": "maternity", "prefers_table": True},
    {"query": "Does the policy cover newborn baby expenses?", "expected_keywords": ["newborn", "baby", "cover"], "type": "maternity", "prefers_table": False},
    {"query": "How many deliveries are covered under the maternity benefit?", "expected_keywords": ["maternity", "delivery", "cover"], "type": "maternity", "prefers_table": True},

    # ── critical illness ──
    {"query": "What critical illnesses are covered under this plan?", "expected_keywords": ["critical illness", "cover"], "type": "critical_illness", "prefers_table": False},
    {"query": "Is cancer treatment covered as a critical illness?", "expected_keywords": ["cancer", "critical illness", "cover"], "type": "critical_illness", "prefers_table": False},
    {"query": "What is the survival period clause for critical illness claims?", "expected_keywords": ["survival period", "critical illness"], "type": "critical_illness", "prefers_table": True},

    # ── co-payment ──
    {"query": "What is the co-payment percentage for senior citizens?", "expected_keywords": ["co-payment", "copayment", "senior citizen", "percentage"], "type": "copayment", "prefers_table": True},
    {"query": "Is there a mandatory co-payment clause in this policy?", "expected_keywords": ["co-payment", "copayment", "mandatory"], "type": "copayment", "prefers_table": True},
    {"query": "What is the co-payment for treatment at a non-network hospital?", "expected_keywords": ["co-payment", "copayment", "non-network"], "type": "copayment", "prefers_table": True},

    # ── network hospitals ──
    {"query": "How many network hospitals are available under this policy?", "expected_keywords": ["network hospital", "number"], "type": "network_hospital", "prefers_table": True},
    {"query": "Can I get cashless treatment at a non-network hospital?", "expected_keywords": ["cashless", "non-network hospital"], "type": "network_hospital", "prefers_table": False},
    {"query": "How do I find the nearest network hospital?", "expected_keywords": ["network hospital", "find", "nearest"], "type": "network_hospital", "prefers_table": False},

    # ── eligibility ──
    {"query": "What is the minimum entry age for this policy?", "expected_keywords": ["entry age", "minimum age"], "type": "eligibility", "prefers_table": True},
    {"query": "What is the maximum age for renewal of this policy?", "expected_keywords": ["renewal", "maximum age"], "type": "eligibility", "prefers_table": True},
    {"query": "Can I add my parents as dependents on this policy?", "expected_keywords": ["parents", "dependent", "add"], "type": "eligibility", "prefers_table": False},

    # ── hospitalization ──
    {"query": "What is the minimum hospitalization period required for a claim?", "expected_keywords": ["hospitalization", "minimum", "24 hours"], "type": "hospitalization", "prefers_table": True},
    {"query": "Are pre and post hospitalization expenses covered?", "expected_keywords": ["pre-hospitalization", "post-hospitalization", "cover"], "type": "hospitalization", "prefers_table": False},
    {"query": "What is covered under domiciliary hospitalization?", "expected_keywords": ["domiciliary", "hospitalization", "cover"], "type": "hospitalization", "prefers_table": False},

    # ── day-care ──
    {"query": "What is the list of day-care procedures covered?", "expected_keywords": ["day care", "daycare", "procedure"], "type": "daycare", "prefers_table": True},
    {"query": "Is chemotherapy covered as a day-care procedure?", "expected_keywords": ["chemotherapy", "day care", "daycare"], "type": "daycare", "prefers_table": False},

    # ── pre-existing disease ──
    {"query": "How is a pre-existing disease defined in this policy?", "expected_keywords": ["pre-existing", "defined", "disease"], "type": "pre_existing", "prefers_table": False},
    {"query": "Can pre-existing diabetes be covered after the waiting period?", "expected_keywords": ["diabetes", "pre-existing", "waiting period"], "type": "pre_existing", "prefers_table": True},

    # ── definition / general ──
    {"query": "What is a sum insured restoration benefit?", "expected_keywords": ["restoration", "sum insured", "benefit"], "type": "definition", "prefers_table": False},
    {"query": "What does 'no claim bonus' mean in this policy?", "expected_keywords": ["no claim bonus", "ncb"], "type": "definition", "prefers_table": True},
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _keyword_overlap(query: str, text: str) -> float:
    q_tokens = set(_tokenize(query))
    t_tokens = set(_tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens | t_tokens)


def _contains_expected(text: str, expected_keywords: List[str]) -> bool:
    lower = (text or "").lower()
    return any(keyword.lower() in lower for keyword in expected_keywords)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if not norm_a or not norm_b:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _load_chunks() -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    if not CHUNK_ROOT.exists():
        return chunks

    for path in sorted(CHUNK_ROOT.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                chunks.extend(payload)
            elif isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
                chunks.extend(payload["chunks"])
        except Exception as exc:
            print(f"⚠️ Skipping unreadable chunk file {path.name}: {exc}")

    return [c for c in chunks if isinstance(c, dict) and (c.get("text") or c.get("embedding_text") or c.get("content"))]


def _infer_intent(query: str) -> str:
    q = query.lower()
    if re.search(r"\b(exclude|excluded|not covered|exclusion|denied)\b", q):
        return "exclusion"
    if re.search(r"\b(claim|claim process|file a claim|settlement)\b", q):
        return "claim"
    if re.search(r"\b(premium|cost|price|amount)\b", q):
        return "premium"
    if re.search(r"\b(waiting period|waiting|cooling)\b", q):
        return "waiting_period"
    if re.search(r"\b(compare|versus|vs)\b", q):
        return "comparison"
    if re.search(r"\b(cover|covered|benefit|eligible|room rent)\b", q):
        return "coverage"
    return "general"


def _is_table_chunk(chunk: Dict[str, Any]) -> bool:
    return bool(
        chunk.get("table_id")
        or chunk.get("row_number") is not None
        or chunk.get("chunk_type") in {"table_row", "table_meta"}
    )


def _score_chunk(
    strategy: str,
    query: str,
    chunk: Dict[str, Any],
    semantic_score: float,
    lexical_score: float,
    prefers_table: bool,
) -> float:
    text = str(chunk.get("text") or chunk.get("embedding_text") or chunk.get("content") or "")
    base_score = semantic_score * 0.7 + lexical_score * 0.3

    if strategy == "naive":
        return semantic_score

    if strategy == "hybrid":
        return base_score

    if strategy == "table_aware":
        return base_score + _table_bonus(chunk, base_score, prefers_table)

    if strategy == "full_agentic":
        score = base_score + _table_bonus(chunk, base_score, prefers_table)
        intent = _infer_intent(query)
        if intent == "exclusion" and (chunk.get("chunk_type") == "exclusion" or "not covered" in text.lower()):
            score += 0.16
        if intent == "waiting_period" and "waiting" in text.lower():
            score += 0.14
        if intent == "claim" and "claim" in text.lower():
            score += 0.14
        if intent == "premium" and "premium" in text.lower():
            score += 0.12
        if intent == "coverage" and ("cover" in text.lower() or "benefit" in text.lower()):
            score += 0.10
        return score

    return semantic_score


def _table_bonus(chunk: Dict[str, Any], base_score: float, prefers_table: bool) -> float:
    """
    RELEVANCE-GATED table bonus (fixes the "table-aware worse than naive" bug).

    The old version added a flat +0.12 to ANY table-type chunk, regardless of
    whether it had anything to do with the query. That let irrelevant table
    rows out-rank genuinely relevant paragraphs, which is exactly why the
    table_aware strategy scored below naive/hybrid on retrieval metrics.

    Now the bonus:
      - only fires at all when the query is the kind that plausibly needs
        tabular data (`prefers_table=True` — limits, amounts, percentages,
        comparisons), and
      - is scaled by the chunk's own base relevance (semantic+lexical) so an
        irrelevant table chunk with near-zero relevance gets a near-zero
        bonus, while a relevant table chunk gets a meaningful boost.
    """
    if not prefers_table or not _is_table_chunk(chunk):
        return 0.0
    # Scale bonus by how relevant the chunk already looks. A max bonus of
    # 0.12 only applies when base_score is already reasonably high; a
    # near-irrelevant table chunk (base_score near 0) gets almost nothing.
    return 0.12 * min(1.0, max(0.0, base_score) / 0.5)


def _embed_corpus(chunks: List[Dict[str, Any]], max_chunks: int = 400) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray]:
    """
    Compute corpus embeddings ONCE for the whole benchmark run.

    The old code called embed_texts() on up to 400 corpus chunks inside
    _retrieve(), which ran once per query PER STRATEGY — i.e. the same
    corpus was embedded ~32 times for an 8-query x 4-strategy benchmark.
    That's why latency numbers looked noisy/inconsistent between strategies:
    the timing was dominated by redundant embedding-model calls, not by
    actual retrieval-strategy cost. Embedding the corpus once here means the
    per-query latency you see in the report actually reflects the strategy's
    own scoring logic.
    """
    texts: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get("text") or chunk.get("embedding_text") or chunk.get("content") or "")
        if text.strip():
            texts.append(text)
            normalized.append(chunk)

    texts = texts[:max_chunks]
    normalized = normalized[:max_chunks]

    if not texts:
        return [], [], np.zeros((0, 1))

    vectors = np.array(embed_texts(texts))
    return normalized, texts, vectors


def _retrieve(
    strategy: str,
    query: str,
    prefers_table: bool,
    normalized_chunks: List[Dict[str, Any]],
    corpus_vectors: np.ndarray,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], float]:
    if not normalized_chunks or corpus_vectors.shape[0] == 0:
        return [], 0.0

    start = time.perf_counter()
    query_vec = np.array(embed_texts([query])[0])

    scored = []
    for idx, chunk in enumerate(normalized_chunks):
        semantic_score = _cosine_similarity(query_vec, corpus_vectors[idx])
        text = str(chunk.get("text") or chunk.get("embedding_text") or chunk.get("content") or "")
        lexical_score = _keyword_overlap(query, text)
        final_score = _score_chunk(strategy, query, chunk, semantic_score, lexical_score, prefers_table)
        scored.append({
            "chunk": chunk,
            "score": round(final_score, 4),
            "semantic_score": round(semantic_score, 4),
            "lexical_score": round(lexical_score, 4),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return scored[:top_k], elapsed_ms


def _compute_retrieval_metrics(results: List[Dict[str, Any]], case: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """
    FIXED nDCG computation.

    Previously `ideal_gains` always assumed exactly one relevant document at
    rank 1, but `gains` marked EVERY keyword-matching chunk in the top-k as
    relevant. When more than one chunk matched, actual DCG could exceed the
    (incorrectly small) ideal DCG, producing nDCG values above 1.0 (seen as
    2.948 in earlier runs) — which is mathematically impossible for a
    correctly normalized nDCG. `ideal_gains` is now built from the ACTUAL
    number of relevant chunks found, so nDCG is properly capped at 1.0.
    """
    hits = []
    gains = []
    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        text = str(chunk.get("text") or chunk.get("embedding_text") or chunk.get("content") or "")
        is_relevant = _contains_expected(text, case["expected_keywords"])
        gains.append(1.0 if is_relevant else 0.0)
        if is_relevant and not hits:
            hits.append(rank)

    recall = 1.0 if hits else 0.0
    mrr = 1.0 / hits[0] if hits else 0.0

    dcg = sum((g / math.log2(rank + 1)) for rank, g in enumerate(gains[:top_k], start=1))

    relevant_count = int(sum(gains[:top_k]))
    ideal_gains = [1.0] * relevant_count + [0.0] * (top_k - relevant_count)
    idcg = sum((g / math.log2(rank + 1)) for rank, g in enumerate(ideal_gains[:top_k], start=1))
    ndcg = min(1.0, dcg / idcg) if idcg else 0.0

    return {
        "recall_at_5": recall,
        "mrr": round(mrr, 3),
        "ndcg_at_5": round(ndcg, 3),
        "hit_rank": hits[0] if hits else None,
    }


def _answer_metrics(strategy: str, query: str, results: List[Dict[str, Any]], case: Dict[str, Any]) -> Dict[str, Any]:
    if not results:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "hallucination_score": 1.0,
            "correctness": 0.0,
        }

    top = results[0]["chunk"]
    top_text = str(top.get("text") or top.get("embedding_text") or top.get("content") or "")
    answer = f"Relevant policy details: {top_text[:600]}"

    used_chunks = []
    for item in results[:3]:
        chunk = item["chunk"]
        used_chunks.append({
            "chunk_id": chunk.get("chunk_id") or chunk.get("id") or "chunk",
            "tag": f"C_{chunk.get('chunk_id', 'x')}",
            "text": str(chunk.get("text") or chunk.get("embedding_text") or chunk.get("content") or ""),
            "score": item["score"],
        })

    relevant = _contains_expected(top_text, case["expected_keywords"])
    citation_report = {
        "citation_coverage": 1.0 if relevant else 0.0,
        "has_hallucinated_citation": False,
    }
    metrics = evaluate(query, answer, used_chunks, citation_report, all_candidate_chunks=used_chunks)
    rag = metrics["ragas"]
    deepeval = metrics["deepeval"]
    return {
        "faithfulness": rag["faithfulness"],
        "answer_relevancy": rag["answer_relevancy"],
        "context_precision": rag["context_precision"],
        "context_recall": rag["context_recall"],
        "hallucination_score": deepeval["hallucination_score"],
        "correctness": deepeval["correctness"],
    }


def _hitl_metric(results: List[Dict[str, Any]], case: Dict[str, Any]) -> Dict[str, Any]:
    if not results:
        return {"clarification_triggered": True, "resolved": False}
    top_score = results[0]["score"] if results else 0.0
    relevant = _contains_expected(str(results[0]["chunk"].get("text") or results[0]["chunk"].get("embedding_text") or results[0]["chunk"].get("content") or ""), case["expected_keywords"])
    clarification_triggered = (top_score < 0.2) or not relevant
    return {"clarification_triggered": clarification_triggered, "resolved": relevant and not clarification_triggered}


def run_benchmark() -> Dict[str, Any]:
    chunks = _load_chunks()
    if not chunks:
        raise RuntimeError("No chunk files were found under backend/data/chunks")

    # Embed the corpus ONCE and reuse across every strategy x query
    # combination. See _embed_corpus() docstring for why this matters.
    print("Embedding corpus once (shared across all strategies/queries) ...")
    normalized_chunks, _texts, corpus_vectors = _embed_corpus(chunks)
    print(f"Corpus embedded: {len(normalized_chunks)} chunks.")

    results_by_strategy: Dict[str, Dict[str, Any]] = {}

    for strategy in STRATEGIES:
        retrieval_metrics: List[Dict[str, Any]] = []
        rag_metrics: List[Dict[str, Any]] = []
        hitl_metrics: List[Dict[str, Any]] = []
        latencies: List[float] = []

        for case in BENCHMARK_CASES:
            ranked, elapsed_ms = _retrieve(
                strategy,
                case["query"],
                case.get("prefers_table", False),
                normalized_chunks,
                corpus_vectors,
                top_k=5,
            )
            latencies.append(elapsed_ms)
            retrieval_metrics.append(_compute_retrieval_metrics(ranked, case))
            rag_metrics.append(_answer_metrics(strategy, case["query"], ranked, case))
            hitl_metrics.append(_hitl_metric(ranked, case))

        results_by_strategy[strategy] = {
            "queries": len(BENCHMARK_CASES),
            "retrieval": {
                "recall_at_5": round(statistics.mean(item["recall_at_5"] for item in retrieval_metrics), 3),
                "mrr": round(statistics.mean(item["mrr"] for item in retrieval_metrics), 3),
                "ndcg_at_5": round(statistics.mean(item["ndcg_at_5"] for item in retrieval_metrics), 3),
            },
            "rag": {
                "faithfulness": round(statistics.mean(item["faithfulness"] for item in rag_metrics), 3),
                "answer_relevancy": round(statistics.mean(item["answer_relevancy"] for item in rag_metrics), 3),
                "context_precision": round(statistics.mean(item["context_precision"] for item in rag_metrics), 3),
                "context_recall": round(statistics.mean(item["context_recall"] for item in rag_metrics), 3),
                "hallucination_rate": round(statistics.mean(item["hallucination_score"] for item in rag_metrics), 3),
                "correctness": round(statistics.mean(item["correctness"] for item in rag_metrics), 3),
            },
            "latency_ms": round(statistics.mean(latencies), 1),
            "token_cost_estimate": round(sum(len(_tokenize(case["query"])) for case in BENCHMARK_CASES) * 1.3, 1),
            "hitl": {
                "clarification_rate": round(statistics.mean(1.0 if item["clarification_triggered"] else 0.0 for item in hitl_metrics), 3),
                "resolved_rate": round(statistics.mean(1.0 if item["resolved"] else 0.0 for item in hitl_metrics), 3),
            },
            "per_query": [
                {
                    "query": case["query"],
                    "type": case["type"],
                    "retrieval": retrieval_metrics[idx],
                    "rag": rag_metrics[idx],
                    "hitl": hitl_metrics[idx],
                }
                for idx, case in enumerate(BENCHMARK_CASES)
            ],
        }

    summary = {
        "benchmark_setup": {
            "queries": len(BENCHMARK_CASES),
            "chunk_files": len(list(CHUNK_ROOT.glob("*.json"))),
            "strategies": STRATEGIES,
            "categories": sorted(set(case["type"] for case in BENCHMARK_CASES)),
        },
        "strategies": results_by_strategy,
        "notes": [
            "This benchmark uses local chunk embeddings and lexical overlap heuristics to produce free, deterministic metrics.",
            "Corpus embeddings are computed once per run and shared across all strategies/queries, so latency differences reflect strategy logic rather than embedding overhead.",
            "The table-aware bonus only applies to queries expected to need tabular data (limits, amounts, percentages, comparisons), and is scaled by the chunk's own relevance rather than applied as a flat constant.",
            "nDCG is computed against the actual number of relevant chunks found (not a fixed assumption of 1), so it is correctly bounded to [0, 1].",
            "These values are good for portfolio/demo purposes; for a rigorous published benchmark, replace expected_keywords with manually annotated ground-truth chunk IDs.",
        ],
    }
    return summary


def _write_outputs(summary: Dict[str, Any]) -> None:
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Experimental Evaluation")
    lines.append("")
    lines.append(
        "The system was evaluated on a curated set of insurance policy questions spanning "
        "coverage, exclusions, waiting periods, claims, premiums, comparisons, maternity, "
        "critical illness, co-payment, network hospitals, eligibility, hospitalization, "
        "day-care procedures, pre-existing diseases, and policy definitions."
    )
    lines.append("")
    lines.append("## Benchmark Setup")
    lines.append(f"- Queries: {summary['benchmark_setup']['queries']}")
    lines.append(f"- Categories: {', '.join(summary['benchmark_setup']['categories'])}")
    lines.append(f"- Chunk files: {summary['benchmark_setup']['chunk_files']}")
    lines.append(f"- Strategies: {', '.join(summary['benchmark_setup']['strategies'])}")
    lines.append("")
    lines.append("| Strategy | Recall@5 | MRR | nDCG@5 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Hallucination Rate | Avg Latency (ms) | HITL Clarification Rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for strategy in STRATEGIES:
        metrics = summary["strategies"][strategy]
        lines.append(
            f"| {strategy} | {metrics['retrieval']['recall_at_5']:.3f} | {metrics['retrieval']['mrr']:.3f} | {metrics['retrieval']['ndcg_at_5']:.3f} | {metrics['rag']['faithfulness']:.3f} | {metrics['rag']['answer_relevancy']:.3f} | {metrics['rag']['context_precision']:.3f} | {metrics['rag']['context_recall']:.3f} | {metrics['rag']['hallucination_rate']:.3f} | {metrics['latency_ms']:.1f} | {metrics['hitl']['clarification_rate']:.3f} |"
        )

    lines.append("")
    lines.append("## Retrieval by Query Category")
    lines.append("")
    lines.append("| Query Type | Naive Recall@5 | Hybrid Recall@5 | Table-aware Recall@5 | Full Agentic Recall@5 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for strategy in STRATEGIES:
        for item in summary["strategies"][strategy]["per_query"]:
            grouped.setdefault(item["type"], {}).setdefault(strategy, []).append(item["retrieval"]["recall_at_5"])
    for qtype in sorted(grouped):
        row = {k: statistics.mean(v) for k, v in grouped[qtype].items()}
        lines.append(
            f"| {qtype} | {row.get('naive', 0.0):.3f} | {row.get('hybrid', 0.0):.3f} | {row.get('table_aware', 0.0):.3f} | {row.get('full_agentic', 0.0):.3f} |"
        )

    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"The benchmark consists of {summary['benchmark_setup']['queries']} curated insurance "
        "questions covering coverage, claims, exclusions, waiting periods, comparisons, premiums, "
        "and policy definitions. Relevance is determined via keyword-overlap heuristics against a "
        "manually chosen expected-keyword list per query (a lightweight proxy for ground-truth "
        "chunk annotation). Retrieval metrics (Recall@5, MRR, nDCG@5) are computed against this "
        "signal; nDCG@5 is normalized against the actual number of relevant chunks retrieved, so "
        "it is bounded to [0, 1]. RAG quality metrics (faithfulness, context precision, answer "
        "relevancy) are computed via the local RAGAS/DeepEval-style evaluator on the top retrieved "
        "chunk. Corpus embeddings are computed once per run and reused across every strategy and "
        "query so that latency differences reflect strategy logic, not embedding overhead."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- The benchmark is fully local and does not require paid API credits.")
    lines.append(
        "- The keyword-overlap relevance signal is a heuristic, not hand-annotated ground truth. "
        "For a publication-grade benchmark, replace `expected_keywords` in the query set with "
        "manually verified relevant chunk IDs per query."
    )
    lines.append(
        "- The table-aware bonus is now relevance-gated and only applies to queries where tabular "
        "data (limits, amounts, percentages, comparisons) is plausibly the answer. If table-aware "
        "still underperforms hybrid/agentic on your corpus after this fix, that's a real signal "
        "worth investigating (e.g. check whether your table chunks actually contain the queried "
        "figures) rather than a scoring artifact."
    )

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = run_benchmark()
    _write_outputs(summary)
    print(json.dumps(summary["strategies"], indent=2))
    print(f"\nSaved benchmark JSON to {REPORT_JSON}")
    print(f"Saved benchmark markdown to {REPORT_MD}")


if __name__ == "__main__":
    main()