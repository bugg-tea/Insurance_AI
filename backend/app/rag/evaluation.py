"""
Evaluation: RAGAS-style + DeepEval-style metrics
===================================================
"RAGAS Evaluation: Faithfulness, Answer Relevancy, Context Precision,
Context Recall, Noise Robustness" + "DeepEval: Hallucination Score, Bias
Score, Correctness"

Setup
-----
Fully free, fully local. No RAGAS / DeepEval package install required —
those libraries typically call out to an LLM-as-judge (often paid). This
module reimplements the same *metric definitions* using:
  - your own citation_checker.py output (ground truth for faithfulness)
  - the same free local embedding model already used for retrieval
    (backend.app.utils.embeddings.embed_texts) for relevancy/semantic
    overlap scoring — reuses your existing model, no new download.
  - keyword/lexical overlap heuristics where no semantic signal is needed.

If the embedding model can't be loaded (e.g. sentence-transformers not
installed), relevancy/semantic metrics fall back to lexical Jaccard
overlap automatically — the pipeline still runs end-to-end for free.

Run standalone:
    python -m backend.agents_claude.evaluation
"""

from __future__ import annotations

import re
import math
from typing import Dict, Any, List

try:
    from backend.app.utils.embeddings import embed_texts as _embed_texts
except Exception:
    _embed_texts = None


# Crude, transparent placeholder bias lexicon — DeepEval's real bias score
# uses an LLM judge; this free heuristic only flags overtly loaded language
# so it should be treated as a smoke-test signal, not a certified metric.
BIAS_FLAG_TERMS = [
    "obviously", "everyone knows", "clearly stupid", "always lies",
    "never trustworthy", "all insurers are scams",
]


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _semantic_or_lexical_similarity(text_a: str, text_b: str) -> float:
    if _embed_texts is not None:
        try:
            vecs = _embed_texts([text_a, text_b])
            return round(max(0.0, min(1.0, float(_cosine(vecs[0], vecs[1])))), 3)
        except Exception:
            pass
    # Free fallback — Jaccard token overlap.
    ta, tb = _tokenize(text_a), _tokenize(text_b)
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 3)


# =========================================================
# RAGAS-STYLE METRICS
# =========================================================

def faithfulness(citation_report: Dict[str, Any]) -> float:
    """Fraction of factual claims grounded in retrieved context, penalized
    hard for any hallucinated citation tag."""
    coverage = float(citation_report.get("citation_coverage", 0.0))
    if citation_report.get("has_hallucinated_citation"):
        coverage *= 0.3
    return round(coverage, 3)


def answer_relevancy(query: str, answer: str) -> float:
    return _semantic_or_lexical_similarity(query, answer)



SIMILARITY_THRESHOLD = 0.30


def context_precision(query: str, used_chunks: List[Dict[str, Any]]) -> float:
    """
    Of the chunks actually used, what fraction are semantically relevant
    to the user's query.
    """
    if not used_chunks:
        return 0.0

    relevant = 0

    for chunk in used_chunks:
        similarity = _semantic_or_lexical_similarity(
            query,
            chunk.get("text", "")
        )

        if similarity >= SIMILARITY_THRESHOLD:
            relevant += 1

    return round(relevant / len(used_chunks), 3)

def context_recall(
    query: str,
    used_chunks: List[Dict[str, Any]],
    all_candidate_chunks: List[Dict[str, Any]],
) -> float:
    """
    Of all retrieved chunks that are semantically relevant to the query,
    what fraction survived context compression and reached the LLM?
    """
    if not all_candidate_chunks:
        return 1.0

    relevant_candidates = []

    for chunk in all_candidate_chunks:
        similarity = _semantic_or_lexical_similarity(
            query,
            chunk.get("text", "")
        )

        if similarity >= SIMILARITY_THRESHOLD:
            relevant_candidates.append(chunk)

    if not relevant_candidates:
        return 1.0

    used_ids = {
        c.get("chunk_id")
        for c in used_chunks
    }

    found = sum(
        1
        for chunk in relevant_candidates
        if chunk.get("chunk_id") in used_ids
    )

    return round(found / len(relevant_candidates), 3)

def noise_robustness(used_chunks: List[Dict[str, Any]], score_floor: float = 0.2) -> float:
    """Fraction of used chunks that scored ABOVE the noise floor — i.e. how
    much of the packed context is signal vs. low-confidence filler."""
    if not used_chunks:
        return 1.0
    above_floor = sum(1 for c in used_chunks if float(c.get("score") or 0.0) >= score_floor)
    return round(above_floor / len(used_chunks), 3)


# =========================================================
# DEEPEVAL-STYLE METRICS
# =========================================================

def hallucination_score(faithfulness_score: float) -> float:
    return round(1.0 - faithfulness_score, 3)


def bias_score(answer: str) -> float:
    """Heuristic only — flags overtly loaded/absolutist language. Returns
    0.0 (no flags) to 1.0 (heavily flagged). NOT a substitute for a proper
    bias audit; documented as a smoke-test signal in the module docstring."""
    lower = (answer or "").lower()
    hits = sum(1 for term in BIAS_FLAG_TERMS if term in lower)
    return round(min(1.0, hits / 3), 3)


def correctness(answer: str, used_chunks: List[Dict[str, Any]]) -> float:
    """Lexical overlap between the answer's content words and the union of
    its supporting chunks — a cheap proxy for 'did the answer actually
    restate what the evidence says, vs. drift off-topic'."""
    if not used_chunks:
        return 0.0
    evidence_text = " ".join(c.get("text", "") for c in used_chunks)
    return _semantic_or_lexical_similarity(answer, evidence_text)


# =========================================================
# AGGREGATE REPORT
# =========================================================

def evaluate(
    query: str,
    answer: str,
    used_chunks: List[Dict[str, Any]],
    
    citation_report: Dict[str, Any],
    all_candidate_chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    all_candidate_chunks = all_candidate_chunks or used_chunks
    f = faithfulness(citation_report)

    return {
        "ragas": {
            "faithfulness": f,
            "answer_relevancy": answer_relevancy(query, answer),
            "context_precision": context_precision(query, used_chunks),
            "context_recall": context_recall(query, used_chunks, all_candidate_chunks),
            "noise_robustness": noise_robustness(used_chunks),
        },
        "deepeval": {
            "hallucination_score": hallucination_score(f),
            "bias_score": bias_score(answer),
            "correctness": correctness(answer, used_chunks),
        },
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json
    from backend.app.rag.citation_checeker import check_citations

    used_chunks = [
        {"chunk_id": "abc", "tag": "C_abc12345", "text": "Waiting period for pre-existing diseases is 48 months.", "score": 0.9},
        {"chunk_id": "irrelevant", "tag": "C_irr00000", "text": "Hospital network list is available online.", "score": 0.1},
    ]
    all_candidate_chunks = [
    {
        "chunk_id": "abc",
        "tag": "C_abc12345",
        "text": "Waiting period for pre-existing diseases is 48 months.",
        "score": 0.90,
    },
    {
        "chunk_id": "xyz",
        "tag": "C_xyz11111",
        "text": "Waiting period may differ by insurer.",
        "score": 0.82,
    },
    {
        "chunk_id": "irrelevant",
        "tag": "C_irr00000",
        "text": "Hospital network list is available online.",
        "score": 0.10,
    },
]
    query = "What is the waiting period for pre-existing diseases?"
    answer = "The waiting period for pre-existing diseases is 48 months. [C_abc12345]"

    report = check_citations(answer, used_chunks)
    result = evaluate(
    query=query,
    answer=answer,
    used_chunks=used_chunks,
    citation_report=report,
    all_candidate_chunks=all_candidate_chunks,
)
    print(json.dumps(result, indent=2))