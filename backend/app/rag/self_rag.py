"""
Self-RAG: Self-Reflection
==========================
Step 7 of the Advanced RAG Pipeline: ... LLM -> **Self Reflection (Self-RAG)**
-> Response Validator ...

Purpose
-------
Grades two things, classic Self-RAG style:
  1. ISREL — is the retrieved context relevant to the query at all?
  2. ISSUP — is the generated answer actually supported by that context?

Setup
-----
Prefers the same LLM client as answer_generator.py for a more nuanced
critique. If unavailable, falls back to FREE heuristic scoring:
  - relevance  = query/context keyword (token) overlap ratio
  - groundedness = fraction of answer sentences that carry a valid
    citation tag pointing at a chunk actually present in context
    (delegates to citation_checker.py for the heavy lifting).

No new dependencies required.

Run standalone:
    python -m backend.agents_claude.self_rag
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

try:
    from backend.agents_claude.llm_client import get_client
except Exception:
    get_client = None  # type: ignore

from backend.app.rag.citation_checeker import check_citations

# Optional embedding-based relevance using local sentence-transformers pipeline
try:
    from backend.app.utils.embeddings import embed_texts
    import numpy as np
except Exception:
    embed_texts = None
    np = None


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


# Small stopword list to avoid counting common function words as "coverage"
_STOPWORDS = {
    "what", "is", "the", "a", "an", "do", "does", "this", "that", "are", "is", "of", "and",
    "in", "for", "to", "how", "when", "where", "which",
}


def _heuristic_relevance(query: str, context_block: str) -> float:
    if not context_block.strip():
        return 0.0
    q = _tokenize(query)
    c = _tokenize(context_block)
    if not q:
        return 0.0
    overlap = len(q & c) / len(q)
    return round(min(1.0, overlap), 3)


def _llm_reflection(client: Any, query: str, context_block: str, answer: str) -> Optional[Dict[str, Any]]:
    prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

QUESTION
--------
{query}

CONTEXT
-------
{context_block}

ANSWER
------
{answer}

Evaluate the following:

1. relevance
How relevant is the retrieved CONTEXT to answering the QUESTION?
Ignore whether the ANSWER is correct.

2. groundedness
How well is the ANSWER is supported by the CONTEXT?
If the answer invents facts not found in the context, groundedness should be low.

Return ONLY valid JSON:

{{
    "relevance": 0.0,
    "groundedness": 0.0
}}
"""
        
    
    try:
        if hasattr(client, "call_json"):
            result = client.call_json(
                [{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            if isinstance(result, dict) and "relevance" in result and "groundedness" in result:
                return {
                    "relevance": float(result["relevance"]),
                    "groundedness": float(result["groundedness"]),
                }
    except Exception as e:
        print(f"⚠️  Self-RAG LLM reflection failed, using heuristic: {e}")
    return None


def reflect(
    query: str,
    context_block: str,
    answer: str,
    used_chunks: List[Dict[str, Any]],
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "relevance": float 0-1,       # ISREL
          "groundedness": float 0-1,    # ISSUP
          "citation_report": {...},     # from citation_checker
          "method": "llm" | "heuristic",
        }
    """

    citation_report = check_citations(answer, used_chunks)

    client = llm_client
    if client is None and get_client is not None:
        try:
            client = get_client()
        except Exception:
            client = None

    if client is not None:
        llm_scores = _llm_reflection(client, query, context_block, answer)
        if llm_scores is not None:
            return {
                **llm_scores,
                "citation_report": citation_report,
                "method": "llm",
            }
    # Heuristic path: compute token-overlap relevance, coverage, and
    # optionally an embedding-based relevance if the local embedding
    # pipeline is available.

    relevance = _heuristic_relevance(query, context_block)
    groundedness = citation_report["citation_coverage"]

    # Coverage: fraction of query tokens present somewhere in retrieved chunks
    qtokens = _tokenize(query)
    qtokens = {t for t in qtokens if t not in _STOPWORDS}
    retrieved_tokens = set()
    for c in used_chunks:
        retrieved_tokens.update(_tokenize(c.get("text", "")))
    retrieved_tokens = {t for t in retrieved_tokens if t not in _STOPWORDS}
    coverage = round(len(qtokens & retrieved_tokens) / max(1, len(qtokens)), 3)

    embedding_relevance = None
    if embed_texts is not None and np is not None and qtokens:
        try:
            # Build texts: query + each chunk's embedding_text or text
            q_vec = embed_texts([query])
            chunk_texts = [c.get("embedding_text") or c.get("text") or "" for c in used_chunks]
            if chunk_texts:
                chunk_vecs = embed_texts(chunk_texts)
                # dot product since embed_texts returns L2-normalized vectors
                sims = (chunk_vecs @ q_vec[0]).tolist()
                embedding_relevance = round(float(sum(sims) / len(sims)), 3)
        except Exception as e:
            embedding_relevance = None

    result = {
        "relevance": relevance,
        "groundedness": groundedness,
        "coverage": coverage,
        "citation_report": citation_report,
        "method": "heuristic",
    }
    if embedding_relevance is not None:
        result["embedding_relevance"] = embedding_relevance

    return result


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json

    used_chunks = [
        {"chunk_id": "abc12345", "tag": "C_abc12345", "text": "Waiting period for pre-existing diseases is 48 months."},
        {"chunk_id": "def67890", "tag": "C_def67890", "text": "Maternity benefits have a waiting period of 24 months."},
    ]

    query = "What is the waiting period for pre-existing diseases?"

    good_answer = "The waiting period for pre-existing diseases is 48 months. [C_abc12345]"
    bad_answer = "The waiting period for pre-existing diseases is 12 months and includes free annual checkups."

    for label, ans in [("grounded", good_answer), ("hallucinated/uncited", bad_answer)]:
        result = reflect(query, "\n".join(c["text"] for c in used_chunks), ans, used_chunks)
        print(f"\n--- {label} ---")
        print(json.dumps(result, indent=2))