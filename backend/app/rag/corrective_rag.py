"""
Corrective RAG (CRAG)
=======================
Bottom branch of the Advanced RAG Pipeline:

    If Faithfulness Low -> Corrective RAG -> Retrieve Again -> Regenerate

Purpose
-------
When response_validator.py fails an answer, this module decides HOW to
retry rather than just blindly re-running the same query:

  - low_relevance        -> the retrieval query itself was probably too
                             narrow/garbled -> REWRITE the query (broaden it)
                             and increase top_k.
  - low_faithfulness /
    hallucinated_citation -> the context probably WAS okay but the LLM
                             over-reached -> keep the query, but widen top_k
                             so context_compressor has more grounding
                             material, and tighten the generation prompt.
  - no_supporting_chunks  -> corpus likely doesn't cover this -> rewrite
                             query toward more generic/synonym phrasing
                             once, then give up gracefully rather than loop.

Setup
-----
No new dependencies. Optionally uses the same llm_client as
answer_generator.py for smarter query rewrites; falls back to a free
heuristic rewrite (synonym/keyword broadening) if unavailable.

Bounded by MAX_CRAG_ATTEMPTS so this can never loop forever.

Run standalone:
    python -m backend.app.rag.corrective_rag
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Callable, Optional
import difflib

try:
    from backend.agents_claude.llm_client import get_client
except Exception:
    get_client = None  # type: ignore

MAX_CRAG_ATTEMPTS = 2

# Free, no-LLM synonym broadening for the most common insurance terms —
# used as the heuristic fallback query rewrite.
SYNONYM_MAP = {
    "waiting period": ["wait time", "cooling period", "moratorium period"],
    "cover": ["coverage", "benefit", "included", "payable"],
    "exclude": ["exclusion", "not covered", "denied"],
    "premium": ["price", "cost", "amount payable"],
    "claim": ["reimbursement", "settlement"],
}


def _heuristic_rewrite(query: str, reasons: List[str]) -> str:
    q = query
    lower = query.lower()
    for term, synonyms in SYNONYM_MAP.items():
        if term in lower:
            q = f"{query} {synonyms[0]}"
            break
    else:
        # No known term matched — broaden generically by stripping question
        # words that can over-constrain BM25/vector matching.
        q = re.sub(r"\b(what is|please tell me|can you|i want to know)\b", "", query, flags=re.I).strip()
        if q == query:
            q = f"{query} policy details"
    return q


def _llm_rewrite(client: Any, query: str, reasons: List[str]) -> Optional[str]:
    prompt = (
        f"The following search query returned weak/unfaithful results from an "
        f"insurance policy document corpus. Reasons: {reasons}.\n\n"
        f"Original query: \"{query}\"\n\n"
        "Rewrite it as a single broader/clearer search query that is more "
        "likely to retrieve the relevant policy clause. Return ONLY the "
        "rewritten query text, nothing else."
    )
    try:
        if hasattr(client, "call"):
            rewritten = client.call([{"role": "user", "content": prompt}], max_tokens=60)
            if rewritten and rewritten.strip():
                return rewritten.strip().strip('"')
    except Exception as e:
        print(f"⚠️  CRAG LLM rewrite failed, using heuristic: {e}")
    return None


def _is_in_domain(client: Optional[Any], query: str) -> bool:
    """Return True if the question appears to be about insurance policies.
    Prefer the LLM-based detector when available; otherwise use a simple
    heuristic blacklist of obviously out-of-domain tokens (alien, dragon,
    zombie, etc.)."""
    if client is None:
        try:
            if get_client is not None:
                client = get_client()
        except Exception:
            client = None

    if client is not None and hasattr(client, "call"):
        prompt = (
            "Return JSON: {\"in_domain\": true|false} \n"
            f"Question: {query}\n\n"
            "Does this question belong to the insurance policy domain?"
        )
        try:
            resp = client.call([{"role": "user", "content": prompt}], max_tokens=30)
            txt = (resp or "").lower()
            if "false" in txt or "no" in txt or "out of domain" in txt:
                return False
            if "true" in txt or "yes" in txt or "in_domain" in txt:
                return True
        except Exception:
            pass

    # Heuristic fallback: blacklist obvious fantasy/fiction terms.
    blacklist = [
        "alien", "zombie", "dragon", "spaceship", "martian", "ghost",
        "teleport", "magic", "nuclear war", "superpower",
    ]
    q = query.lower()
    for tok in blacklist:
        if tok in q:
            return False
    return True


def rewrite_query(query: str, reasons: List[str], llm_client: Optional[Any] = None) -> Dict[str, str]:
    client = llm_client
    if client is None and get_client is not None:
        try:
            client = get_client()
        except Exception:
            client = None
    if client is not None:
        rewritten = _llm_rewrite(client, query, reasons)

        if rewritten:
            return {
                "query": rewritten,
                "method": "llm",
            }

    return {
        "query": _heuristic_rewrite(query, reasons),
        "method": "heuristic",
    }


def run_crag_loop(
    query: str,
    retrieve_fn: Callable[[str, int], List[Dict[str, Any]]],
    generate_and_validate_fn: Callable[[str, List[Dict[str, Any]]], Dict[str, Any]],
    initial_top_k: int = 8,
    max_attempts: int = MAX_CRAG_ATTEMPTS,
) -> Dict[str, Any]:
    """
    Generic CRAG loop, decoupled from any specific retrieval/generation
    implementation so it can be unit-tested with mocks and reused unchanged
    inside the orchestrator graph.

    retrieve_fn(query, top_k) -> List[chunk dicts]
    generate_and_validate_fn(query, chunks) -> {
        "answer": ..., "reflection": ..., "validation": {"passed": bool, "reasons": [...], ...}, ...
    }

    Returns the LAST attempt's full result dict, with an added
    "crag_attempts" trail describing what happened at each retry.
    """

    trail = []
    top_k = initial_top_k
    current_query = query
    result = None
    rewritten_once = False
    rewrite_method = None

    for attempt in range(1, max_attempts + 2):  # +1 initial pass, +max_attempts retries
        chunks = retrieve_fn(current_query, top_k)
        result = generate_and_validate_fn(current_query, chunks)
        validation = result.get("validation", {})

        trail.append({
            "attempt": attempt,
            "query_used": current_query,
            "top_k": top_k,
            "chunks_retrieved": len(chunks),
            "passed": validation.get("passed", False),
            "faithfulness_score": validation.get("faithfulness_score"),
            "retry_reason": validation.get("reasons", []),
            "query_rewritten": rewritten_once,
            "rewrite_method": rewrite_method,
        })

        reasons = validation.get("reasons", [])

        # If validation passed or we've exhausted attempts, stop.
        faith = validation.get("faithfulness_score", 0.0)
        relevance = validation.get("relevance_score", 1.0)

# HARD STOP ONLY FOR TRULY BAD ANSWERS
        if (
            validation.get("passed")
            or (faith < 0.3 and relevance < 0.2)
            or attempt > max_attempts
):
            break
        

        # Only retry retrieval for a specific set of failure reasons.
        # Partial answers should trigger regeneration, not retrieval.
        retryable_prefixes = (
            "low_relevance",
            "low_faithfulness",
            "hallucinated_citation",
            "no_supporting_chunks",
            "low_coverage",
            "retrieval_low_confidence",
        )
        if not any(str(r).startswith(p) for p in retryable_prefixes for r in reasons):
            # Nothing worth retrying at the retrieval layer — stop here.
            break

        # Corrective step: decide whether to widen top_k or rewrite
        reasons = validation.get("reasons", [])
        if any(r.startswith("low_relevance") for r in reasons):
            top_k = min(top_k + 8, 30)

        elif any(r.startswith("no_supporting_chunks") for r in reasons):
            top_k = min(top_k + 8, 30)

        elif any(r.startswith(("low_faithfulness", "hallucinated_citation")) for r in reasons):
            top_k = min(top_k + 2, 30)

        # Only attempt a rewrite once, and only for relevance / no_supporting_chunks
        if not rewritten_once and any(r.startswith(("low_relevance", "no_supporting_chunks")) for r in reasons):
            # Check if the question is in-domain first
            client = None
            if get_client is not None:
                try:
                    client = get_client()
                except Exception:
                    client = None

            if not _is_in_domain(client, current_query):
                # Out-of-domain — do not attempt rewrite; return a graceful
                # message indicating no relevant clause exists. Use a
                # user-friendly explanation referencing the original query.
                result = result or {}
                result.setdefault("validation", {})
                result["validation"]["passed"] = False
                result["validation"]["faithfulness_score"] = 0.0
                result["validation"]["reasons"] = result["validation"].get("reasons", []) + ["out_of_domain"]
                result.setdefault("answer", f"The retrieved insurance policy documents do not contain any clause related to '{current_query}'. Therefore I cannot determine coverage from the provided documents.")
                trail.append({
                    "attempt": attempt + 1,
                    "query_used": current_query,
                    "top_k": top_k,
                    "chunks_retrieved": len(chunks),
                    "passed": False,
                    "faithfulness_score": 0.0,
                    "retry_reason": ["out_of_domain"],
                    "query_rewritten": False,
                    "rewrite_method": None,
                })
                break

            rewrite_result = rewrite_query(current_query, reasons)

            # Prevent question drift: reject rewrites that are semantically
            # too different from the original query.
            rewritten_text = rewrite_result.get("query")
            try:
                sim = difflib.SequenceMatcher(None, current_query.lower(), rewritten_text.lower()).ratio()
            except Exception:
                sim = 0.0

            if sim < 0.65:
                # Reject the rewrite to avoid intent drift; instead broaden top_k
                rewrite_method = "rejected_low_similarity"
                rewritten_once = True
                top_k = min(top_k + 8, 30)
            else:
                current_query = rewritten_text
                rewrite_method = rewrite_result["method"]
                rewritten_once = True

    if result is None:
        result = {}

    result["original_query"] = query
    result["final_query"] = current_query
    result["crag_attempts"] = trail
    result["crag_triggered"] = len(trail) > 1
    return result


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json

    CORPUS = {
        "good": [{"chunk_id": "abc", "tag": "C_abc12345", "text": "Waiting period for pre-existing diseases is 48 months."}],
        "bad": [{"chunk_id": "xyz", "tag": "C_xyz98765", "text": "Hospital network list is available on our website."}],
    }

    def mock_retrieve(q, top_k):
        q = q.lower()
        if "waitng perod" in q:
            return CORPUS["bad"]
        if (
            "waiting period" in q
            or "pre-existing" in q
            or "insurance policy" in q
        ):
            return CORPUS["good"]
        return CORPUS["bad"]

    def mock_generate_and_validate(q, chunks):
        if not chunks:
            answer = "No information found."
            passed, faith, reasons = False, 0.0, ["no_supporting_chunks"]
        else:
            text = chunks[0]["text"]
            tag = chunks[0]["tag"]
            if "Waiting period" in text:
                answer = f"{text} [{tag}]"
                passed, faith, reasons = True, 0.95, []
            else:
                answer = f"{text} [{tag}]"
                passed, faith, reasons = False, 0.4, ["low_relevance (0.1 < 0.25)"]

        return {
            "answer": answer,
            "used_chunks": chunks,
            "validation": {"passed": passed, "faithfulness_score": faith, "reasons": reasons},
        }

    result = run_crag_loop(
        query="waht is waitng perod for pre existing",
        retrieve_fn=mock_retrieve,
        generate_and_validate_fn=mock_generate_and_validate,
    )
    print(json.dumps(result, indent=2))
