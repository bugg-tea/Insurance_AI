"""
Response Validator
===================
Step 8 of the Advanced RAG Pipeline: ... Self Reflection -> **Response
Validator** -> Citation Checker -> JSON Output.

Purpose
-------
The final gate before an answer is shown to the user:
  1. Schema check — required fields present, correct types.
  2. Faithfulness gate — combines self_rag's relevance + groundedness +
     citation_checker's hallucination flag into one faithfulness score.
  3. Decision — PASS (ship it) or FAIL (route to corrective_rag.py).

CONFIG (tune per your risk tolerance — insurance QA should stay strict):
  FAITHFULNESS_THRESHOLD = 0.55
  RELEVANCE_THRESHOLD    = 0.25

No setup required — pure Python.

Run standalone:
    python -m backend.agents_claude.response_validator
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import re
try:
    from backend.agents_claude.llm_client import get_client
except Exception:
    get_client = None

FAITHFULNESS_THRESHOLD = 0.2
RELEVANCE_THRESHOLD = 0.15

from sentence_transformers import SentenceTransformer
import numpy as np
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model
def embedding_similarity(a, b):
    model = get_model()
    emb = model.encode([a, b])
    return float(
        np.dot(emb[0], emb[1]) /
        (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
    )
def compute_faithfulness(reflection: Dict[str, Any]) -> float:
    """
    Blends Self-RAG groundedness with citation-checker's hard hallucination
    flag. A single fabricated source tag drags faithfulness down hard,
    regardless of how good the rest of the answer looks.
    """
    groundedness = float(reflection.get("groundedness", 0.0))
    citation_report = reflection.get("citation_report", {}) or {}
    hallucinated = bool(citation_report.get("has_hallucinated_citation", False))

    score = groundedness
    if hallucinated:
        score *= 0.3  # heavy penalty — never trust an answer citing a fake source

    return round(max(0.0, min(1.0, score)), 3)


def _compute_completeness_llm(client: Any, query: str, answer: str) -> Optional[float]:
    prompt = (
        "Return JSON: {\"completeness\": float 0.0-1.0 }\n"
        f"Question: {query}\nAnswer: {answer}\n\n"
        "Does the answer address every part of the question? Return only the JSON."
    )
    try:
        if hasattr(client, "call_json"):
            out = client.call_json([{"role": "user", "content": prompt}], max_tokens=40)
            if isinstance(out, dict) and "completeness" in out:
                return float(out["completeness"])
        elif hasattr(client, "call"):
            txt = client.call([{"role": "user", "content": prompt}], max_tokens=40)
            # try to extract a number
            m = re.search(r"([0-9]*\.?[0-9]+)", txt or "")
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None


def _heuristic_completeness(query: str, answer: str, used_chunks: list) -> float:
    if not answer or not used_chunks:
        return 0.0

    context = " ".join(c.get("text", "") for c in used_chunks)

    # answer vs context similarity
    sim_answer_context = embedding_similarity(answer, context)

    # bonus: answer must align with query intent
    sim_query_answer = embedding_similarity(query, answer)

    # weighted score
    score = (0.7 * sim_answer_context) + (0.3 * sim_query_answer)

    return round(min(1.0, score), 3)
def clean_text(text: str) -> str:
    # remove citation tags like [C_xxx]
    return re.sub(r"\[C_[a-zA-Z0-9_]+\]", "", text)



def validate(
    query: str,
    answer: str,
    reflection: Dict[str, Any],
    used_chunks: list,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "passed": bool,
          "faithfulness_score": float,
          "relevance_score": float,
          "reasons": [str, ...],         # why it failed, if it did
          "schema_valid": bool,
        }
    """

    reasons = []
    issues: List[Dict[str, Any]] = []
    schema_valid = isinstance(answer, str) and len(answer.strip()) > 3
    
    if not schema_valid:
        reasons.append("malformed_response: missing query or answer text")

    relevance = float(reflection.get("relevance", 0.0))
    faithfulness = compute_faithfulness(reflection)
    context = " ".join(c.get("text", "") for c in used_chunks)

    coverage = embedding_similarity(query, context)
    
    
    

    # Completeness check
    heuristic_completeness = _heuristic_completeness(query, answer, used_chunks)

    completeness_score = None
    client = None
    if get_client is not None:
        try:
            client = get_client()
        except Exception:
            client = None
    if client is not None:
        completeness_score = _compute_completeness_llm(client, query, answer)
    if not used_chunks:
        print("⚠️ WARNING: used_chunks is empty")
    if completeness_score is None:
        completeness_score = heuristic_completeness
    elif completeness_score <= 0.0 and heuristic_completeness > 0.2:
        completeness_score = heuristic_completeness

    if completeness_score > 1.0:
        completeness_score = 1.0
    
    completeness_threshold = 0.001

    if relevance < RELEVANCE_THRESHOLD:
        reasons.append(f"low_relevance ({relevance} < {RELEVANCE_THRESHOLD})")
        issues.append({"type": "low_relevance", "score": relevance, "threshold": RELEVANCE_THRESHOLD})

    if faithfulness < FAITHFULNESS_THRESHOLD:
        reasons.append(f"low_faithfulness ({faithfulness} < {FAITHFULNESS_THRESHOLD})")
        issues.append({"type": "low_faithfulness", "score": faithfulness, "threshold": FAITHFULNESS_THRESHOLD})

    if coverage < 0.0:
        reasons.append(f"low_coverage ({coverage} < 0.0)")
        issues.append({"type": "low_coverage", "score": coverage, "threshold": 0.5})

    # If completeness_score couldn't be determined, default to 1.0 (safe)
   
    
    

    if completeness_score < completeness_threshold:
        reasons.append(f"partial_answer ({completeness_score} < {completeness_threshold})")
        issues.append({"type": "partial_answer", "score": completeness_score, "threshold": completeness_threshold})

    citation_report = reflection.get("citation_report", {}) or {}
    if citation_report.get("has_hallucinated_citation"):
        reasons.append("hallucinated_citation: answer cites a source not in retrieved context")
        issues.append({"type": "hallucinated_citation", "score": 0.0, "threshold": 0.0})

    if not used_chunks:
        reasons.append("no_supporting_chunks: answer generated with empty context")
        issues.append({"type": "no_supporting_chunks", "score": 0.0, "threshold": 1})
  # override CRAG (useless query)
    
    # retrieval low confidence check based on used_chunks scores
    top_score = 0.0
    try:
        top_score = max(float(c.get("score", 0.0)) for c in used_chunks) if used_chunks else 0.0
    except Exception:
        top_score = 0.0
    if top_score < 0.45:
        reasons.append(f"retrieval_low_confidence ({top_score} < 0.45)")
        issues.append({"type": "retrieval_low_confidence", "score": top_score, "threshold": 0.45})
    
    
    
    
    # =========================================================
# CRAG DECISION LAYER (NEW FIX)
# =========================================================

    def _has(reason_list, key: str) -> bool:
        return any(key in str(r) for r in reason_list)

    low_rel = _has(reasons, "low_relevance")
    low_faith = _has(reasons, "low_faithfulness")
    no_chunks = _has(reasons, "no_supporting_chunks")
    halluc = citation_report.get("has_hallucinated_citation", False)

# -----------------------------
# CRAG trigger (prefer retry)
# -----------------------------
    crag_required = (
        low_rel or low_faith or no_chunks or halluc or coverage < 0.1
)

# priority for CRAG strategy
    if low_rel:
        crag_priority = "low_relevance"
    elif no_chunks:
        crag_priority = "no_support"
    elif low_faith or halluc:
        crag_priority = "low_faithfulness"
    else:
        crag_priority = None

# -----------------------------
# VERY IMPORTANT: DROP LOGIC (EXTREMELY STRICT)
# -----------------------------
# Only drop if EVERYTHING is bad (true garbage case)
    drop_immediately = (
        relevance < 0.05 and
        faithfulness < 0.05 and
        coverage < 0.1 and
        not used_chunks
)
    passed = (
        schema_valid
        and faithfulness >= FAITHFULNESS_THRESHOLD
        and relevance >= RELEVANCE_THRESHOLD
        
        and completeness_score >= completeness_threshold
        and coverage >= 0.1
)
    top_score = 0.0

    return {
        "passed": passed,
        "faithfulness_score": faithfulness,
        "relevance_score": relevance,
        "completeness_score": completeness_score,
        "coverage": round((coverage), 3),
        "reasons": reasons,
        "issues": issues,
        "schema_valid": schema_valid,
        "crag_required": crag_required,
"crag_priority": crag_priority,
"drop_immediately": drop_immediately
        
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json
    from backend.app.rag.self_rag import reflect

    used_chunks = [
        {"chunk_id": "abc12345", "tag": "C_abc12345", "text": "Waiting period for pre-existing diseases is 48 months."},
    ]
    query = "What is the waiting period for pre-existing diseases?"

    print("--- PASS case ---")
    ans = "The waiting period for pre-existing diseases is 48 months. [C_abc12345]"
    refl = reflect(query, used_chunks[0]["text"], ans, used_chunks)
    print(json.dumps(validate(query, ans, refl, used_chunks), indent=2))

    print("\n--- FAIL case (hallucinated source) ---")
    ans2 = "The waiting period is 48 months. [C_zzz99999]"
    refl2 = reflect(query, used_chunks[0]["text"], ans2, used_chunks)
    print(json.dumps(validate(query, ans2, refl2, used_chunks), indent=2))

    print("\n--- FAIL case (empty context) ---")
    ans3 = "I don't know."
    refl3 = reflect(query, "", ans3, [])
    print(json.dumps(validate(query, ans3, refl3, []), indent=2))