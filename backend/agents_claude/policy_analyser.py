"""
Policy Analysis Agent
=====================
Reads retrieved chunks and extracts:
  - Coverage details (what is covered, limits, sub-limits)
  - Exclusions mentioned in the chunks
  - Waiting periods
  - Key clauses
  - Confidence in the analysis

Low-latency design:
  - Context capped at 3000 tokens to the LLM (fits in llama-3.1-8b context fast)
  - Strict JSON output — no "let me explain..." preamble
  - Single LLM call per query (no chaining)
  - Chunks are pre-sorted by relevance (reranker already did the work)
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.agents_claude.llm_client import get_client, extract_json
from backend.agents_claude.graph_state import GraphState


# ── CONFIG ────────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 4000   # ~1000 tokens — fast inference
MAX_CHUNKS        = 10
MIN_RELEVANCE_SCORE = 0.35

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert insurance policy analyst.

TASK: Analyze the provided policy excerpts and answer the user query.

RULES:
1. Answer ONLY from the provided policy text. Do NOT hallucinate.
2. Quote exact figures (amounts, percentages, days/months).
3. Mark missing information as "not mentioned in document".
4. Be concise — bullet points preferred over paragraphs.
5. Highlight any conditions or sub-limits.
6. Return ONLY valid JSON.

OUTPUT FORMAT:
{
  "coverage_found": true,
  "answer_summary": "Brief 1-2 sentence answer",
  "details": {
    "covered_items": [],
    "coverage_limits": [],
    "waiting_periods": [],
    "sub_limits": [],
    "conditions": []
  },
  "exclusions_found": [],
  "clauses": [],
  "confidence": 0.85,
  "source_chunks": [],
  "missing_info": []
}

CONFIDENCE GUIDE:
  0.9+ = explicit clear answer in text
  0.7  = inferred from partial text
  0.5  = text exists but ambiguous
  0.3  = guessing / very little evidence
  0.0  = no relevant information found
"""


# ── CONTEXT BUILDER ───────────────────────────────────────────────────────────

def _build_context(chunks: List[Dict]) -> str:
    """
    Builds a compact context string from retrieved chunks.
    Format is optimized for LLM comprehension, not human reading.
    """
    parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks[:MAX_CHUNKS]):
        text = chunk.get("text", "").strip()
        if not text:
            continue

        ctype  = chunk.get("chunk_type", "text")
        hint   = chunk.get("context_hint", "")
        pg     = chunk.get("page_start", "?")
        score = (
    chunk.get("final_score")
    or chunk.get("retrieval_score")
    or chunk.get("score")
    or 0.0
)
        

        label = f"[{i+1}] [{ctype.upper()}] [page:{pg}] [score:{score:.2f}]"
        if hint:
            label += f" [{hint}]"

        entry = f"{label}\n{text}"

        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


# ── AGENT ─────────────────────────────────────────────────────────────────────

class PolicyAnalysisAgent:
    def __init__(self):
        self.llm = get_client()

    def run(self, state: GraphState) -> Dict[str, Any]:
        raw_chunks = state.get("retrieved_chunks", [])
        def get_score(c):
            return float(
        c.get("final_score")
        or c.get("retrieval_score")
        or c.get("score")
        or 0.0
    )   
        filtered = [
            c for c in raw_chunks
            if get_score(c) >= MIN_RELEVANCE_SCORE
]

# sort best-first
        filtered.sort(key=get_score, reverse=True)

# cap context
        chunks = filtered[:MAX_CHUNKS]

        query  = state.get("normalized_query") or state.get("raw_query", "")
        intent = state.get("intent", "general")

        if not chunks:
            return {
                "policy_analysis": {
                    "coverage_found": False,
                    "answer_summary": "No relevant policy sections found.",
                    "details": {},
                    "confidence": 0.0,
                    "missing_info": ["No document chunks retrieved"],
                }
            }

        context = _build_context(chunks)

        user_message = f"""USER QUERY: {query}

INTENT: {intent}

POLICY TEXT:
{context}

Analyze the above policy text and answer the query. Return JSON only."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        result = self.llm.call_json(messages, max_tokens=512)

        if not result:
            return {
                "policy_analysis": {
                    "coverage_found": False,
                    "answer_summary": "Analysis failed — LLM returned no output.",
                    "confidence": 0.0,
                    "details": {},
                }
            }

        # Merge retrieval confidence with LLM confidence
        retrieval_conf  = state.get("confidence", 1.0)
        llm_conf        = float(result.get("confidence", 0.7))
        blended_conf    = round(0.4 * retrieval_conf + 0.6 * llm_conf, 3)

        result["confidence"] = blended_conf

        return {"policy_analysis": result, "confidence": blended_conf}


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

_agent = None

def policy_analysis_node(state: GraphState) -> GraphState:
    global _agent
    if _agent is None:
        _agent = PolicyAnalysisAgent()
    updates = _agent.run(state)
    return {**state, **updates}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.agents.policy_analysis_agent
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    MOCK_CHUNKS = [
        {
            "chunk_id": "c1",
            "text": (
                "Section 3 – Waiting Period: Pre-existing diseases (PED) are covered after "
                "a waiting period of 48 months of continuous policy coverage. "
                "Specific diseases including cataract, hernia, and joint replacement surgeries "
                "are covered after a waiting period of 24 months."
            ),
            "chunk_type": "clause",
            "score": 0.88,
            "page_start": 5,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c2",
            "text": (
                "Maternity Benefit: In-patient hospitalization expenses for delivery and "
                "medically necessary caesarean section are covered up to Rs. 50,000 per delivery. "
                "Waiting period: 24 months from inception of first policy."
            ),
            "chunk_type": "table_row",
            "score": 0.75,
            "page_start": 8,
            "context_hint": "table_context",
        },
        {
            "chunk_id": "c3",
            "text": (
                "Exclusions – The following are not payable under this policy: "
                "1. Dental treatment unless requiring hospitalization. "
                "2. Cosmetic or plastic surgery. "
                "3. Intentional self-inflicted injury. "
                "4. War, invasion, or nuclear perils."
            ),
            "chunk_type": "exclusion",
            "score": 0.92,
            "page_start": 12,
            "context_hint": "high_priority_exclusion",
        },
    ]

    agent = PolicyAnalysisAgent()

    TEST_CASES = [
        {
            "raw_query": "What is the waiting period for cataract surgery?",
            "normalized_query": "What is the waiting period for cataract surgery?",
            "intent": "waiting_period",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.85,
        },
        {
            "raw_query": "What is not covered in this policy?",
            "normalized_query": "What are the exclusions in this policy?",
            "intent": "exclusion",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.90,
        },
        {
            "raw_query": "Is maternity covered?",
            "normalized_query": "Is maternity benefit covered in this policy?",
            "intent": "coverage",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.75,
        },
    ]

    print("\n" + "═" * 65)
    print("  POLICY ANALYSIS AGENT — TEST RUN")
    print("═" * 65)

    for tc in TEST_CASES:
        print(f"\n  QUERY: {tc['raw_query']}")
        result = agent.run(tc)
        analysis = result.get("policy_analysis", {})
        print(f"  COVERAGE FOUND : {analysis.get('coverage_found')}")
        print(f"  ANSWER         : {analysis.get('answer_summary')}")
        print(f"  CONFIDENCE     : {analysis.get('confidence')}")
        details = analysis.get("details", {})
        if details.get("waiting_periods"):
            print(f"  WAITING PERIODS: {details['waiting_periods']}")
        if details.get("covered_items"):
            print(f"  COVERED ITEMS  : {details['covered_items']}")
        if analysis.get("exclusions_found"):
            print(f"  EXCLUSIONS     : {analysis['exclusions_found']}")
        print("  " + "-" * 58)

    print(f"\n✅ Policy Analysis Agent tests complete.")
    print(f"   Cache stats: {agent.llm.cache_stats()}")