"""
Claim Eligibility Agent
=======================
Determines whether a specific claim is eligible under the policy.

Outputs:
  - eligible: yes / no / partial / unclear
  - coverage_amount: exact figure or range
  - deductible / co-payment
  - waiting_period status
  - sub-limits that apply
  - confidence score
  - reason for decision

Design:
  - Focuses on DECISION MAKING (not just extraction)
  - Structured prompt forces yes/no/partial/unclear answer
  - Merges retrieval confidence with LLM reasoning confidence
  - Partial coverage is explicitly handled (important for insurance)
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.agents_claude.llm_client import get_client, extract_json
from backend.agents_claude.graph_state import GraphState


MAX_CONTEXT_CHARS = 3500
MAX_CHUNKS        = 6

SYSTEM_PROMPT = """You are a senior insurance claim eligibility assessor.

TASK: Based on the policy excerpts, determine if the described claim is eligible.

RULES:
1. Answer ONLY from the provided policy text.
2. Give an eligibility verdict: "yes", "no", "partial", or "unclear".
3. "partial" = covered but with limits, sub-limits, or co-payment.
4. "unclear" = insufficient policy text to decide with confidence.
5. Extract exact monetary limits (Rs. amounts, percentages).
6. Flag any waiting periods that must be satisfied.
7. Flag co-payment requirements explicitly.
8. Return ONLY valid JSON.

OUTPUT FORMAT:
{
  "eligible": "yes",
  "verdict_reason": "Hospitalization expenses are explicitly covered under Section 3.",
  "coverage_amount": "Up to Rs. 5,00,000 per year",
  "deductible": "Rs. 5,000 per hospitalization",
  "co_payment": "10% of claim amount",
  "waiting_period": {
    "applicable": true,
    "duration": "24 months",
    "satisfied": "unknown"
  },
  "sub_limits": ["ICU: Rs. 5,000/day", "Room rent: Rs. 2,000/day"],
  "conditions": ["Requires 24-hour hospitalization", "Pre-authorization required"],
  "confidence": 0.85,
  "missing_info": []
}

ELIGIBILITY DECISION GUIDE:
  - "yes"     → Explicitly covered, no ambiguity
  - "no"      → Explicitly excluded or not covered
  - "partial" → Covered with co-pay, sub-limit, or waiting period
  - "unclear" → Policy text is ambiguous or missing
"""


def _build_context(chunks: List[Dict]) -> str:
    parts = []
    total = 0
    for i, chunk in enumerate(chunks[:MAX_CHUNKS]):
        text  = chunk.get("text", "").strip()
        ctype = chunk.get("chunk_type", "text")
        hint  = chunk.get("context_hint", "")
        pg    = chunk.get("page_start", "?")
        score = chunk.get("score", 0.0)
        if not text:
            continue
        label = f"[{i+1}] [{ctype.upper()}] [page:{pg}] [score:{score:.2f}]"
        if hint:
            label += f" [{hint}]"
        entry = f"{label}\n{text}"
        if total + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)


class ClaimEligibilityAgent:
    def __init__(self):
        self.llm = get_client()

    def run(self, state: GraphState) -> Dict[str, Any]:
        chunks = state.get("retrieved_chunks", [])
        query  = state.get("normalized_query") or state.get("raw_query", "")

        # Pull policy analysis if already done (avoids redundant context)
        prior_analysis = state.get("policy_analysis", {})

        if not chunks:
            return {
                "claim_result": {
                    "eligible": "unclear",
                    "verdict_reason": "No policy documents found.",
                    "confidence": 0.0,
                }
            }

        context = _build_context(chunks)

        # Enrich context with prior policy analysis summary if available
        prior_summary = ""
        if prior_analysis.get("answer_summary"):
            prior_summary = f"\nPRIOR ANALYSIS SUMMARY: {prior_analysis['answer_summary']}\n"

        user_message = f"""CLAIM QUERY: {query}
{prior_summary}
POLICY EXCERPTS:
{context}

Assess claim eligibility based on the above policy text. Return JSON only."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        result = self.llm.call_json(messages, max_tokens=512)

        if not result:
            return {
                "claim_result": {
                    "eligible": "unclear",
                    "verdict_reason": "Analysis failed — insufficient LLM output.",
                    "confidence": 0.0,
                }
            }

        # Blend confidence
        retrieval_conf  = state.get("confidence", 1.0)
        llm_conf        = float(result.get("confidence", 0.7))
        result["confidence"] = round(0.35 * retrieval_conf + 0.65 * llm_conf, 3)

        return {
            "claim_result": result,
            "confidence": result["confidence"],
        }


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

_agent = None

def claim_eligibility_node(state: GraphState) -> GraphState:
    global _agent
    if _agent is None:
        _agent = ClaimEligibilityAgent()
    updates = _agent.run(state)
    return {**state, **updates}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.agents.claim_eligibility_agent
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    MOCK_CHUNKS = [
        {
            "chunk_id": "c1",
            "text": (
                "In-patient Hospitalization: The policy covers hospitalization expenses "
                "for illness or injury for a minimum of 24 consecutive hours. "
                "Room rent is limited to Rs. 3,000 per day. "
                "ICU charges are limited to Rs. 6,000 per day. "
                "A co-payment of 10% applies for non-network hospitals."
            ),
            "chunk_type": "clause",
            "score": 0.88,
            "page_start": 4,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c2",
            "text": (
                "Pre-existing Diseases: Diseases or conditions that the insured had "
                "before the policy start date are covered after 48 months of continuous coverage. "
                "Diabetes, hypertension, and thyroid disorders are classified as pre-existing diseases."
            ),
            "chunk_type": "clause",
            "score": 0.82,
            "page_start": 6,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c3",
            "text": (
                "Exclusions: The following are not payable: "
                "1. Treatment for obesity or weight control. "
                "2. Infertility treatment. "
                "3. Dental procedures unless arising from accidental injury. "
                "4. Vision correction surgeries such as LASIK."
            ),
            "chunk_type": "exclusion",
            "score": 0.95,
            "page_start": 11,
            "context_hint": "high_priority_exclusion",
        },
        {
            "chunk_id": "c4",
            "text": (
                "Day Care Procedures: The policy covers 586 day care procedures that "
                "do not require 24-hour hospitalization due to advancement in medical technology. "
                "Cataract surgery and chemotherapy are included in the list."
            ),
            "chunk_type": "table_row",
            "score": 0.79,
            "page_start": 9,
            "context_hint": "table_context",
        },
    ]

    agent = ClaimEligibilityAgent()

    TEST_CASES = [
        {
            "normalized_query": "Is hospitalization for diabetes covered?",
            "intent": "claim",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.85,
            "policy_analysis": {"answer_summary": "Pre-existing diseases covered after 48 months."},
        },
        {
            "normalized_query": "Can I claim for LASIK eye surgery?",
            "intent": "claim",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.90,
            "policy_analysis": {},
        },
        {
            "normalized_query": "Is cataract surgery covered without full hospitalization?",
            "intent": "claim",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.80,
            "policy_analysis": {},
        },
    ]

    print("\n" + "═" * 65)
    print("  CLAIM ELIGIBILITY AGENT — TEST RUN")
    print("═" * 65)

    for tc in TEST_CASES:
        print(f"\n  QUERY      : {tc['normalized_query']}")
        result = agent.run(tc)
        cr = result.get("claim_result", {})
        print(f"  ELIGIBLE   : {cr.get('eligible', '?')}")
        print(f"  REASON     : {cr.get('verdict_reason', '?')}")
        print(f"  COVERAGE   : {cr.get('coverage_amount', 'N/A')}")
        print(f"  CO-PAYMENT : {cr.get('co_payment', 'N/A')}")
        print(f"  WAITING    : {cr.get('waiting_period', 'N/A')}")
        if cr.get("sub_limits"):
            print(f"  SUB-LIMITS : {cr['sub_limits']}")
        print(f"  CONFIDENCE : {cr.get('confidence', '?')}")
        print("  " + "-" * 58)

    print("\n✅ Claim Eligibility Agent tests complete.")