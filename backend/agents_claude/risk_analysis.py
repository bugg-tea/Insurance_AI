"""
Risk Analysis Agent
===================
Scans policy chunks to surface HIDDEN RISKS — the things that trip up
policyholders at claim time.

Outputs:
  - hidden_exclusions:  things excluded but not obvious
  - fine_print_flags:   important conditions buried in clauses
  - co_payment_details: all co-pay / deductible scenarios
  - sub_limits:         room rent caps, ICU caps, disease-wise limits
  - risk_score:         0–10 (10 = very risky policy for the user)
  - risk_flags:         human-readable list of red flags

Design philosophy:
  - ALWAYS look for exclusion chunks first (boosted by retrieval)
  - Risk score is NOT just about coverage — it's about hidden traps
  - Prompt engineered to be adversarial (finds problems, not benefits)
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.agents_claude.llm_client import get_client, extract_json
from backend.agents_claude.graph_state import GraphState


MAX_CONTEXT_CHARS = 4000
MAX_CHUNKS        = 8   # Risk analysis benefits from more context

SYSTEM_PROMPT = """You are an adversarial insurance policy auditor.

TASK: Find every hidden risk, trap, and exclusion in the provided policy text.

FOCUS ON:
1. Hidden exclusions (things excluded that customers don't expect)
2. Fine print conditions that reduce or block claims
3. Co-payment / deductible traps
4. Sub-limits that make coverage less valuable than advertised
5. Waiting periods that are unusually long
6. Disease-specific exclusions or limits
7. Network hospital restrictions
8. Pre-authorization requirements
9. Proportional deduction clauses (e.g., room rent ratio clause)

RISK SCORE GUIDE (0–10):
  8–10 = Many hidden traps, low effective coverage
  5–7  = Moderate risks, some concerning clauses
  2–4  = Minor issues, reasonable coverage
  0–1  = Very consumer-friendly policy

Return ONLY valid JSON:
{
  "risk_score": 6,
  "risk_summary": "One sentence overall risk assessment",
  "hidden_exclusions": [
    {"item": "Mental illness treatment", "detail": "Explicitly excluded under Section 4.2"},
    {"item": "Adventure sports injuries", "detail": "Not covered unless add-on rider purchased"}
  ],
  "fine_print_flags": [
    {"flag": "Room rent ratio clause", "detail": "Proportional deduction applies if room rent exceeds limit"},
    {"flag": "24-hour hospitalization required", "detail": "Day surgeries not covered unless listed"}
  ],
  "co_payment_details": [
    {"scenario": "Non-network hospital", "amount": "20% of claim"},
    {"scenario": "Senior citizens (>60 years)", "amount": "30% of claim"}
  ],
  "sub_limits": [
    {"item": "Room rent", "limit": "Rs. 2,000/day or 1% of SI"},
    {"item": "ICU", "limit": "Rs. 4,000/day or 2% of SI"},
    {"item": "Cataract surgery", "limit": "Rs. 25,000 per eye"}
  ],
  "waiting_period_flags": [
    {"item": "Pre-existing diseases", "period": "48 months"},
    {"item": "Specific diseases (hernia, cataract)", "period": "24 months"}
  ],
  "confidence": 0.85,
  "missing_info": []
}
"""


def _build_context(chunks: List[Dict]) -> str:
    """
    Prioritize exclusion chunks — they're most important for risk analysis.
    """
    # Sort: exclusions first, then by score
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            -(1 if c.get("chunk_type") == "exclusion" else 0),
            -c.get("score", 0.0),
        ),
    )

    parts = []
    total = 0
    for i, chunk in enumerate(sorted_chunks[:MAX_CHUNKS]):
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


class RiskAnalysisAgent:
    def __init__(self):
        self.llm = get_client()

    def run(self, state: GraphState) -> Dict[str, Any]:
        chunks = state.get("retrieved_chunks", [])
        query  = state.get("normalized_query") or state.get("raw_query", "")

        if not chunks:
            return {
                "risk_result": {
                    "risk_score": 0,
                    "risk_summary": "No policy documents to analyze.",
                    "confidence": 0.0,
                }
            }

        context = _build_context(chunks)

        # Include prior claim result if available for context
        prior_claim = state.get("claim_result", {})
        claim_note  = ""
        if prior_claim.get("eligible") == "partial":
            claim_note = (
                f"\nNOTE: Claim eligibility was assessed as PARTIAL. "
                f"Reason: {prior_claim.get('verdict_reason', '')}. "
                "Focus risk analysis on conditions limiting this claim.\n"
            )

        user_message = f"""ANALYSIS REQUEST: {query}
{claim_note}
POLICY TEXT:
{context}

Identify all hidden risks, exclusions, and traps. Return JSON only."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        result = self.llm.call_json(messages, max_tokens=768)

        if not result:
            return {
                "risk_result": {
                    "risk_score": 5,
                    "risk_summary": "Risk analysis unavailable — LLM error.",
                    "confidence": 0.0,
                }
            }

        retrieval_conf = state.get("confidence", 1.0)
        llm_conf       = float(result.get("confidence", 0.7))
        result["confidence"] = round(0.4 * retrieval_conf + 0.6 * llm_conf, 3)

        return {
            "risk_result": result,
            "confidence": result["confidence"],
        }


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

_agent = None

def risk_analysis_node(state: GraphState) -> GraphState:
    global _agent
    if _agent is None:
        _agent = RiskAnalysisAgent()
    updates = _agent.run(state)
    return {**state, **updates}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.agents.risk_analysis_agent
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    MOCK_CHUNKS = [
        {
            "chunk_id": "c1",
            "text": (
                "Room Rent Sub-limit: If the insured avails a room with rent higher than "
                "the eligible limit, all other hospital expenses (doctor fees, nursing, OT charges) "
                "will be reduced proportionally. Eligible limit: Rs. 2,000 per day or 1% of sum insured."
            ),
            "chunk_type": "clause",
            "score": 0.85,
            "page_start": 7,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c2",
            "text": (
                "Exclusions (Permanent): The following conditions are permanently excluded: "
                "1. Congenital external diseases or defects. "
                "2. Genetic disorders or diseases. "
                "3. Infertility, sub-fertility, or IVF treatment. "
                "4. Mental illness, psychiatric or psychosomatic disorders. "
                "5. HIV/AIDS and related complications. "
                "6. Obesity treatment including bariatric surgery."
            ),
            "chunk_type": "exclusion",
            "score": 0.95,
            "page_start": 12,
            "context_hint": "high_priority_exclusion",
        },
        {
            "chunk_id": "c3",
            "text": (
                "Co-Payment Clause: A co-payment of 20% per claim applies for: "
                "a) Treatment at non-network hospitals. "
                "b) Insured persons above 60 years of age at the time of claim. "
                "c) Claims arising from adventure sports or hazardous activities."
            ),
            "chunk_type": "clause",
            "score": 0.88,
            "page_start": 10,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c4",
            "text": (
                "Pre-Authorization: Planned hospitalizations require pre-authorization "
                "from the insurer at least 72 hours before admission. "
                "Emergency hospitalizations must be notified within 24 hours of admission. "
                "Failure to notify may result in 20% reduction of admissible claim amount."
            ),
            "chunk_type": "clause",
            "score": 0.80,
            "page_start": 15,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c5",
            "text": (
                "Specific Disease Sub-limits: "
                "Cataract: Rs. 25,000 per eye per year. "
                "Hernia: Rs. 30,000 per occurrence. "
                "Knee replacement: Rs. 1,00,000 per knee. "
                "These limits apply regardless of the sum insured."
            ),
            "chunk_type": "table_row",
            "score": 0.83,
            "page_start": 9,
            "context_hint": "table_context",
        },
    ]

    agent = RiskAnalysisAgent()

    TEST_CASES = [
        {
            "normalized_query": "What are the hidden risks in this policy?",
            "intent": "exclusion",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.87,
            "claim_result": {},
        },
        {
            "normalized_query": "Is knee replacement covered?",
            "intent": "coverage",
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.80,
            "claim_result": {
                "eligible": "partial",
                "verdict_reason": "Covered with sub-limit of Rs. 1,00,000 per knee.",
            },
        },
    ]

    print("\n" + "═" * 65)
    print("  RISK ANALYSIS AGENT — TEST RUN")
    print("═" * 65)

    for tc in TEST_CASES:
        print(f"\n  QUERY        : {tc['normalized_query']}")
        result = agent.run(tc)
        rr = result.get("risk_result", {})
        print(f"  RISK SCORE   : {rr.get('risk_score', '?')}/10")
        print(f"  SUMMARY      : {rr.get('risk_summary', '?')}")
        print(f"  HIDDEN EXCL  : {len(rr.get('hidden_exclusions', []))} found")
        for x in rr.get("hidden_exclusions", [])[:3]:
            print(f"    → {x.get('item')}: {x.get('detail', '')[:60]}")
        print(f"  FINE PRINT   : {len(rr.get('fine_print_flags', []))} flags")
        print(f"  CO-PAYMENT   : {rr.get('co_payment_details', [])}")
        print(f"  SUB-LIMITS   : {len(rr.get('sub_limits', []))} found")
        print(f"  CONFIDENCE   : {rr.get('confidence', '?')}")
        print("  " + "-" * 58)

    print("\n✅ Risk Analysis Agent tests complete.")