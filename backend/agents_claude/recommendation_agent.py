"""
Recommendation Agent
====================
Synthesizes all prior agent outputs and produces a final recommendation:
  - Best policy choice
  - Ranked alternatives
  - Pros / Cons
  - Confidence-weighted reasoning
  - Actionable next steps for the user

Design:
  - Aggregates: policy_analysis + claim_result + risk_result + compared_policies
  - Single summarization LLM call (no extra retrieval)
  - Output is user-facing: plain language, not jargon
  - Low-latency: uses context already in state, no new retrieval
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.agents_claude.llm_client import get_client, extract_json
from backend.agents_claude.graph_state import GraphState


SYSTEM_PROMPT = """You are a trusted independent insurance advisor helping a customer.

TASK: Based on the analysis provided, give a clear, actionable recommendation.

RULES:
1. Be direct — give a clear recommendation, not "it depends" without substance.
2. Explain WHY in plain language a customer can understand.
3. Flag important caveats (waiting periods, exclusions, co-payments).
4. Provide 2-3 specific action items the user should take.
5. If confidence is low, say so honestly and explain what information is missing.
6. Return ONLY valid JSON.

OUTPUT FORMAT:
{
  "recommendation": "Buy HDFC ERGO Optima Secure for this use case.",
  "confidence_level": "high",
  "reasoning": "Clear 2-3 sentence explanation",
  "best_policy": {
    "name": "HDFC ERGO Optima Secure",
    "key_benefit": "No room rent limit, zero co-payment, 36-month PED wait"
  },
  "pros": [
    "No co-payment at any age",
    "No room rent sub-limit",
    "Shorter PED waiting period (36 months vs 48 months)"
  ],
  "cons": [
    "Higher premium compared to basic plans",
    "Adventure sports excluded without add-on"
  ],
  "alternatives": [
    {
      "name": "Star Comprehensive",
      "suitable_for": "Budget-conscious buyers who don't need premium room",
      "key_tradeoff": "Room rent sub-limit and 10% co-pay for age 60+"
    }
  ],
  "action_items": [
    "Compare final premium quotes from both insurers before buying",
    "Check network hospital list includes hospitals near your home",
    "Ask about No-Claim Bonus restoration benefit"
  ],
  "important_warnings": [
    "Pre-existing diabetes will be covered only after 36 months",
    "Mental illness treatment is excluded under both policies reviewed"
  ],
  "confidence": 0.82
}

CONFIDENCE LEVELS:
  "high"    → Full analysis available, clear answer
  "medium"  → Some gaps but reasonable recommendation
  "low"     → Limited data, recommend user seek professional advisor
"""


def _build_analysis_summary(state: GraphState) -> str:
    """Compiles all prior agent outputs into a single analysis block for the LLM."""
    parts = []

    # Query context
    parts.append(f"USER QUERY: {state.get('normalized_query', state.get('raw_query', ''))}")
    parts.append(f"INTENT: {state.get('intent', 'general')}")

    # Policy analysis
    pa = state.get("policy_analysis", {})
    if pa:
        parts.append("\n=== POLICY ANALYSIS ===")
        parts.append(f"Coverage Found: {pa.get('coverage_found')}")
        parts.append(f"Summary: {pa.get('answer_summary', '')}")
        details = pa.get("details", {})
        if details.get("waiting_periods"):
            parts.append(f"Waiting Periods: {details['waiting_periods']}")
        if details.get("covered_items"):
            parts.append(f"Covered Items: {details['covered_items'][:5]}")
        if pa.get("exclusions_found"):
            parts.append(f"Exclusions Found: {pa['exclusions_found'][:5]}")

    # Claim result
    cr = state.get("claim_result", {})
    if cr:
        parts.append("\n=== CLAIM ELIGIBILITY ===")
        parts.append(f"Eligible: {cr.get('eligible')}")
        parts.append(f"Reason: {cr.get('verdict_reason', '')}")
        if cr.get("co_payment"):
            parts.append(f"Co-payment: {cr['co_payment']}")
        if cr.get("sub_limits"):
            parts.append(f"Sub-limits: {cr['sub_limits']}")

    # Risk result
    rr = state.get("risk_result", {})
    if rr:
        parts.append("\n=== RISK ANALYSIS ===")
        parts.append(f"Risk Score: {rr.get('risk_score', 'N/A')}/10")
        parts.append(f"Risk Summary: {rr.get('risk_summary', '')}")
        if rr.get("hidden_exclusions"):
            excl_summary = [x.get("item") for x in rr["hidden_exclusions"][:4]]
            parts.append(f"Hidden Exclusions: {excl_summary}")
        if rr.get("fine_print_flags"):
            flags = [f.get("flag") for f in rr["fine_print_flags"][:3]]
            parts.append(f"Fine Print Flags: {flags}")

    # Comparison
    compared = state.get("compared_policies", [])
    if compared:
        parts.append(f"\n=== COMPARED POLICIES: {compared} ===")
        comparison_table = pa.get("comparison_table", [])
        for row in comparison_table[:6]:
            parts.append(
                f"  {row.get('category')}: "
                f"{row.get('policy_a')} vs {row.get('policy_b')} → {row.get('winner')}"
            )

    return "\n".join(parts)


class RecommendationAgent:
    def __init__(self):
        self.llm = get_client()

    def run(self, state: GraphState) -> Dict[str, Any]:
        analysis = _build_analysis_summary(state)
        confidence = state.get("confidence", 0.7)

        user_message = f"""{analysis}

OVERALL RETRIEVAL CONFIDENCE: {confidence:.2f}

Based on the above analysis, provide a final recommendation. Return JSON only."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        result = self.llm.call_json(messages, max_tokens=640)

        if not result:
            return {
                "recommendation": {
                    "recommendation": "Unable to generate recommendation — analysis data insufficient.",
                    "confidence_level": "low",
                    "confidence": 0.0,
                }
            }

        llm_conf = float(result.get("confidence", 0.7))
        result["confidence"] = round(0.5 * confidence + 0.5 * llm_conf, 3)

        return {"recommendation": result, "confidence": result["confidence"]}


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

_agent = None

def recommendation_node(state: GraphState) -> GraphState:
    global _agent
    if _agent is None:
        _agent = RecommendationAgent()
    updates = _agent.run(state)
    return {**state, **updates}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.agents.recommendation_agent
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    agent = RecommendationAgent()

    TEST_CASES = [
        # Case 1: Single policy, claim question
        {
            "raw_query": "Should I buy this policy for my diabetic father?",
            "normalized_query": "Is this health policy suitable for a person with diabetes?",
            "intent": "coverage",
            "is_comparison": False,
            "confidence": 0.78,
            "policy_analysis": {
                "coverage_found": True,
                "answer_summary": "Diabetes is covered as a pre-existing disease after 48 months waiting period.",
                "details": {
                    "waiting_periods": ["48 months for pre-existing diseases including diabetes"],
                    "covered_items": ["Hospitalization for diabetes complications after waiting period"],
                },
                "exclusions_found": ["Day 1 diabetes complications not covered"],
            },
            "claim_result": {
                "eligible": "partial",
                "verdict_reason": "Covered after 48 months waiting period. Co-payment of 10% for age 60+.",
                "co_payment": "10% for insured above 60 years",
                "sub_limits": [],
            },
            "risk_result": {
                "risk_score": 5,
                "risk_summary": "Moderate risk — long PED waiting period and age-based co-payment.",
                "hidden_exclusions": [
                    {"item": "Diabetic retinopathy", "detail": "May be excluded as PED complication"},
                ],
                "fine_print_flags": [
                    {"flag": "Age-based co-payment", "detail": "10% co-pay kicks in at age 60"},
                ],
            },
            "compared_policies": [],
        },
        # Case 2: Comparison query
        {
            "raw_query": "HDFC ERGO vs Star Health — which is better?",
            "normalized_query": "Compare HDFC ERGO and Star Health Insurance.",
            "intent": "comparison",
            "is_comparison": True,
            "confidence": 0.85,
            "policy_analysis": {
                "coverage_found": True,
                "answer_summary": "HDFC ERGO offers no room rent limit; Star Health has lower premium.",
                "comparison_table": [
                    {"category": "Room Rent", "policy_a": "No limit", "policy_b": "1% SI/day", "winner": "HDFC ERGO"},
                    {"category": "Co-payment", "policy_a": "NIL", "policy_b": "10% age 60+", "winner": "HDFC ERGO"},
                    {"category": "PED Wait", "policy_a": "36 months", "policy_b": "48 months", "winner": "HDFC ERGO"},
                    {"category": "Network", "policy_a": "13,000+", "policy_b": "14,000+", "winner": "Star Health"},
                ],
            },
            "claim_result": {},
            "risk_result": {},
            "compared_policies": ["HDFC ERGO", "Star Health"],
        },
    ]

    print("\n" + "═" * 65)
    print("  RECOMMENDATION AGENT — TEST RUN")
    print("═" * 65)

    for tc in TEST_CASES:
        print(f"\n  QUERY: {tc['raw_query']}")
        result = agent.run(tc)
        rec = result.get("recommendation", {})
        print(f"  RECOMMENDATION   : {rec.get('recommendation', '?')[:100]}")
        print(f"  CONFIDENCE LEVEL : {rec.get('confidence_level', '?')}")
        print(f"  REASONING        : {str(rec.get('reasoning', ''))[:120]}")
        print(f"  PROS             : {rec.get('pros', [])[:2]}")
        print(f"  CONS             : {rec.get('cons', [])[:2]}")
        if rec.get("action_items"):
            print(f"  ACTION ITEMS     : {rec['action_items'][:2]}")
        if rec.get("important_warnings"):
            print(f"  WARNINGS         : {rec['important_warnings'][:2]}")
        print(f"  CONFIDENCE       : {rec.get('confidence', '?')}")
        print("  " + "-" * 58)

    print("\n✅ Recommendation Agent tests complete.")