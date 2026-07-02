"""
Comparison Agent
================
Compares two or more insurance policies on key dimensions:
  - Coverage scope
  - Premium vs value
  - Waiting periods
  - Exclusions
  - Sub-limits
  - Network hospitals
  - Claim settlement ratio (if in documents)

Design:
  - Single LLM call with structured comparison table output
  - Works even if only 1 policy is loaded (compares against general knowledge)
  - Produces a winner recommendation per category
  - Low-latency: tight context, strict JSON
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from backend.agents_claude.llm_client import get_client, extract_json
from backend.agents_claude.graph_state import GraphState
from backend.agents_claude.policy_builder import PolicyBuilder



MAX_CONTEXT_CHARS = 4500
MAX_CHUNKS        = 8

SYSTEM_PROMPT = """You are an expert insurance policy comparator.

TASK: Compare the insurance policies described in the provided text.

RULES:
1. Compare ONLY on information present in the text.
2. For missing data, use "not mentioned" — do NOT guess.
3. Give a clear winner per category.
4. Produce a final recommendation with reasons.
5. Return ONLY valid JSON.
IMPORTANT RULE (CRITICAL):
You MUST return "comparison_table" with EXACTLY 6 objects.

If information is missing, still create rows using "not mentioned".

You are NOT allowed to return empty or partial comparison_table.

If you fail, output is invalid.
OUTPUT FORMAT:
{
  "policies_compared": ["Policy A", "Policy B"],
  "comparison_table": [
    {
      "category": "Coverage Limit",
      "policy_a": "Rs. 5,00,000",
      "policy_b": "Rs. 10,00,000",
      "winner": "Policy B",
      "note": "Policy B offers double the coverage"
    },
    {
      "category": "Waiting Period (PED)",
      "policy_a": "48 months",
      "policy_b": "36 months",
      "winner": "Policy B",
      "note": "Shorter waiting period is better"
    }
  ],
  "pros_cons": {
    "policy_a": {
      "pros": [],
      "cons": []
    },
    "policy_b": {
      "pros": [],
      "cons": []
    }
  },
  "recommendation": {
    "best_overall": "Policy B",
    "best_for_family": "Policy A",
    "best_for_seniors": "Policy B",
    "reasoning": "..."
  },
  "confidence": 0.80,
  "missing_data": []
}

EXAMPLE (YOU MUST FOLLOW STRUCTURE EXACTLY):

"comparison_table": [
  {
    "category": "Coverage Limit",
    "policy_a": "Rs. 5,00,000",
    "policy_b": "Rs. 10,00,000",
    "winner": "Policy B",
    "note": "Policy B has higher limit"
  },
  {
    "category": "Room Rent Sub-limit",
    "policy_a": "not mentioned",
    "policy_b": "1% of sum insured",
    "winner": "Policy A",
    "note": "Policy A has no restriction"
  },
  {
    "category": "Waiting Period (PED)",
    "policy_a": "36 months",
    "policy_b": "48 months",
    "winner": "Policy A",
    "note": "Shorter is better"
  },
  {
    "category": "Co-payment",
    "policy_a": "NIL",
    "policy_b": "10%",
    "winner": "Policy A",
    "note": "Lower is better"
  },
  {
    "category": "Network Hospitals",
    "policy_a": "13,000+",
    "policy_b": "14,000+",
    "winner": "Policy B",
    "note": "Higher network is better"
  },
  {
    "category": "Exclusions Count",
    "policy_a": "not mentioned",
    "policy_b": "not mentioned",
    "winner": "Tie",
    "note": "Cannot determine exactly"
  }
]

COMPARISON CATEGORIES (always include these if data available):
  Coverage Limit, Room Rent Sub-limit, ICU Charges, Waiting Period (PED),
  Waiting Period (Specific Diseases), Co-payment, Network Hospitals,
  Day Care Procedures, Maternity Benefit, No-Claim Bonus, Premium Range,
  Claim Settlement Ratio, Exclusions Count.
"""


def _build_context(chunks: List[Dict]) -> str:
    parts = []
    total = 0

    # Group by policy/company if metadata available
    policy_groups: Dict[str, List[str]] = {}
    ungrouped = []

    for chunk in chunks[:MAX_CHUNKS]:
        text  = chunk.get("text", "").strip()
        if not text:
            continue
        company = chunk.get("company", chunk.get("metadata", {}).get("company", ""))
        ctype   = chunk.get("chunk_type", "text")
        pg      = chunk.get("page_start", "?")
        score   = chunk.get("score", 0.0)
        label   = f"[{ctype.upper()}] [page:{pg}] [score:{score:.2f}]"
        entry   = f"{label}\n{text}"

        if company:
            policy_groups.setdefault(company, []).append(entry)
        else:
            ungrouped.append(entry)

    # Render grouped
    for company, entries in policy_groups.items():
        header = f"=== POLICY: {company} ==="
        block  = header + "\n" + "\n\n---\n".join(entries)
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    # Render ungrouped
    if ungrouped:
        block = "=== POLICY EXCERPTS ===\n" + "\n\n---\n".join(ungrouped)
        if total + len(block) <= MAX_CONTEXT_CHARS:
            parts.append(block)

    return "\n\n" + ("═" * 50) + "\n\n".join(parts)


class ComparisonAgent:
    def __init__(self):
        self.llm = get_client()

    def run(self, state: GraphState) -> Dict[str, Any]:
        chunks   = state.get("retrieved_chunks", [])
        query    = state.get("normalized_query") or state.get("raw_query", "")
        entities = state.get("entities", {})
        companies = entities.get("companies", [])

        if not chunks:
            return {
                "compared_policies": [],
                "final_report": {
                    "answer_summary": "No policy documents uploaded for comparison.",
                    "confidence": 0.0,
                }
            }

        builder = PolicyBuilder()
        structured_policies = builder.build(chunks)
        context = structured_policies
        company_hint = ""
        if companies:
            company_hint = f"POLICIES TO COMPARE: {' vs '.join(companies)}\n"
        user_message = f"""
COMPARISON QUERY: {query}

STRUCTURED POLICY DATA (DO NOT TREAT AS TEXT):
{json.dumps(context, indent=2)}

INSTRUCTIONS:
- Compare using ONLY structured fields
- If value is "not mentioned", keep it
- DO NOT hallucinate missing values
- Always fill all 6 categories
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        result = self.llm.call_json(messages, max_tokens=768)
        
        if not isinstance(result, dict):
            result = {}

        table = result.get("comparison_table")

        if not isinstance(table, list) or len(table) < 6:
            result = {
                "policies_compared": companies or ["Policy A", "Policy B"],
                "comparison_table": _fallback_table(structured_policies, companies),
                "recommendation": {
                    "best_overall": "not mentioned",
                    "reasoning": "LLM output incomplete"
                },
                "pros_cons": {},
                "confidence": 0.5,
            }
        
        result.setdefault("pros_cons", {})
        result.setdefault("confidence", 0.7)
        
        if not isinstance(result["comparison_table"], list):
            result["comparison_table"] = []
            
        retrieval_conf = state.get("confidence", 1.0)
        
        llm_conf = result.get("confidence", 0.7)
        if llm_conf is None:
            llm_conf = 0.7
        
        result["confidence"] = round(
    0.4 * retrieval_conf + 0.6 * llm_conf,
    3
)

        

        return {
            "compared_policies": result.get("policies_compared", companies),
            "policy_analysis": result,   # Reuse for report generation
            "confidence": result["confidence"],
        }


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

_agent = None

def comparison_node(state: GraphState) -> GraphState:
    global _agent
    if _agent is None:
        _agent = ComparisonAgent()
    updates = _agent.run(state)
    return {**state, **updates}

def _fallback_table(policies, companies):
    categories = [
        "coverage_limit",
        "room_rent",
        "icu_charges",
        "ped_waiting",
        "co_payment",
        "network_hospitals"
    ]

    rows = []
    left_policy = companies[0] if len(companies) > 0 else "Policy A"
    right_policy = companies[1] if len(companies) > 1 else "Policy B"

    for cat in categories:
        a = policies.get(left_policy, {}).get(cat, "not mentioned")
        b = policies.get(right_policy, {}).get(cat, "not mentioned")

        rows.append({
            "category": cat,
            "policy_a": a,
            "policy_b": b,
            "winner": "Tie",
            "note": "Fallback comparison (LLM failed)"
        })

    return rows
# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.agents.comparison_agent
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    # Mock chunks simulating TWO different policy documents
    MOCK_CHUNKS = [
        # HDFC ERGO
        {
            "chunk_id": "h1",
            "text": (
                "HDFC ERGO Optima Secure: Sum insured options Rs. 5L to Rs. 2Cr. "
                "No room rent sub-limit for SI above Rs. 10L. "
                "PED waiting period: 36 months. "
                "Co-payment: NIL for all ages. "
                "Network: 13,000+ hospitals."
            ),
            "chunk_type": "table_row",
            "score": 0.88,
            "page_start": 2,
            "company": "HDFC ERGO",
            "context_hint": "table_context",
        },
        {
            "chunk_id": "h2",
            "text": (
                "HDFC ERGO Optima Secure Exclusions: Dental treatment (unless emergency), "
                "Obesity treatment, Cosmetic surgery, Experimental treatments, "
                "War and nuclear risks."
            ),
            "chunk_type": "exclusion",
            "score": 0.90,
            "page_start": 8,
            "company": "HDFC ERGO",
            "context_hint": "high_priority_exclusion",
        },
        # Star Health
        {
            "chunk_id": "s1",
            "text": (
                "Star Comprehensive Insurance Policy: Sum insured Rs. 5L to Rs. 1Cr. "
                "Room rent sub-limit: 1% of sum insured per day. "
                "PED waiting period: 48 months. "
                "Co-payment: 10% for age above 60 years. "
                "Network: 14,000+ hospitals."
            ),
            "chunk_type": "table_row",
            "score": 0.85,
            "page_start": 2,
            "company": "Star Health",
            "context_hint": "table_context",
        },
        {
            "chunk_id": "s2",
            "text": (
                "Star Health Exclusions: Psychiatric disorders, Infertility treatment, "
                "HIV/AIDS, Congenital conditions, Cosmetic surgery, "
                "Naturopathy and alternative medicine unless listed."
            ),
            "chunk_type": "exclusion",
            "score": 0.87,
            "page_start": 9,
            "company": "Star Health",
            "context_hint": "high_priority_exclusion",
        },
        {
            "chunk_id": "s3",
            "text": (
                "Star Comprehensive: Maternity benefit included from year 3. "
                "New-born baby covered from day 1. "
                "Cataract surgery: Rs. 40,000 per eye sub-limit. "
                "No-claim bonus: 10% increase in SI per claim-free year, max 50%."
            ),
            "chunk_type": "clause",
            "score": 0.82,
            "page_start": 5,
            "company": "Star Health",
            "context_hint": "legal_clause",
        },
    ]

    agent = ComparisonAgent()

    TEST_CASES = [
        {
            "raw_query": "Compare HDFC ERGO and Star Health insurance",
            "normalized_query": "Compare HDFC ERGO and Star Health Insurance.",
            "intent": "comparison",
            "is_comparison": True,
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.85,
            "entities": {"companies": ["HDFC ERGO", "Star Health"], "conditions": [], "procedures": []},
        },
        {
            "raw_query": "Which is better for senior citizens?",
            "normalized_query": "Which health insurance policy is better for senior citizens?",
            "intent": "comparison",
            "is_comparison": True,
            "retrieved_chunks": MOCK_CHUNKS,
            "confidence": 0.80,
            "entities": {"companies": ["HDFC ERGO", "Star Health"], "conditions": [], "procedures": []},
        },
    ]

    print("\n" + "═" * 65)
    print("  COMPARISON AGENT — TEST RUN")
    print("═" * 65)

    for tc in TEST_CASES:
        print(f"\n  QUERY: {tc['raw_query']}")
        result = agent.run(tc)
        pa = result.get("policy_analysis", {})
        print(f"  POLICIES  : {result.get('compared_policies')}")
        table = pa.get("comparison_table", [])
        print(f"  CATEGORIES: {len(table)} compared")
        for row in table[:4]:
            print(f"    {row.get('category')}: {row.get('policy_a')} vs {row.get('policy_b')} → Winner: {row.get('winner')}")
        rec = pa.get("recommendation", {})
        if rec:
            print(f"  BEST OVERALL : {rec.get('best_overall')}")
            print(f"  REASONING    : {str(rec.get('reasoning', ''))[:120]}")
        print(f"  CONFIDENCE   : {result.get('confidence')}")
        print("  " + "-" * 58)

    print("\n✅ Comparison Agent tests complete.")