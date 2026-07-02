"""
Report Generator Agent
======================
Final stage of the pipeline. Compiles all agent outputs into a
structured, user-facing report in multiple formats:
  - Markdown (default, fastest)
  - JSON (for API consumers / frontend)
  - PDF-ready Markdown (for download)

Design:
  - NO extra LLM call if all data is present (just formats existing data)
  - LLM call only used for "narrative summary" section
  - Output is deterministic and reproducible
  - Confidence displayed clearly with color-coded labels
  - Sources cited with page numbers
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from backend.agents_claude.llm_client import get_client, extract_json
from backend.agents_claude.graph_state import GraphState


# ── CONFIDENCE LABELS ────────────────────────────────────────────────────────

def _confidence_label(score: float) -> str:
    # User-defined thresholds:
    # - 80% and above: BEST
    # - 60% - 80%: GOOD
    # - >30% - 60%: MEDIUM
    # - <=30%: LOW
    if score >= 0.80:
        return "🟢 BEST"
    if score >= 0.60:
        return "🟡 GOOD"
    if score > 0.30:
        return "🟠 MEDIUM"
    return "🔴 LOW — seek professional advice"


# ── MARKDOWN REPORT BUILDER ───────────────────────────────────────────────────

def _build_markdown_report(state: GraphState, narrative: str) -> str:
    query      = state.get("normalized_query", state.get("raw_query", ""))
    intent     = state.get("intent", "general")
    confidence = state.get("confidence", 0.0)
    timestamp  = datetime.now().strftime("%d %b %Y, %H:%M")

    lines = [
        "# 🏥 Insurance Policy Analysis Report",
        f"**Query:** {query}",
        f"**Intent:** {intent.replace('_', ' ').title()}",
        f"**Confidence:** {_confidence_label(confidence)} ({confidence:.0%})",
        f"**Generated:** {timestamp}",
        "",
        "---",
        "",
    ]

    # ── EXECUTIVE SUMMARY ────────────────────────────────────────────────────
    if narrative:
        lines += ["## 📋 Executive Summary", "", narrative, ""]

    # ── POLICY ANALYSIS ──────────────────────────────────────────────────────
    pa = state.get("policy_analysis", {})
    if pa:
        lines += ["## 📄 Policy Analysis", ""]

        if pa.get("answer_summary"):
            lines += [f"> {pa['answer_summary']}", ""]

        details = pa.get("details", {})

        if details.get("covered_items"):
            lines += ["**✅ Covered:**"]
            for item in details["covered_items"]:
                lines.append(f"- {item}")
            lines.append("")

        if details.get("waiting_periods"):
            lines += ["**⏳ Waiting Periods:**"]
            for wp in details["waiting_periods"]:
                lines.append(f"- {wp}")
            lines.append("")

        if details.get("sub_limits"):
            lines += ["**📊 Sub-limits:**"]
            for sl in details["sub_limits"]:
                lines.append(f"- {sl}")
            lines.append("")

        if pa.get("exclusions_found"):
            lines += ["**❌ Exclusions Found:**"]
            for ex in pa["exclusions_found"]:
                lines.append(f"- {ex}")
            lines.append("")

    # ── CLAIM ELIGIBILITY ────────────────────────────────────────────────────
    cr = state.get("claim_result", {})
    if cr:
        eligible = cr.get("eligible", "unclear")
        emoji = {"yes": "✅", "no": "❌", "partial": "⚠️", "unclear": "❓"}.get(eligible, "❓")

        lines += [
            "## 🧾 Claim Eligibility",
            "",
            f"**Decision:** {emoji} {eligible.upper()}",
            f"**Reason:** {cr.get('verdict_reason', 'N/A')}",
            "",
        ]

        if cr.get("coverage_amount"):
            lines.append(f"**Coverage Amount:** {cr['coverage_amount']}")
        if cr.get("co_payment"):
            lines.append(f"**Co-payment:** ⚠️ {cr['co_payment']}")
        if cr.get("deductible"):
            lines.append(f"**Deductible:** {cr['deductible']}")

        wp = cr.get("waiting_period", {})
        if wp and wp.get("applicable"):
            lines.append(f"**Waiting Period:** {wp.get('duration', 'N/A')}")

        if cr.get("sub_limits"):
            lines += ["", "**Sub-limits that apply:**"]
            for sl in cr["sub_limits"]:
                lines.append(f"- {sl}")

        lines.append("")

    # ── RISK ANALYSIS ────────────────────────────────────────────────────────
    rr = state.get("risk_result", {})
    if rr:
        rs = rr.get("risk_score", 0)
        risk_emoji = "🔴" if rs >= 7 else "🟡" if rs >= 4 else "🟢"

        lines += [
            "## ⚠️ Risk Analysis",
            "",
            f"**Risk Score:** {risk_emoji} {rs}/10",
            f"**Assessment:** {rr.get('risk_summary', '')}",
            "",
        ]

        if rr.get("hidden_exclusions"):
            lines += ["**🚨 Hidden Exclusions:**"]
            for x in rr["hidden_exclusions"]:
                lines.append(f"- **{x.get('item')}**: {x.get('detail', '')}")
            lines.append("")

        if rr.get("fine_print_flags"):
            lines += ["**🔍 Fine Print Warnings:**"]
            for f in rr["fine_print_flags"]:
                lines.append(f"- **{f.get('flag')}**: {f.get('detail', '')}")
            lines.append("")

        if rr.get("co_payment_details"):
            lines += ["**💰 Co-payment Scenarios:**"]
            for cp in rr["co_payment_details"]:
                lines.append(f"- {cp.get('scenario')}: {cp.get('amount')}")
            lines.append("")

    # ── COMPARISON ───────────────────────────────────────────────────────────
    compared = state.get("compared_policies", [])
    if compared and pa.get("comparison_table"):
        lines += ["## ⚖️ Policy Comparison", ""]
        lines.append(f"**Comparing:** {' vs '.join(compared)}")
        lines.append("")
        lines.append("| Category | " + " | ".join(compared) + " | Winner |")
        lines.append("|---|" + "---|" * (len(compared) + 1))
        for row in pa["comparison_table"]:
            policy_vals = [str(row.get(f"policy_{chr(97+i)}", "N/A")) for i in range(len(compared))]
            lines.append(f"| {row.get('category')} | " + " | ".join(policy_vals) + f" | **{row.get('winner')}** |")
        lines.append("")

    # ── RECOMMENDATION ───────────────────────────────────────────────────────
    rec = state.get("recommendation", {})
    if rec:
        lines += ["## 💡 Recommendation", ""]
        lines.append(f"> **{rec.get('recommendation', '')}**")
        lines.append("")

        if rec.get("reasoning"):
            lines += [rec["reasoning"], ""]

        best = rec.get("best_policy", {})
        if best:
            lines += [
                f"**🏆 Best Choice:** {best.get('name', '')}",
                f"**Key Benefit:** {best.get('key_benefit', '')}",
                "",
            ]

        if rec.get("pros"):
            lines += ["**✅ Pros:**"]
            for p in rec["pros"]:
                lines.append(f"- {p}")
            lines.append("")

        if rec.get("cons"):
            lines += ["**❌ Cons:**"]
            for c in rec["cons"]:
                lines.append(f"- {c}")
            lines.append("")

        if rec.get("alternatives"):
            lines += ["**🔄 Alternatives:**"]
            for alt in rec["alternatives"]:
                lines.append(f"- **{alt.get('name')}** — {alt.get('suitable_for')} (tradeoff: {alt.get('key_tradeoff')})")
            lines.append("")

        if rec.get("action_items"):
            lines += ["**📌 Action Items:**"]
            for i, action in enumerate(rec["action_items"], 1):
                lines.append(f"{i}. {action}")
            lines.append("")

        if rec.get("important_warnings"):
            lines += ["**⚠️ Important Warnings:**"]
            for w in rec["important_warnings"]:
                lines.append(f"- {w}")
            lines.append("")

    # ── SOURCES ──────────────────────────────────────────────────────────────
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        lines += ["## 📚 Sources", ""]
        seen_pages = set()
        for c in chunks[:6]:
            pg   = c.get("page_start")
            cid  = c.get("chunk_id", "")
            hint = c.get("context_hint", "")
            if pg and pg not in seen_pages:
                lines.append(f"- Page {pg} [{hint or cid}]")
                seen_pages.add(pg)
        lines.append("")

    # ── FOOTER ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "_This report is auto-generated from policy documents. "
        "For complex claims, consult a certified insurance advisor._",
    ]

    return "\n".join(lines)


# ── NARRATIVE GENERATOR ───────────────────────────────────────────────────────

NARRATIVE_SYSTEM = """You are an insurance advisor writing a brief executive summary.

Write 2-3 sentences summarizing the key finding for the user.
Be direct, specific, and use plain English.
Do NOT use jargon. Do NOT repeat the full analysis.
Return ONLY the narrative text (no JSON, no markdown formatting)."""

def _generate_narrative(state: GraphState, llm) -> str:
    pa  = state.get("policy_analysis", {})
    cr  = state.get("claim_result", {})
    rec = state.get("recommendation", {})
    rr  = state.get("risk_result", {})

    summary_parts = []
    if pa.get("answer_summary"):
        summary_parts.append(f"Coverage finding: {pa['answer_summary']}")
    if cr.get("verdict_reason"):
        summary_parts.append(f"Claim decision: {cr.get('eligible', '').upper()} — {cr['verdict_reason']}")
    if rec.get("recommendation"):
        summary_parts.append(f"Recommendation: {rec['recommendation']}")
    if rr.get("risk_summary"):
        summary_parts.append(f"Risk: {rr['risk_summary']}")

    if not summary_parts:
        return ""

    msg = "\n".join(summary_parts)
    messages = [
        {"role": "system", "content": NARRATIVE_SYSTEM},
        {"role": "user",   "content": f"Summarize this for the user:\n{msg}"},
    ]
    return llm.call(messages, max_tokens=150)


# ── AGENT ─────────────────────────────────────────────────────────────────────

class ReportGeneratorAgent:
    def __init__(self):
        self.llm = get_client()

    def run(self, state: GraphState) -> Dict[str, Any]:
        narrative  = _generate_narrative(state, self.llm)
        markdown   = _build_markdown_report(state, narrative)
        confidence = state.get("confidence", 0.0)

        # Collect source pages
        chunks = state.get("retrieved_chunks", [])
        sources = list({c.get("page_start") for c in chunks if c.get("page_start")})

        json_report = {
            "query":        state.get("normalized_query", state.get("raw_query", "")),
            "intent":       state.get("intent", "general"),
            "confidence":   confidence,
            "narrative":    narrative,
            "policy_analysis": state.get("policy_analysis", {}),
            "claim_result":    state.get("claim_result", {}),
            "risk_result":     state.get("risk_result", {}),
            "recommendation":  state.get("recommendation", {}),
            "synthesized_answer": state.get("synthesized_answer"),
            "synthesis_citations": state.get("synthesis_citations", []),
            "synthesis_sources": state.get("synthesis_sources", []),
            "synthesis_faithfulness": state.get("synthesis_faithfulness"),
            "synthesis_passed": state.get("synthesis_passed"),
            "synthesis_crag_triggered": state.get("synthesis_crag_triggered"),
            "synthesis_evaluation": state.get("synthesis_evaluation"),
            "compared_policies": state.get("compared_policies", []),
            "sources":         sources,
        }

        needs_human = confidence < 0.50

        return {
            "final_report": {
                "markdown":          markdown,
                "json":              json_report,
                "confidence":        confidence,
                "confidence_label":  _confidence_label(confidence),
                "sources":           sources,
            },
            "needs_human_review": needs_human,
        }


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

_agent = None

def report_generator_node(state: GraphState) -> GraphState:
    global _agent
    if _agent is None:
        _agent = ReportGeneratorAgent()
    updates = _agent.run(state)
    return {**state, **updates}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.agents.report_generator
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    agent = ReportGeneratorAgent()

    MOCK_STATE: GraphState = {
        "raw_query": "Is cataract surgery covered in HDFC ERGO?",
        "normalized_query": "Is cataract surgery covered under HDFC ERGO health insurance?",
        "intent": "coverage",
        "is_comparison": False,
        "confidence": 0.82,
        "retrieved_chunks": [
            {"chunk_id": "c1", "text": "Cataract covered after 24 months waiting period.", "chunk_type": "clause", "score": 0.88, "page_start": 5, "context_hint": "legal_clause"},
            {"chunk_id": "c2", "text": "Sub-limit: Rs. 25,000 per eye for cataract surgery.", "chunk_type": "table_row", "score": 0.80, "page_start": 9, "context_hint": "table_context"},
        ],
        "policy_analysis": {
            "coverage_found": True,
            "answer_summary": "Cataract surgery is covered after a 24-month waiting period with a sub-limit of Rs. 25,000 per eye.",
            "details": {
                "covered_items": ["Cataract surgery (day care procedure)"],
                "waiting_periods": ["24 months from policy inception"],
                "sub_limits": ["Rs. 25,000 per eye per year"],
            },
            "exclusions_found": [],
        },
        "claim_result": {
            "eligible": "partial",
            "verdict_reason": "Covered after 24-month waiting period with Rs. 25,000 sub-limit per eye.",
            "coverage_amount": "Rs. 25,000 per eye",
            "co_payment": "NIL (network hospital)",
            "waiting_period": {"applicable": True, "duration": "24 months", "satisfied": "unknown"},
            "sub_limits": ["Rs. 25,000 per eye"],
            "confidence": 0.85,
        },
        "risk_result": {
            "risk_score": 4,
            "risk_summary": "Moderate risk — sub-limit may not cover full surgery cost in metro cities.",
            "hidden_exclusions": [
                {"item": "Bilateral cataract (both eyes same year)", "detail": "Sub-limit applies per eye — Rs. 50,000 total"},
            ],
            "fine_print_flags": [
                {"flag": "Network hospital required", "detail": "Cashless only at network hospitals; others need reimbursement"},
            ],
            "co_payment_details": [],
            "confidence": 0.80,
        },
        "recommendation": {
            "recommendation": "Yes, HDFC ERGO covers cataract surgery but with a Rs. 25,000 per-eye sub-limit after 24 months.",
            "confidence_level": "high",
            "reasoning": "The policy explicitly covers cataract as a day care procedure. The main concern is the Rs. 25,000 sub-limit, which may fall short in private hospitals where cataract surgery can cost Rs. 35,000–50,000.",
            "best_policy": {"name": "HDFC ERGO Optima Secure", "key_benefit": "Day care coverage, no hospitalization required"},
            "pros": ["Covered without full hospitalization", "Cashless at network hospitals"],
            "cons": ["Rs. 25,000 sub-limit may not cover full costs", "24-month waiting period required"],
            "alternatives": [{"name": "Niva Bupa ReAssure", "suitable_for": "Higher sub-limits needed", "key_tradeoff": "Slightly higher premium"}],
            "action_items": [
                "Verify the surgery date is after your 24-month policy anniversary",
                "Choose a network hospital for cashless claim",
                "Get a cost estimate from the hospital before proceeding",
            ],
            "important_warnings": ["Sub-limit of Rs. 25,000 per eye — not per claim"],
            "confidence": 0.82,
        },
        "compared_policies": [],
    }

    print("\n" + "═" * 65)
    print("  REPORT GENERATOR — TEST RUN")
    print("═" * 65)

    result = agent.run(MOCK_STATE)
    report = result.get("final_report", {})

    print(f"\n  CONFIDENCE     : {report.get('confidence_label')}")
    print(f"  NEEDS HUMAN    : {result.get('needs_human_review')}")
    print(f"  SOURCES        : {report.get('sources')}")
    print(f"\n{'─' * 65}")
    print("  MARKDOWN REPORT PREVIEW (first 80 lines):")
    print(f"{'─' * 65}\n")
    md_lines = report.get("markdown", "").split("\n")
    print("\n".join(md_lines[:80]))
    if len(md_lines) > 80:
        print(f"\n  ... [{len(md_lines) - 80} more lines] ...")

    print(f"\n✅ Report Generator test complete.")
    print(f"   Total markdown lines: {len(md_lines)}")