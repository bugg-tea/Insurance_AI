"""
Graph State — Single source of truth for the entire LangGraph pipeline.

Design goals:
  - All agents READ from and WRITE to this object.
  - Immutable snapshots: agents return dicts that LangGraph merges.
  - Keep it flat where possible to avoid deep-copy overhead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # ── INPUT ────────────────────────────────────────────────────────────────
    raw_query: str                     # original user text
    normalized_query: str              # after query normalization LLM
    intent: str                        # coverage | exclusion | claim | comparison | waiting_period | general
    is_comparison: bool                # True → Comparison Agent takes over
    needs_retrieval: bool

    # ── RETRIEVAL ────────────────────────────────────────────────────────────
    retrieved_chunks: List[Dict]       # from HybridSearch + reranker
    retrieval_score: float             # avg top-3 score (used for confidence routing)

    # ── POLICY ANALYSIS ──────────────────────────────────────────────────────
    policy_analysis: Dict              # clauses, coverage_found, exclusions_found
    compared_policies: List[Dict]      # for comparison queries

    # ── CLAIM ELIGIBILITY ────────────────────────────────────────────────────
    claim_result: Dict                 # eligible, coverage_amount, deductible, waiting_period, confidence

    # ── RISK ANALYSIS ────────────────────────────────────────────────────────
    risk_result: Dict                  # hidden_exclusions, co_payment, risk_score, flags

    # ── RECOMMENDATION ───────────────────────────────────────────────────────
    recommendation: Dict               # best_policy, pros, cons, alternatives

    # ── REPORT ───────────────────────────────────────────────────────────────
    final_report: Dict                 # markdown, json, confidence, sources

    # ── ROUTING / CONTROL ────────────────────────────────────────────────────
    confidence: float                  # 0.0–1.0
    needs_human_review: bool
    follow_up_question: Optional[str]  # if agent needs clarification
    error: Optional[str]
    retry_count: int

    # ── SESSION ──────────────────────────────────────────────────────────────
    session_id: str
    user_id: str
    conversation_history: List[Dict]   # [{role, content}]