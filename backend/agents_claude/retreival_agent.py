"""
Retrieval Agent
===============
Wraps the HybridSearch + Reranker pipeline into a LangGraph node.

Responsibilities:
  1. Take normalized_query from state
  2. Run hybrid retrieval (FAISS + BM25 + reranker)
  3. Compute retrieval confidence score
  4. Trigger routing: low score → knowledge graph → follow-up question

Low-latency design:
  - Retrieval pipeline is pre-built once and reused (singleton)
  - top_k=8 default (enough context, not too much for small LLM)
  - Confidence thresholds drive routing without extra LLM calls
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.app.retrieval.final import build_retrieval_pipeline
from backend.agents_claude.graph_state import GraphState
import math

# ── CONFIG ────────────────────────────────────────────────────────────────────

TOP_K                  = 8     # chunks sent to downstream agents
CONFIDENCE_HIGH        = 0.65  # above this → proceed normally
CONFIDENCE_LOW         = 0.30  # below this → ask user for clarification
MIN_CHUNKS_REQUIRED    = 2     # if fewer chunks retrieved, signal low confidence


# ── AGENT ─────────────────────────────────────────────────────────────────────




class RetrievalAgent:
    """
    Retrieves relevant policy chunks for a given (normalized) query.
    Pipeline must be initialized with document chunks before first use.
    """

    def __init__(self, pipeline=None):
        self._pipeline = pipeline  
        # injected at startup

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))
    def set_pipeline(self, chunks: List[Dict]):
        """Call this once after documents are ingested."""
        self._pipeline = build_retrieval_pipeline(chunks)

    # ── MAIN ─────────────────────────────────────────────────────────────────

    def run(self, state: GraphState) -> Dict[str, Any]:
        if not self._pipeline:
            return {
                "retrieved_chunks": [],
                "retrieval_score": 0.0,
                "confidence": 0.0,
                "follow_up_question": "No documents have been uploaded yet. Please upload a policy PDF first.",
            }

        query = state.get("normalized_query") or state.get("raw_query", "")
        if not query:
            return {
                "retrieved_chunks": [],
                "retrieval_score": 0.0,
                "confidence": 0.0,
                "error": "No query provided to retrieval agent.",
            }

        # ── RETRIEVE ─────────────────────────────────────────────────────────
        chunks = self._pipeline.search(query, top_k=TOP_K)

        # ── CONFIDENCE SCORING ───────────────────────────────────────────────
        confidence, retrieval_score = self._compute_confidence(chunks)

        # ── ROUTING SIGNALS ──────────────────────────────────────────────────
        follow_up = None
        if confidence < CONFIDENCE_LOW:
            follow_up = self._generate_follow_up(state, chunks)

        return {
            "retrieved_chunks": chunks,
            "retrieval_score": retrieval_score,
            "confidence": confidence,
            "follow_up_question": follow_up,
        }
    
    # ── CONFIDENCE ───────────────────────────────────────────────────────────
    def normalize_score(self, x: float) -> float:
    # keeps everything stable in 0–1 range
        return max(0.0, min(1.0, x / 6.0))

    def _compute_confidence(self, chunks):
        if not chunks:
            return 0.0, 0.0

        scores = [
            float(
                c.get("final_score",
                c.get("rerank_score",
                c.get("retrieval_score", 0.0)))
        )
            for c in chunks
    ]

        top1 = max(scores)
        avg = sum(scores) / len(scores)

        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        spread = math.sqrt(variance)

        confidence = (
            0.6 * self.normalize_score(top1) +
            0.3 * self.normalize_score(avg) +
            0.1 * self.normalize_score(spread * 2)
    )

        if len(chunks) < MIN_CHUNKS_REQUIRED:
            confidence *= 0.7

        
        return round(confidence, 3), round(avg, 3)  # ── FOLLOW-UP ────────────────────────────────────────────────────────────
    

    def _generate_follow_up(self, state: GraphState, chunks: List[Dict]) -> str:
        intent = state.get("intent", "general")
        entities = state.get("entities", {}) or {}
        companies = entities.get("organizations", [])

        


        if not companies:
            return (
                "I couldn't find enough relevant information. "
                "Could you specify which insurance company or policy you're asking about?"
            )

        if intent == "coverage":
            return (
                f"I found limited information about coverage. "
                f"Could you clarify which specific condition or treatment you're asking about for {', '.join(companies)}?"
            )

        if intent == "exclusion":
            return (
                "Could you clarify which specific exclusion or condition you want to check? "
                "For example: 'pre-existing disease exclusion' or 'maternity exclusion'."
            )

        return (
            "I found limited policy information. "
            "Could you rephrase your question or specify the policy section you need?"
        )


# ── LANGGRAPH NODE ────────────────────────────────────────────────────────────

# Singleton pipeline shared across requests (built at startup)
_agent: Optional[RetrievalAgent] = None


def get_retrieval_agent() -> RetrievalAgent:
    global _agent
    if _agent is None:
        _agent = RetrievalAgent()
    return _agent


def retrieval_node(state: GraphState) -> GraphState:
    agent = get_retrieval_agent()
    updates = agent.run(state)
    return {**state, **updates}


# ── ROUTING CONDITION ─────────────────────────────────────────────────────────

def route_after_retrieval(state: GraphState) -> str:
    """
    Called by LangGraph conditional edge after retrieval_node.
    Returns next node name.
    """
    confidence = state.get("confidence", 0.0)
    follow_up  = state.get("follow_up_question")

    if follow_up and confidence < CONFIDENCE_LOW:
        return "human_review"

    if state.get("intent") == "comparison":
        if state.get("confidence", 0) > 0.1:
            return "comparison_agent"
    if state.get("is_comparison"):
        return "comparison_agent"

    intent = state.get("intent", "general")

    routing = {
        "claim":          "claim_eligibility_agent",
        "coverage":       "policy_analysis_agent",
        "exclusion":      "risk_analysis_agent",
        "waiting_period": "policy_analysis_agent",
        "premium":        "policy_analysis_agent",
        "comparison":     "comparison_agent",
        "definition":     "policy_analysis_agent",
        "general":        "policy_analysis_agent",
    }

    return routing.get(intent, "policy_analysis_agent")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.agents.retrieval_agent
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("\n" + "═" * 65)
    print("  RETRIEVAL AGENT — TEST RUN (with mock chunks)")
    print("═" * 65)

    # ── Mock chunks (replace with real FAISS+BM25 in production) ─────────────
    MOCK_CHUNKS = [
        {
            "chunk_id": "c1",
            "text": "Waiting period for pre-existing diseases is 48 months.",
            "chunk_type": "clause",
            "retrieval_score": 4.8,
            "score": 4.8,
            "table_id": None,
            "row_number": None,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c2",
            "text": "Maternity benefits have a waiting period of 24 months.",
            "chunk_type": "clause",
            "retrieval_score": 4.2,
            "score": 4.2,
            "table_id": "T1",
            "row_number": 3,
            "context_hint": "table_context|legal_clause",
        },
        {
            "chunk_id": "c3",
            "text": "Cataract surgery is covered after a 2-year waiting period.",
            "chunk_type": "clause",
            "retrieval_score": 4.3,
            "score": 4.3,
            "table_id": None,
            "row_number": None,
            "context_hint": "legal_clause",
        },
        {
            "chunk_id": "c4",
            "text": "Exclusions: dental treatment, cosmetic surgery, and war injuries.",
            "chunk_type": "exclusion",
            "retrieval_score": 5.2,
            "score": 5.2,
            "table_id": None,
            "row_number": None,
            "context_hint": "high_priority_exclusion",
        },
    ]

    # Build a mock pipeline that just returns mock chunks
    class MockPipeline:
        def search(self, query: str, top_k: int = 8):
            return MOCK_CHUNKS[:top_k]

    agent = RetrievalAgent(pipeline=MockPipeline())

    TEST_CASES = [
        {
            "raw_query": "What is the waiting period for cataract surgery?",
            "normalized_query": "What is the waiting period for cataract surgery?",
            "intent": "waiting_period",
            "is_comparison": False,
        },
        {
            "raw_query": "What is excluded from this policy?",
            "normalized_query": "What are the exclusions in this policy?",
            "intent": "exclusion",
            "is_comparison": False,
        },
        {
            "raw_query": "Compare HDFC ERGO and Star Health",
            "normalized_query": "Compare HDFC ERGO and Star Health Insurance.",
            "intent": "comparison",
            "is_comparison": True,
        },
    ]

    for tc in TEST_CASES:
        result = agent.run(tc)
        route  = route_after_retrieval({**tc, **result})

        print(f"\n  QUERY      : {tc['raw_query']}")
        print(f"  CHUNKS     : {len(result['retrieved_chunks'])}")
        print(f"  CONFIDENCE : {result['confidence']}")
        print(f"  SCORE      : {result['retrieval_score']}")
        print(f"  ROUTE →    : {route}")
        if result.get("follow_up_question"):
            print(f"  FOLLOW-UP  : {result['follow_up_question']}")
        print("  " + "-" * 58)

    print("\n✅ Retrieval Agent tests complete.")