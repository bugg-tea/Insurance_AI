"""
LangGraph Orchestrator
======================
Wires all agents into a LangGraph StateGraph with:

  1. Query normalization
  2. Retrieval with conditional routing
  3. Intent-based agent branching:
       - coverage/waiting_period/premium/definition → Policy Analysis
       - claim → Claim Eligibility → Risk Analysis
       - exclusion → Risk Analysis → Policy Analysis
       - comparison → Comparison Agent
  4. Recommendation Agent (always runs)
  5. Answer Synthesis Agent (Advanced RAG: compression -> grounded LLM ->
     self-RAG -> validation -> CRAG retries -> free RAGAS/DeepEval eval)
  6. Report Generator (always runs)
  7. Human Review Node (low confidence path)
  8. Retry logic (LLM failure → fallback model → error node)

LOW LATENCY DESIGN:
  - Agents run sequentially only when data dependencies require it
  - No parallel nodes (Groq free tier has rate limits)
  - State is passed by reference in dict merges
  - Each node returns only the keys it modifies
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Literal, List, Any

_pending_human_inputs: Dict[str, Dict[str, Any]] = {}


try:
    from langgraph.graph import StateGraph, END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
from langgraph.types import interrupt, Command
from backend.agents_claude.graph_state import GraphState
from backend.app.retrieval.final import build_retrieval_pipeline
from backend.app.rag.observability import traced_node, summarize_trace
from backend.agents_claude.query_normalizer      import query_normalizer_node
from backend.agents_claude.retreival_agent       import retrieval_node, CONFIDENCE_LOW
from backend.agents_claude.policy_analyser import policy_analysis_node
from backend.agents_claude.claim_eligibilty import claim_eligibility_node
from backend.agents_claude.risk_analysis   import risk_analysis_node
from backend.agents_claude.comparision_agent      import comparison_node
from backend.agents_claude.recommendation_agent  import recommendation_node
from backend.agents_claude.report_generator      import report_generator_node
from backend.app.rag.advanced_rag import run_advanced_rag   # === ADDED ===
import os
import json
from langgraph.checkpoint.memory import MemorySaver


_CHUNKS = None

# (retrieval_node already imported above)

# === REMOVED ===
# The block below used to be here, immediately after the imports:
#
#     def query_normalizer_node(state):
#         return {
#             **state,
#             "normalized_query": state["raw_query"],
#         }
#
# It shadowed the REAL `query_normalizer_node` imported above from
# query_normalizer.py (the LLM-based QueryNormalizerAgent with spelling/OCR
# fixes, intent detection, entity extraction). Every query was silently
# skipping all of that. Deleting this dummy redefinition is the fix — the
# imported function already does the right thing and needs no wrapper.


def safe_retrieval_node(state):
    updates = retrieval_node(state)

    # 🚫 REMOVE PDF DEPENDENCY MESSAGE
    if updates.get("follow_up_question"):
        if "upload a policy pdf" in updates["follow_up_question"].lower():
            updates["follow_up_question"] = None

    return updates


def load_chunks() -> List[Dict[str, Any]]:
    global _CHUNKS

    if _CHUNKS is not None:
        return _CHUNKS

    path = "backend/data/chunks.json"   # or your dataset folder output

    if not os.path.exists(path):
        print("⚠️ No chunks file found. Using empty corpus.")
        _CHUNKS = []
        return _CHUNKS

    with open(path, "r", encoding="utf-8") as f:
        _CHUNKS = json.load(f)

    print(f"✅ Loaded {len(_CHUNKS)} chunks into memory")

    return _CHUNKS


# === ADDED ===
# ── ANSWER SYNTHESIS NODE (Advanced RAG layer) ────────────────────────────────

def answer_synthesis_node(state: GraphState) -> GraphState:
    """
    Advanced RAG layer: context compression -> grounded prompt -> LLM ->
    self-RAG reflection -> response validation -> citation check -> CRAG
    retry loop if faithfulness is low -> free RAGAS/DeepEval-style eval.

    Runs AFTER recommendation_agent so it can ground its answer in whatever
    retrieved_chunks the earlier nodes already pulled, and BEFORE
    report_generator so the markdown/JSON report can surface a grounded,
    cited answer instead of (or alongside) the agent summaries.

    Has its OWN internal CRAG retry loop (corrective_rag.run_crag_loop,
    bounded by MAX_CRAG_ATTEMPTS in corrective_rag.py) — it never needs to
    loop back through the graph itself, so routing stays simple: one node,
    one edge in, one edge out.
    """
    chunks = state.get("retrieved_chunks") or []
    query = state.get("normalized_query") or state.get("raw_query", "")

    if not chunks or not query:
        return {
            **state,
            "synthesized_answer": None,
            "synthesis_skipped_reason": "no_query_or_chunks",
        }

    pipeline = _retrieval_pipeline
    if pipeline is None:
        pipeline = init_retrieval_pipeline()

    if pipeline is None:
        return {
            **state,
            "synthesized_answer": None,
            "synthesis_skipped_reason": "no_retrieval_pipeline_bound",
        }

    result = run_advanced_rag(
        query=query,
        retrieval_pipeline=pipeline,
        session_id=state.get("session_id", "default"),
    )

    return {
        **state,
        "synthesized_answer": result["answer"],
        "synthesis_citations": result["citations"],
        "synthesis_sources": result["sources"],
        "synthesis_faithfulness": result["faithfulness_score"],
        "synthesis_passed": result["passed_validation"],
        "synthesis_crag_triggered": result["crag_triggered"],
        "synthesis_evaluation": result["evaluation"],
    }


# ── SPECIAL NODES ─────────────────────────────────────────────────────────────

MAX_REASK = 5

def human_review_node(state: GraphState):
    retry = state.get("human_retry_count", 0)

    follow_up = state.get(
        "follow_up_question",
        "Could you provide more details?"
    )

    if retry >= MAX_REASK:
        return Command(
            update={
                **state,
                "human_review_exhausted": True,
                "follow_up_question": None,
                "needs_human_review": False,
            },
            goto="policy_analysis_agent",
        )

    session_id = state.get("session_id", "default")
    pending = _pending_human_inputs.get(session_id)
    if pending:
        user_answer = pending.get("response")
        if user_answer is None:
            return Command(
                update={
                    **state,
                    "human_review_exhausted": True,
                    "follow_up_question": None,
                    "needs_human_review": False,
                },
                goto="policy_analysis_agent",
            )
        _pending_human_inputs.pop(session_id, None)
        return Command(
            update={
                **state,
                "raw_query": user_answer,
                "normalized_query": user_answer,
                "intent": None,
                "retrieved_chunks": [],
                "follow_up_question": None,
                "confidence": 0.0,
                "error": None,
                "human_retry_count": retry + 1,
            },
            goto="query_normalizer",
        )

    user_answer = interrupt({
        "question": follow_up,
        "attempt": retry + 1,
        "max_attempts": MAX_REASK,
    })

    if isinstance(user_answer, dict):
        user_answer = user_answer.get("response") or user_answer.get("answer") or user_answer.get("text")

    return Command(
        update={
            **state,
            "raw_query": user_answer,
            "normalized_query": user_answer,
            "intent": None,
            "retrieved_chunks": [],
            "follow_up_question": None,
            "confidence": 0.0,
            "error": None,
            "human_retry_count": retry + 1,
        },
        goto="query_normalizer",
    )
    
    
def error_node(state: GraphState) -> GraphState:
    """Terminal error state after all retries are exhausted."""
    error_msg = state.get("error", "Unknown error occurred.")
    retry_count = state.get("retry_count", 0)

    return {
        **state,
        "final_report": {
            "markdown": (
                f"## ❌ Processing Error\n\n"
                f"Unable to complete analysis after {retry_count} attempts.\n\n"
                f"**Error:** {error_msg}\n\n"
                "Please try again or rephrase your query."
            ),
            "json": {
                "status": "error",
                "error": error_msg,
                "retries": retry_count,
            },
            "confidence": 0.0,
            "confidence_label": "🔴 ERROR",
            "sources": [],
        },
        "needs_human_review": True,
    }


# ── SAFE NODE WRAPPERS ────────────────────────────────────────────────────────
# Catches exceptions and routes to error node via retry logic

MAX_RETRIES = 2

def _safe(node_fn):
    """Wraps a node function with retry + error capture."""
    from langgraph.errors import GraphInterrupt

    def wrapper(state):

        try:
            return node_fn(state)

    # THIS IS IMPORTANT
        except GraphInterrupt:
            raise

        except Exception as e:
            print(f"[NODE ERROR] {node_fn.__name__}: {e}")
            traceback.print_exc()
            raise
    
    wrapper.__name__ = node_fn.__name__
    return wrapper


def _traced_safe(node_fn, node_name: str):
    """Wrap a graph node with observability tracing and exception safety."""
    return _safe(traced_node(node_name)(node_fn))


# ── ROUTING FUNCTIONS ─────────────────────────────────────────────────────────

def route_after_normalization(state: GraphState) -> str:
    """Simple pass-through after normalization — always go to retrieval."""
    if state.get("error"):
        return "error_node"
    return "retrieval_node"


def route_after_policy_analysis(state: GraphState) -> str:
    """After policy analysis, always run risk analysis then recommendation."""
    if state.get("error") and state.get("retry_count", 0) >= MAX_RETRIES:
        return "error_node"
    return "risk_analysis_agent"


def route_after_claim(state: GraphState) -> str:
    if state.get("error") and state.get("retry_count", 0) >= MAX_RETRIES:
        return "error_node"
    return "risk_analysis_agent"


def route_after_risk(state):
    if state.get("error") and state.get("retry_count", 0) >= MAX_RETRIES:
        return "error_node"
    return "recommendation_agent"



def ensure_recommendation_gate(state):
    if state.get("recommendation_done"):
        return "report_generator"
    return "recommendation_agent"
def route_after_comparison(state: GraphState) -> str:
    if state.get("error") and state.get("retry_count", 0) >= MAX_RETRIES:
        return "error_node"
    return "recommendation_agent"


def route_after_recommendation(state: GraphState) -> str:
    if state.get("error") and state.get("retry_count", 0) >= MAX_RETRIES:
        return "error_node"
    return "answer_synthesis_agent"   # === CHANGED === (was "report_generator")


def route_after_synthesis(state: GraphState) -> str:   # === ADDED ===
    if state.get("error") and state.get("retry_count", 0) >= MAX_RETRIES:
        return "error_node"
    return "report_generator"


def route_final(state: GraphState) -> str:
    """After report generation, check if human review needed."""
    if state.get("needs_human_review") and not state.get("final_report", {}).get("markdown"):
        return "human_review_node"
    return END

def route_after_retrieval(state):
    intent = state.get("intent")

    if intent in ["coverage", "waiting_period", "premium", "definition"]:
        return "policy_analysis_agent"

    if intent == "claim":
        return "claim_eligibility_agent"

    if intent == "comparison":
        return "comparison_agent"

    if intent == "exclusion":
        return "policy_analysis_agent"
    
    # For unknown intents, use the configured retrieval confidence threshold
    # to decide whether to ask for human review. Use the canonical
    # `CONFIDENCE_LOW` from the retrieval agent so thresholds stay consistent.
    if intent not in ["coverage", "waiting_period", "premium", "definition", "claim", "comparison", "exclusion"]:
        if state.get("confidence", 1.0) < CONFIDENCE_LOW:
            return "human_review_node"
        return "policy_analysis_agent"

    # If we reach here, prefer human review only when confidence is very low.
    if state.get("confidence", 1.0) < CONFIDENCE_LOW:
        return "human_review_node"
    return "policy_analysis_agent"
# ── GRAPH BUILDER ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Constructs the LangGraph StateGraph.
    Returns a compiled graph ready for .invoke() calls.
    """
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph not installed. Run: pip install langgraph"
        )

    graph = StateGraph(GraphState)

    # ── ADD NODES ─────────────────────────────────────────────────────────────
    graph.add_node("query_normalizer",       _traced_safe(query_normalizer_node, "query_normalizer"))
    graph.add_node("retrieval_node", _traced_safe(safe_retrieval_node, "retrieval_node"))
    graph.add_node("policy_analysis_agent",  _traced_safe(policy_analysis_node, "policy_analysis_agent"))
    graph.add_node("claim_eligibility_agent",_traced_safe(claim_eligibility_node, "claim_eligibility_agent"))
    graph.add_node("risk_analysis_agent",    _traced_safe(risk_analysis_node, "risk_analysis_agent"))
    graph.add_node("comparison_agent",       _traced_safe(comparison_node, "comparison_agent"))
    graph.add_node("recommendation_agent",   _traced_safe(recommendation_node, "recommendation_agent"))
    graph.add_node("answer_synthesis_agent", _traced_safe(answer_synthesis_node, "answer_synthesis_agent"))   # === ADDED ===
    graph.add_node("report_generator",       _traced_safe(report_generator_node, "report_generator"))
    graph.add_node("human_review_node",      _traced_safe(human_review_node, "human_review_node"))
    graph.add_node("error_node",             _traced_safe(error_node, "error_node"))
    

    # ── ENTRY POINT ───────────────────────────────────────────────────────────
    graph.set_entry_point("query_normalizer")

    # ── EDGES ─────────────────────────────────────────────────────────────────

    # Normalization → Retrieval (or error)
    graph.add_conditional_edges(
        "query_normalizer",
        route_after_normalization,
        {
            "retrieval_node": "retrieval_node",
            "error_node":     "error_node",
        },
    )

    # Retrieval → Intent-based branching
    graph.add_conditional_edges(
        "retrieval_node",
        route_after_retrieval,
        {
            "policy_analysis_agent":   "policy_analysis_agent",
            "claim_eligibility_agent": "claim_eligibility_agent",
            "risk_analysis_agent":     "risk_analysis_agent",
            "comparison_agent":        "comparison_agent",
            "human_review_node":       "human_review_node",
        },
    )

    # Policy Analysis → Risk → Recommendation
    graph.add_conditional_edges(
        "policy_analysis_agent",
        route_after_policy_analysis,
        {
            "risk_analysis_agent": "risk_analysis_agent",
            "error_node":          "error_node",
        },
    )

    # Claim → Risk → Recommendation
    graph.add_conditional_edges(
        "claim_eligibility_agent",
        route_after_claim,
        {
            "risk_analysis_agent": "risk_analysis_agent",
            "error_node":          "error_node",
        },
    )

    # Risk → Recommendation
    graph.add_conditional_edges(
        "risk_analysis_agent",
        route_after_risk,
        {
            "recommendation_agent": "recommendation_agent",
            "error_node":           "error_node",
        },
    )

    # Comparison → Recommendation
    graph.add_conditional_edges(
        "comparison_agent",
        route_after_comparison,
        {
            "recommendation_agent": "recommendation_agent",
            "error_node":           "error_node",
        },
    )

    # Recommendation → Answer Synthesis (Advanced RAG)   # === CHANGED ===
    graph.add_conditional_edges(
        "recommendation_agent",
        route_after_recommendation,
        {
            "answer_synthesis_agent": "answer_synthesis_agent",
            "error_node":             "error_node",
        },
    )

    # Answer Synthesis → Report   # === ADDED ===
    graph.add_conditional_edges(
        "answer_synthesis_agent",
        route_after_synthesis,
        {
            "report_generator": "report_generator",
            "error_node":       "error_node",
        },
    )

    # Report → END (or human review for very low confidence)
    graph.add_edge("report_generator", END)

    # Human review and error are terminal-ish; route human review back to normalizer
    # to avoid tight retrieval loops and reset intent/state.
    graph.add_edge("human_review_node", "query_normalizer")
    graph.add_edge("error_node",        END)

    
    
    return graph.compile(
        checkpointer=MemorySaver()
)
    # === REMOVED ===
    # `return graph.compile(checkpointer=checkpointer)` used to sit here.
    # It was unreachable dead code (function already returned above) AND
    # referenced an undefined `checkpointer` variable. Deleted.
    


# ── SINGLETON ────────────────────────────────────────────────────────────────

_graph = None
_retrieval_pipeline = None


def init_retrieval_pipeline():
    """
    Loads chunks ONCE and initializes retrieval system.
    No dependency on PDF upload anymore.
    """
    global _retrieval_pipeline

    if _retrieval_pipeline is not None:
        return _retrieval_pipeline

    path = "backend/data/chunks.json"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        chunks = []

    print(f"📦 Loaded chunks for retrieval: {len(chunks)}")

    _retrieval_pipeline = build_retrieval_pipeline(chunks)

    # Inject into retrieval agent singleton
    from backend.agents_claude.retreival_agent import get_retrieval_agent
    agent = get_retrieval_agent()
    agent.set_pipeline(chunks)

    return _retrieval_pipeline
def get_graph():
    global _graph

    init_retrieval_pipeline()

    if _graph is None:
        _graph = build_graph()   # ONLY ONCE

    return _graph

# ── PIPELINE ENTRY POINT ──────────────────────────────────────────────────────
def run_pipeline(raw_query: str, session_id="default", user_id="anonymous", human_response: str | None = None):

    init_retrieval_pipeline()

    if human_response is not None:
        _pending_human_inputs[session_id] = {"response": human_response}
    
    
    
    initial_query = raw_query

    # If a prior turn already asked for clarification and the caller is now
    # sending the answer, treat it as the next turn's user input.
    if human_response is None:
        pending = _pending_human_inputs.get(session_id)
        if pending and pending.get("response") is not None:
            initial_query = pending["response"]

    initial_state = {
        "raw_query": initial_query,
        "session_id": session_id,
        "user_id": user_id,
        "retry_count": 0,
        "human_retry_count": 0,
        "needs_human_review": False,
    }

    # Try compiled graph but keep its result for comparison/diagnostics.
# Try compiled graph but keep its result for comparison/diagnostics.
    graph_final = None
    from langgraph.errors import GraphInterrupt

    try:
        graph = get_graph()

        if human_response is not None:
            # Resuming a previously interrupted run — this feeds the
            # human's answer back in as the return value of interrupt().
            result = graph.invoke(
                Command(resume=human_response),
                config={"configurable": {"thread_id": session_id}},
            )
        else:
            result = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}},
            )

        if isinstance(result, dict):
            if result.get("__interrupt__"):
                interrupt_payload = result["__interrupt__"][0]
                payload = interrupt_payload.get("value") if isinstance(interrupt_payload, dict) else interrupt_payload
                if isinstance(payload, dict):
                    question = payload.get("question") or payload.get("value")
                    attempt = payload.get("attempt", 1)
                else:
                    question = str(payload)
                    attempt = 1
                return {
                    "status": "awaiting_human_input",
                    "question": question,
                    "attempt": attempt,
                    "session_id": session_id,
                }

            if result.get("human_review_pending"):
                return {
                    "status": "awaiting_human_input",
                    "question": result.get("human_review_question") or "Could you provide a bit more detail?",
                    "attempt": result.get("human_review_attempt", 1),
                    "session_id": session_id,
                }

        if human_response is not None:
            _pending_human_inputs.pop(session_id, None)

        graph_final = result.get("final_report", {}) if isinstance(result, dict) else {}
        if graph_final:
            graph_final.setdefault("diagnostics", {}).setdefault("trace_summary", summarize_trace(session_id))
            return graph_final

    except GraphInterrupt as gi:
        # Safety net: if an interrupt ever escapes graph.invoke() uncaught,
        # surface it as a normal awaiting_human_input response instead of
        # a 500 error.
        try:
            interrupt_obj = gi.args[0][0] if gi.args and gi.args[0] else None
            payload = getattr(interrupt_obj, "value", None) or {}
        except Exception:
            payload = {}
        return {
            "status": "awaiting_human_input",
            "question": payload.get("question", "Could you provide a bit more detail?"),
            "attempt": payload.get("attempt", 1),
            "session_id": session_id,
        }
    except Exception as exc:
        print(f"[run_pipeline] graph invoke failed: {exc}")
        graph_final = None
    # Run a deterministic sequential pipeline to collect per-agent outputs.
    state = dict(initial_state)
    step_outputs = []

    # 1) Normalization
    try:
        state = _safe(query_normalizer_node)(state)
        step_outputs.append({"node": "query_normalizer", "state": {"normalized_query": state.get("normalized_query")}})
    except Exception:
        state["error"] = "normalization_failed"

    # 2) Retrieval
    try:
        updates = _safe(safe_retrieval_node)(state)
        if isinstance(updates, dict):
            state.update(updates)
        chunks = state.get("retrieved_chunks") or []
        step_outputs.append({
            "node": "retrieval_node",
            "retrieved_count": len(chunks),
            "sample_chunks": [ (c.get('source') if isinstance(c, dict) else None, (c.get('text')[:400] if isinstance(c, dict) and c.get('text') else str(c)[:400])) for c in chunks[:5] ]
        })
    except Exception:
        state["error"] = "retrieval_failed"

    # 3) Intent branching & major agents
    MAX_STEPS = 6
    steps = 0
    node_map = {
        "policy_analysis_agent": policy_analysis_node,
        "claim_eligibility_agent": claim_eligibility_node,
        "risk_analysis_agent": risk_analysis_node,
        "comparison_agent": comparison_node,
    }
    
    
    next_node = route_after_retrieval(state)

    # Respect Human-in-the-Loop routing: if retrieval requests clarification,
    # stop here and return the question instead of forcing policy analysis.
    if next_node == "human_review_node":
        return {
            "status": "awaiting_human_input",
            "question": state.get("follow_up_question") or "Could you provide a bit more detail?",
            "attempt": state.get("human_retry_count", 0) + 1,
            "session_id": session_id,
        }

    while next_node and steps < MAX_STEPS:
    
    
        

    # LangGraph Command object
       
        # choose a sensible default based on intent
    
        steps += 1
        if next_node in node_map:
            fn = node_map[next_node]
            try:
                out = _safe(fn)(state)
                if isinstance(out, dict):
                    state.update(out)
                # capture key outputs
                snapshot = {k: state.get(k) for k in ("intent", "confidence", "follow_up_question")}
                step_outputs.append({"node": next_node, "snapshot": snapshot, "raw_output_keys": list(out.keys()) if isinstance(out, dict) else None})
            except Exception:
                state["error"] = f"{next_node}_failed"

            if next_node == "policy_analysis_agent":
                next_node = route_after_policy_analysis(state)
            elif next_node == "claim_eligibility_agent":
                next_node = route_after_claim(state)
            elif next_node == "risk_analysis_agent":
                next_node = route_after_risk(state)
            elif next_node == "comparison_agent":
                next_node = route_after_comparison(state)
            else:
                break
        else:
            break

    # 4) Recommendation
    try:
        rec_out = _safe(recommendation_node)(state)
        if isinstance(rec_out, dict):
            state.update(rec_out)
        step_outputs.append({"node": "recommendation_agent", "keys": list(rec_out.keys()) if isinstance(rec_out, dict) else None})
    except Exception:
        state["error"] = "recommendation_failed"

    # 4.5) Answer Synthesis (Advanced RAG)   # === ADDED ===
    # The graph path gets this via the new edge wired in build_graph(); this
    # sequential fallback path needs the SAME step explicitly, or runs
    # through run_pipeline() (e.g. diagnostics mode, langgraph unavailable)
    # would silently skip the entire Advanced RAG layer.
    try:
        synth_out = _safe(answer_synthesis_node)(state)
        if isinstance(synth_out, dict):
            state.update(synth_out)
        step_outputs.append({
            "node": "answer_synthesis_agent",
            "passed": state.get("synthesis_passed"),
            "faithfulness": state.get("synthesis_faithfulness"),
            "crag_triggered": state.get("synthesis_crag_triggered"),
            "skipped_reason": state.get("synthesis_skipped_reason"),
        })
    except Exception:
        state["error"] = "answer_synthesis_failed"

    # 5) Report
    final_report = {}
    try:
        rep_out = _safe(report_generator_node)(state)
        if isinstance(rep_out, dict):
            state.update(rep_out)
        final_report = state.get("final_report", {})
        step_outputs.append({"node": "report_generator", "final_report_keys": list(final_report.keys()) if isinstance(final_report, dict) else None})
    except Exception:
        state["error"] = "report_generation_failed"

    # If empty, synthesise a deterministic summary with retrieved info included.
    if not final_report or not final_report.get("markdown"):
        parts = []
        parts.append(f"# Automated Report\n\n**Query:** {state.get('raw_query', '')}\n")
        parts.append(f"**Intent:** {state.get('intent', 'unknown')}\n")
        parts.append(f"**Confidence:** {state.get('confidence', 0.0)}\n\n")

        # === ADDED === — surface the grounded, cited answer when we have one
        if state.get("synthesis_passed") and state.get("synthesized_answer"):
            parts.append(f"## Grounded Answer:\n{state['synthesized_answer']}\n")
            parts.append(f"**Citations:** {state.get('synthesis_citations', [])}\n")
            parts.append(f"**Faithfulness Score:** {state.get('synthesis_faithfulness')}\n\n")

        chunks = state.get("retrieved_chunks") or []
        if chunks:
            parts.append("## Retrieved Chunks (samples):\n")
            for i, c in enumerate(chunks[:10], 1):
                text = c.get("text") if isinstance(c, dict) else str(c)
                text = text.replace("\n", " ")
                parts.append(f"- ({c.get('source') if isinstance(c, dict) else None}) {text[:800]}\n")

        for key in ("policy_summary", "risk_summary", "claim_summary", "comparison_summary"):
            if state.get(key):
                parts.append(f"## {key.replace('_', ' ').title()}:\n{state[key]}\n")

        parts.append("\n*This report was auto-generated by the orchestrator fallback pipeline.*")

        final_report = {
            "markdown": "\n".join(parts),
            "json": {
                "status": "auto_generated",
                "intent": state.get("intent"),
                "synthesized_answer": state.get("synthesized_answer"),          # === ADDED ===
                "synthesis_citations": state.get("synthesis_citations", []),    # === ADDED ===
                "synthesis_evaluation": state.get("synthesis_evaluation"),      # === ADDED ===
            },
            "confidence": state.get("confidence", 0.0),
            "confidence_label": state.get("confidence_label", "?"),
            "sources": [c.get("source") for c in chunks if isinstance(c, dict) and c.get("source")],
        }

    # Attach diagnostics/step outputs
    diagnostics = {"steps": step_outputs, "retrieved_chunks": state.get("retrieved_chunks", [])}

    if graph_final and graph_final.get("markdown"):
        graph_final.setdefault("diagnostics", {}).update(diagnostics)
        return graph_final

    final_report.setdefault("diagnostics", {}).update(diagnostics)
    return final_report

# ═══════════════════════════════════════════════════════════════════════════════
# TEST — run:  python -m backend.app.graphs.orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    End-to-end graph test using MOCK retrieval (no real FAISS/Groq needed for routing test).
    Set GROQ_API_KEY to test real LLM calls.
    """
    import os, json
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    if not _LANGGRAPH_AVAILABLE:
        print("❌ langgraph not installed. Run: pip install langgraph")
        exit(1)

    print("\n" + "═" * 65)
    print("  LANGGRAPH ORCHESTRATOR — ROUTING TEST")
    print("═" * 65)

    # Test the routing logic WITHOUT running real LLM calls
    # (just trace which nodes would be activated)

    TEST_ROUTING_CASES = [
        ("waht is waitng perod for catarct", "waiting_period"),
        ("does hdfc ergo cover diabetes", "coverage"),
        ("star vs hdfc health insurance", "comparison"),
        ("how to file a claim for hospitalization", "claim"),
        ("what is not covered in this policy", "exclusion"),
    ]

    for query, expected_intent in TEST_ROUTING_CASES:
        print(f"\n  QUERY    : {query}")
        print(f"  EXPECTED : intent={expected_intent}")

        # Simulate normalized state to test routing only
        mock_state: GraphState = {
            "normalized_query": query,
            "intent": expected_intent,
            "is_comparison": "vs" in query.lower() or "compare" in query.lower(),
            "confidence": 0.75,
            "retrieved_chunks": [],
            "follow_up_question": None,
            "retry_count": 0,
        }

        route = route_after_retrieval(mock_state)
        print(f"  ROUTE →  : {route}")

    # === ADDED === — also sanity-check the new recommendation -> synthesis -> report chain
    print(f"\n{'═' * 65}")
    print("  POST-RECOMMENDATION ROUTING TEST (Advanced RAG wiring)")
    print("═" * 65)
    mock_state_after_rec: GraphState = {"retry_count": 0, "error": None}
    r1 = route_after_recommendation(mock_state_after_rec)
    r2 = route_after_synthesis(mock_state_after_rec)
    print(f"  recommendation_agent → {r1}")
    print(f"  answer_synthesis_agent → {r2}")
    assert r1 == "answer_synthesis_agent", "route_after_recommendation should point at answer_synthesis_agent"
    assert r2 == "report_generator", "route_after_synthesis should point at report_generator"
    print("  ✅ Advanced RAG wiring routes correctly.")

    print(f"\n{'═' * 65}")
    print("  FULL PIPELINE TEST (requires GROQ_API_KEY + retrieval pipeline)")
    print("  Skipping if GROQ_API_KEY not set.")
    print("═" * 65)

    if os.getenv("GROQ_API_KEY"):
        # This requires real setup — only runs if API key is present
        try:
            result = run_pipeline(
                raw_query="matrnity benifit waiting period",
                session_id="test-session-001",
            )
            print(f"\n  CONFIDENCE : {result.get('confidence_label', '?')}")
            print(f"  SOURCES    : {result.get('sources', [])}")
            md = result.get("markdown", "")
            print(f"\n  REPORT PREVIEW (first 20 lines):")
            for line in md.split("\n")[:20]:
                print(f"    {line}")
            # Print diagnostics / per-step outputs if present
            diag = result.get("diagnostics", {})
            steps = diag.get("steps", [])
            if steps:
                print("\n  STEP OUTPUTS:")
                for s in steps:
                    node = s.get("node")
                    # brief summary
                    summary = {k: v for k, v in s.items() if k != "raw_output"}
                    print(f"    - {node}: {summary}")
            if diag.get("retrieved_chunks"):
                print(f"\n  Retrieved chunks (count): {len(diag.get('retrieved_chunks'))}")
        except Exception as e:
            print(f"\n  ⚠️  Pipeline run failed (expected without full setup): {e}")
    else:
        print("\n  ℹ️  Set GROQ_API_KEY to run full pipeline test.")

    print("\n✅ Orchestrator routing tests complete.")