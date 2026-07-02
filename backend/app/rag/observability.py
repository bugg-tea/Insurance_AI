"""
Observability / Tracing
=========================
"Trace Every Agent -> Prompt Versioning -> Tool Calls -> Latency ->
State Transitions -> Memory Usage -> Errors -> Graph Visualization"

Setup
-----
Works with ZERO setup out of the box: every traced node writes a line of
JSON to backend/data/traces/trace_<session_id>.jsonl on disk — free,
local, no account needed. Good enough for debugging and basic latency
analysis.

OPTIONAL (still free tier available): if you want hosted traces +
a graph visualization UI via LangSmith:
    1. pip install langsmith --break-system-packages
    2. Sign up free at https://smith.langchain.com (free tier exists)
    3. export LANGCHAIN_TRACING_V2=true
       export LANGCHAIN_API_KEY=<your key>
       export LANGCHAIN_PROJECT=insurance-rag
When those env vars are present AND the `langsmith` package is installed,
this module additionally emits an `@traceable`-wrapped call so the run
shows up in your LangSmith project. If not configured, it silently
no-ops on the LangSmith side and you still get the local JSONL trace.

Run standalone:
    python -m backend.agents_claude.observability
"""

from __future__ import annotations

import os
import json
import time
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Optional

TRACE_DIR = Path(__file__).resolve().parents[1] / "data" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

_LANGSMITH_ENABLED = (
    os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    and bool(os.getenv("LANGSMITH_API_KEY"))
)
_traceable = None
if _LANGSMITH_ENABLED:
    try:
        from langsmith import traceable as _traceable  # type: ignore
    except Exception:
        _traceable = None
        _LANGSMITH_ENABLED = False


def _trace_file(session_id: str) -> Path:
    safe = "".join(c for c in str(session_id) if c.isalnum() or c in ("-", "_")) or "default"
    return TRACE_DIR / f"trace_{safe}.jsonl"


def log_event(session_id: str, event: Dict[str, Any]) -> None:
    event = {"ts": time.time(), **event}
    try:
        with open(_trace_file(session_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception as e:
        print(f"⚠️  Failed to write trace event: {e}")


def traced_node(node_name: str):
    """
    Decorator for graph node functions: `def node(state) -> state_updates`.
    Logs latency + a shallow snapshot of input/output keys to the local
    JSONL trace file, keyed by state["session_id"]. Also forwards through
    LangSmith's @traceable when configured (see module docstring).
    """

    def decorator(fn: Callable[[Dict[str, Any]], Dict[str, Any]]):

        @functools.wraps(fn)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            session_id = state.get("session_id", "default")
            run_id = state.get("run_id")
            start = time.time()
            error: Optional[str] = None
            output: Dict[str, Any] = {}

            try:
                output = fn(state)
                return output
            except Exception as e:
                error = str(e)
                raise
            finally:
                end = time.time()

                latency_ms = round((end - start) * 1000, 1)
                state_size = len(state)
                try:
                    state_bytes = len(json.dumps(state, default=str))
                except Exception:
                    state_bytes = None
                log_event(session_id, {
    "run_id": run_id,
    "session_id": session_id,
    "node": node_name,

    "started_at": start,
    "finished_at": end,

    "latency_ms": latency_ms,

    "state_size": state_size,
    "state_bytes": state_bytes,

    "input_keys": sorted(state.keys()),
    "output_keys": sorted(output.keys()) if isinstance(output, dict) else None,

    "error": error,
})
                
                

        if _LANGSMITH_ENABLED and _traceable is not None:
            return _traceable(name=node_name)(wrapper)
        return wrapper

    return decorator


def read_trace(session_id: str) -> list:
    path = _trace_file(session_id)
    if not path.exists():
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize_trace(session_id: str) -> Dict[str, Any]:
    events = read_trace(session_id)
    if not events:
        return {"session_id": session_id, "node_count": 0, "total_latency_ms": 0, "errors": []}

    total_latency = sum(e.get("latency_ms", 0) for e in events)
    errors = [e for e in events if e.get("error")]
    by_node = {}
    for e in events:
        by_node.setdefault(e["node"], []).append(e.get("latency_ms", 0))

    return {
        "session_id": session_id,
        "node_count": len(events),
        "total_latency_ms": round(total_latency, 1),
        "latency_by_node_ms": {k: round(sum(v) / len(v), 1) for k, v in by_node.items()},
        "errors": errors,
        "langsmith_enabled": _LANGSMITH_ENABLED,
        "max_state_size": max(e.get("state_size", 0) for e in events),
        "max_state_bytes": max(e.get("state_bytes", 0) or 0 for e in events),
        "run_ids": sorted({
            e.get("run_id")
            for e in events
            if e.get("run_id") is not None
}),
    }



# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json as _json

    @traced_node("dummy_node")
    def dummy_node(state):
        time.sleep(0.01)
        return {**state, "dummy_output": 42}

    s = {"session_id": "test-trace-session", "raw_query": "hello"}
    s2 = dummy_node(s)
    s3 = dummy_node(s2)

    print(_json.dumps(summarize_trace("test-trace-session"), indent=2))