"""
backend/api/session.py
======================
Redis-backed short-term conversation memory.

Design:
  - Each browser session gets a UUID (generated client-side in Streamlit).
  - On page refresh → new UUID → new Redis key → fresh conversation.
  - TTL = SESSION_TTL_SECONDS (default 3600).  Idle sessions expire automatically.
  - Max MAX_HISTORY_TURNS kept to avoid bloating prompts.
  - All Redis calls are wrapped in try/except so a missing Redis does NOT crash the API
    (degrades gracefully to stateless mode).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import redis

# ── CONFIG ────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_HISTORY_TURNS: int = 20          # keep last N turns (user + assistant each count as 1)

# ── SINGLETON CLIENT ──────────────────────────────────────────────────────────
_redis_client: Optional[redis.Redis] = None


def _get_redis() -> Optional[redis.Redis]:
    """Lazy-init Redis client.  Returns None if connection fails (graceful degradation)."""
    global _redis_client
    if _redis_client is not None:
        try:
            _redis_client.ping()
            return _redis_client
        except Exception:
            _redis_client = None          # reset on ping failure

    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        _redis_client.ping()
        return _redis_client
    except Exception as exc:
        print(f"[session] Redis unavailable ({exc}). Running in stateless mode.")
        return None


# ── KEY HELPERS ───────────────────────────────────────────────────────────────

def _history_key(session_id: str) -> str:
    return f"session:{session_id}:history"


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return conversation history for this session (empty list if none/error)."""
    r = _get_redis()
    if r is None:
        return []
    try:
        raw = r.get(_history_key(session_id))
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return []


def append_turn(session_id: str, role: str, content: str) -> None:
    """
    Append one turn to the session history and reset TTL.
    role must be 'user' or 'assistant'.
    """
    r = _get_redis()
    if r is None:
        return
    try:
        history = get_history(session_id)
        history.append({"role": role, "content": content})
        # Trim to last MAX_HISTORY_TURNS turns
        history = history[-MAX_HISTORY_TURNS:]
        r.setex(_history_key(session_id), SESSION_TTL, json.dumps(history, ensure_ascii=False))
    except Exception:
        pass


def clear_session(session_id: str) -> None:
    """Delete all history for this session."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.delete(_history_key(session_id))
    except Exception:
        pass


def session_exists(session_id: str) -> bool:
    r = _get_redis()
    if r is None:
        return False
    try:
        return r.exists(_history_key(session_id)) == 1
    except Exception:
        return False


def refresh_ttl(session_id: str) -> None:
    """Reset the TTL of an active session (call on each user interaction)."""
    r = _get_redis()
    if r is None:
        return
    try:
        r.expire(_history_key(session_id), SESSION_TTL)
    except Exception:
        pass