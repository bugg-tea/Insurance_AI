"""
streamlit_frontend.py
======================
InsureAI — Production Streamlit frontend.

Session design:
  - A UUID is generated ONCE per page load and stored in st.session_state.
  - Page refresh → st.session_state resets → new UUID → fresh Redis key → new conversation.
  - The UUID is sent with every API request so the backend can look up history.

PDF upload design:
  - Up to 5 PDFs per batch.
  - Files are POSTed to /upload with insurer name and doc_type.
  - Processing is synchronous — the UI shows a progress spinner until done.
  - After successful upload, the user can immediately ask questions about the new docs.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── PAGE CONFIG (MUST BE FIRST STREAMLIT CALL) ────────────────────────────────

st.set_page_config(
    page_title="InsureAI — Policy Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "InsureAI — Multi-agent health insurance policy assistant.",
    },
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

API_URL: str = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))

ALLOWED_DOC_TYPES: List[str] = [
    "Policy", "Brochure", "CIS", "Claim",
    "Coverage", "Exclusions", "PreAuth", "Proposal", "Policy Usage Guide",
]

CONFIDENCE_COLORS: Dict[str, str] = {
    "🟢": "#22c55e",
    "🟡": "#f59e0b",
    "🔴": "#ef4444",
    "🔵": "#3b82f6",
}

MAX_UPLOAD_FILES = 5

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── GLOBAL ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── HEADER ── */
.insure-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 18px 0 10px 0;
    border-bottom: 2px solid #e5e7eb;
    margin-bottom: 20px;
}
.insure-header h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    color: #1e3a5f;
}
.insure-header .subtitle {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 2px;
}

/* ── CHAT MESSAGES ── */
.chat-user {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.95rem;
    color: #1e3a5f;
}
.chat-assistant {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.95rem;
    color: #14532d;
}

/* ── CONFIDENCE BADGE ── */
.confidence-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── SOURCE PILL ── */
.source-pill {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #374151;
    margin: 2px 4px 2px 0;
}

/* ── SIDEBAR SECTION TITLE ── */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin: 16px 0 6px 0;
}

/* ── UPLOAD ZONE ── */
.upload-note {
    font-size: 0.78rem;
    color: #6b7280;
    margin-top: 4px;
}

/* ── STATUS INDICATOR ── */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-online  { background: #22c55e; }
.status-offline { background: #ef4444; }
.status-loading { background: #f59e0b; }

/* ── WELCOME CARD ── */
.welcome-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 24px;
}
.welcome-card h2 { margin: 0 0 8px 0; font-size: 1.3rem; }
.welcome-card p  { margin: 0; font-size: 0.9rem; opacity: 0.85; }

/* ── EXAMPLE QUERIES ── */
.example-chip {
    display: inline-block;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #374151;
    cursor: pointer;
    margin: 4px;
    transition: all 0.15s;
}
.example-chip:hover {
    background: #eff6ff;
    border-color: #3b82f6;
    color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE INIT ────────────────────────────────────────────────────────

def _init_session() -> None:
    """
    Called once per page load.  On refresh st.session_state resets entirely
    so a new UUID is assigned → new Redis key → fresh conversation.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        # list of {"role": "user"|"assistant", "content": str, "meta": dict}
        st.session_state.messages = []

    if "api_online" not in st.session_state:
        st.session_state.api_online = False

    if "upload_results" not in st.session_state:
        st.session_state.upload_results = []

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "awaiting_human_input" not in st.session_state:
        st.session_state.awaiting_human_input = False

    if "pending_human_prompt" not in st.session_state:
        st.session_state.pending_human_prompt = None


_init_session()


# ── API HELPERS ───────────────────────────────────────────────────────────────

def _check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _call_query(query: str, human_response: Optional[str] = None) -> Optional[Dict[str, Any]]:
    payload = {
        "query":      query,
        "session_id": st.session_state.session_id,
        "user_id":    "streamlit_user",
    }
    if human_response is not None:
        payload["human_response"] = human_response
    try:
        r = requests.post(
            f"{API_URL}/query",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may be loading — please retry."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Is the backend running?"}
    except Exception as exc:
        return {"error": str(exc)}


def _call_upload(
    file_bytes_list: List[tuple],
    insurer: str,
    doc_type: str,
) -> Optional[Dict[str, Any]]:
    """
    file_bytes_list: list of (filename, bytes) tuples
    """
    files  = [("files", (name, data, "application/pdf")) for name, data in file_bytes_list]
    data   = {"insurer": insurer, "doc_type": doc_type}
    try:
        r = requests.post(
            f"{API_URL}/upload",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT * 3,   # uploads take longer
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"error": "Upload timed out. Large PDFs may take a few minutes."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API."}
    except Exception as exc:
        return {"error": str(exc)}


def _clear_session() -> None:
    try:
        requests.delete(
            f"{API_URL}/session/{st.session_state.session_id}",
            timeout=5,
        )
    except Exception:
        pass
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


# ── RENDER HELPERS ────────────────────────────────────────────────────────────

def _confidence_color(label: str) -> str:
    for emoji, color in CONFIDENCE_COLORS.items():
        if label.startswith(emoji):
            return color
    return "#9ca3af"


def _render_assistant_message(content: str, meta: Dict[str, Any]) -> None:
    """Render one assistant message with optional sources + confidence."""
    st.markdown(content)

    cols = st.columns([3, 1])
    with cols[0]:
        sources: List[str] = meta.get("sources", [])
        if sources:
            pills = "".join(f'<span class="source-pill">📄 {s}</span>' for s in sources[:6])
            st.markdown(f"<div style='margin-top:6px'>{pills}</div>",
                        unsafe_allow_html=True)

    with cols[1]:
        label = meta.get("confidence_label", "")
        if label:
            color = _confidence_color(label)
            st.markdown(
                f'<div style="text-align:right">'
                f'<span class="confidence-badge" '
                f'style="background:{color}20;color:{color};border:1px solid {color}40">'
                f'{label}</span></div>',
                unsafe_allow_html=True,
            )

    intent = meta.get("intent")
    if intent:
        st.caption(f"Intent detected: **{intent}**")


def _render_chat_history() -> None:
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        meta    = msg.get("meta", {})

        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            if role == "assistant":
                _render_assistant_message(content, meta)
            else:
                st.markdown(content)


# ── EXAMPLE QUERIES ───────────────────────────────────────────────────────────

EXAMPLE_QUERIES: List[str] = [
    "What is the waiting period for cataract surgery?",
    "Does HDFC ERGO Optima Secure cover maternity?",
    "Compare Star Health vs Care Supreme for a family of 4.",
    "What are the exclusions in ICICI Lombard Elevate?",
    "How do I file a cashless claim under Star Health?",
    "What is the room rent limit in Niva Bupa ReAssure?",
]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        # Brand
        st.markdown("## 🏥 InsureAI")
        st.caption("Multi-agent insurance policy assistant")
        st.divider()

        # API status
        st.markdown('<p class="sidebar-section">System Status</p>', unsafe_allow_html=True)
        status = _check_api_health()
        st.session_state.api_online = status
        dot_class = "status-online" if status else "status-offline"
        label     = "API Online" if status else "API Offline"
        st.markdown(
            f'<div><span class="status-dot {dot_class}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Session info
        st.markdown('<p class="sidebar-section">Session</p>', unsafe_allow_html=True)
        st.caption(f"ID: `{st.session_state.session_id[:8]}…`")
        st.caption(f"Messages: {len(st.session_state.messages)}")

        if st.button("🗑️ New Conversation", use_container_width=True):
            _clear_session()
            st.rerun()

        st.divider()

        # PDF Upload
        st.markdown('<p class="sidebar-section">Upload Your PDFs</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="upload-note">Upload up to 5 PDF policy documents. '
            'They will be extracted, chunked, and indexed automatically.</p>',
            unsafe_allow_html=True,
        )

        insurer_name = st.text_input(
            "Insurer name",
            value="My Insurer",
            placeholder="e.g. HDFC ERGO",
            help="This label is stored with the document for attribution.",
        )

        doc_type = st.selectbox(
            "Document type",
            options=ALLOWED_DOC_TYPES,
            index=0,
        )

        uploaded_files = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum {MAX_UPLOAD_FILES} files per batch.",
            label_visibility="collapsed",
        )

        if uploaded_files and len(uploaded_files) > MAX_UPLOAD_FILES:
            st.error(f"Please select at most {MAX_UPLOAD_FILES} files.")
            uploaded_files = uploaded_files[:MAX_UPLOAD_FILES]

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")

            if st.button("⬆️ Upload & Index", use_container_width=True, type="primary"):
                if not st.session_state.api_online:
                    st.error("API is offline. Start the backend first.")
                else:
                    file_bytes_list = [(f.name, f.read()) for f in uploaded_files]
                    with st.spinner(f"Processing {len(file_bytes_list)} file(s)… "
                                    "This may take 1–3 minutes per PDF."):
                        result = _call_upload(file_bytes_list, insurer_name, doc_type)

                    if result and "error" not in result:
                        ingested = result.get("ingested", [])
                        errors   = result.get("errors", [])

                        if ingested:
                            st.success(f"✅ Indexed {len(ingested)} file(s)!")
                            for item in ingested:
                                st.caption(
                                    f"📄 **{item['file_name']}** — "
                                    f"{item['page_count']} pages, "
                                    f"{item['chunk_count']} chunks"
                                )
                            st.session_state.upload_results.extend(ingested)

                        if errors:
                            for err in errors:
                                st.error(f"❌ {err['file']}: {err['error']}")
                    else:
                        err_msg = result.get("error", "Unknown error") if result else "No response"
                        st.error(f"Upload failed: {err_msg}")

        # Previously uploaded files this session
        if st.session_state.upload_results:
            st.divider()
            st.markdown('<p class="sidebar-section">Uploaded This Session</p>',
                        unsafe_allow_html=True)
            for item in st.session_state.upload_results:
                st.caption(
                    f"📄 {item['file_name']} "
                    f"({item.get('insurer','?')} / {item.get('document_type','?')})"
                )

        st.divider()

        # Dataset info
        st.markdown('<p class="sidebar-section">Existing Dataset</p>', unsafe_allow_html=True)
        st.caption("Care Health • HDFC ERGO • ICICI Lombard • Niva Bupa • Star Health")

        st.divider()
        st.caption("Built with LangGraph · FAISS · Groq · Streamlit")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def _render_welcome() -> None:
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Ask me anything about health insurance</h2>
        <p>
            I can compare policies, explain coverage, find exclusions,
            guide you through claims, and analyse waiting periods across
            5 major Indian insurers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Try an example:**")
    cols = st.columns(3)
    for i, q in enumerate(EXAMPLE_QUERIES):
        col = cols[i % 3]
        with col:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_query = q
                st.rerun()


def main() -> None:
    _render_sidebar()

    # Header
    st.markdown("""
    <div class="insure-header">
        <span style="font-size:2rem">🏥</span>
        <div>
            <h1>InsureAI</h1>
            <div class="subtitle">Multi-agent insurance policy assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Welcome screen when no messages yet
    if not st.session_state.messages:
        _render_welcome()

    # Chat history
    _render_chat_history()

    # Handle example query click (set before rerun, consumed here)
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
    else:
        query = None

    # Chat input
    user_input = st.chat_input(
        "Ask about coverage, exclusions, claims, waiting periods …",
        disabled=not st.session_state.api_online,
    )

    active_query = user_input or query

    if st.session_state.get("awaiting_human_input") and user_input is None and query is None:
        active_query = None

    if active_query:
        if not st.session_state.api_online:
            st.error("The API backend is offline. Please start it and refresh the page.")
            return
        if st.session_state.get("awaiting_human_input") and st.session_state.get("pending_human_prompt"):
            human_response = active_query
            st.session_state.awaiting_human_input = False
            st.session_state.pending_human_prompt = None
            st.session_state.messages.append({
                "role": "user",
                "content": human_response,
                "meta": {},
            })
            with st.chat_message("user", avatar="🧑"):
                st.markdown(human_response)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Resuming analysis …"):
                    result = _call_query(human_response, human_response=human_response)
        else:
            st.session_state.awaiting_human_input = False
            st.session_state.pending_human_prompt = None
            st.session_state.messages.append({
                "role": "user",
                "content": active_query,
                "meta": {},
            })
            with st.chat_message("user", avatar="🧑"):
                st.markdown(active_query)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analysing policies …"):
                    result = _call_query(active_query)

        # Rendering now runs for BOTH the HIL-resume path and the normal
        # path, since `result` is set by either branch above.
        if isinstance(result, dict) and "error" not in result:
            diagnostics = result.get("diagnostics") or {}
            if isinstance(diagnostics, dict) and diagnostics.get("status") == "awaiting_human_input":
                prompt_question = diagnostics.get("question") or result.get("markdown") or "Please provide the missing information."
                st.session_state.awaiting_human_input = True
                st.session_state.pending_human_prompt = prompt_question
                st.info(f"🧑 Please answer: {prompt_question}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": prompt_question,
                    "meta": {"confidence_label": "🟡 NEEDS INPUT"},
                })
            else:
                markdown_text = result.get("markdown", "No response generated.")
                meta = {
                    "confidence":       result.get("confidence", 0.0),
                    "confidence_label": result.get("confidence_label", ""),
                    "sources":          result.get("sources", []),
                    "intent":           result.get("intent"),
                }

                _render_assistant_message(markdown_text, meta)

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": markdown_text,
                    "meta":    meta,
                })

        else:
            err = result.get("error", "Unknown error") if isinstance(result, dict) else "No response from API."
            st.error(f"⚠️ {err}")
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"⚠️ Error: {err}",
                "meta":    {},
            })
            
        

if __name__ == "__main__":
    main()