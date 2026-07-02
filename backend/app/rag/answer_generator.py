"""
Answer Generator (Grounded Prompt -> LLM)
==========================================
Step 5-6 of the Advanced RAG Pipeline: Context Compression -> **Grounded
Prompt -> LLM** -> Self Reflection ...

Setup
-----
Uses your existing `backend.agents_claude.llm_client.get_client()` (same
client query_normalizer.py already uses — e.g. Groq free tier). If that
client is unavailable for any reason (no API key, import error, network),
this module automatically falls back to a FREE, no-LLM extractive answer
built directly from the compressed context, so the pipeline never hard-fails.

No new dependencies required.

Run standalone:
    python -m backend.agents_claude.answer_generator
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

try:
    from backend.agents_claude.llm_client import get_client
except Exception:
    get_client = None  # type: ignore


GROUNDED_SYSTEM_PROMPT = """You are an Insurance Policy QA Assistant.

Answer the user's question using ONLY the CONTEXT provided below. Each
context item has a tag like [C_xxxxxxxx] — you MUST cite the tag(s) you
used for every factual claim, inline, like: "The waiting period is 48
months [C_a1b2c3d4]."

Rules:
- If the context does not contain the answer, say so explicitly instead of
  guessing. Never invent numbers, dates, or policy names.
- Every sentence containing a factual claim must end with at least one
  citation tag.
- Be concise and direct. No filler, no disclaimers beyond what's asked.
- Do not cite a tag that isn't in the CONTEXT.

Return ONLY the answer text (no JSON, no markdown headers).
"""


def _build_user_prompt(query: str, context_block: str) -> str:
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer, citing context tags inline as instructed."
    )


def _extractive_fallback(query: str, used_chunks: List[Dict[str, Any]]) -> str:
    """
    Free, no-LLM fallback: stitches together the highest scoring chunk
    sentences with their citation tags. Lower quality than an LLM answer,
    but never silently fabricates anything, and is always available.
    """
    if not used_chunks:
        return "I couldn't find relevant information in the policy documents to answer this question."

    parts = []
    for c in used_chunks[:4]:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"{text} [{c['tag']}]")

    if not parts:
        return "I couldn't find relevant information in the policy documents to answer this question."

    return " ".join(parts)


def generate_answer(
    query: str,
    context_block: str,
    used_chunks: List[Dict[str, Any]],
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "answer": str,
          "used_llm": bool,
          "raw_citations_found": [tags...],
        }
    """

    answer_text = None
    used_llm = False

    client = llm_client
    if client is None and get_client is not None:
        try:
            client = get_client()
        except Exception:
            client = None

    if client is not None and context_block.strip():
        try:
            messages = [
                {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, context_block)},
            ]
            if hasattr(client, "call"):
                answer_text = client.call(messages, max_tokens=500)
            elif hasattr(client, "call_json"):
                # Some clients only expose call_json — wrap plain text in a key.
                raw = client.call_json(
                    [
                        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT + '\nReturn JSON: {"answer": "..."}'},
                        {"role": "user", "content": _build_user_prompt(query, context_block)},
                    ],
                    max_tokens=500,
                )
                answer_text = raw.get("answer") if isinstance(raw, dict) else None
            if answer_text:
                used_llm = True
        except Exception as e:
            print(f"⚠️  LLM answer generation failed, using extractive fallback: {e}")
            answer_text = None

    if not answer_text:
        answer_text = _extractive_fallback(query, used_chunks)
        
    # ---------------------------------------------------------
# Build the set of valid citation tags from compressed context
# ---------------------------------------------------------

    valid_tags = {
        chunk["tag"]
        for chunk in used_chunks
        if chunk.get("tag")
}
    citation_tags = sorted(
    set(re.findall(r"\[C_[a-zA-Z0-9]+\]", answer_text))
)
    valid_citations = [
        tag
        for tag in citation_tags
        if tag.strip("[]") in valid_tags
]   
    if used_llm and not valid_citations:
        print("⚠️ LLM returned answer without valid citations.")
        print("⚠️ Falling back to extractive answer.")

        answer_text = _extractive_fallback(query, used_chunks)

        citation_tags = sorted(
            set(re.findall(r"\[C_[a-zA-Z0-9]+\]", answer_text))
    )

        valid_citations = [
            tag
            for tag in citation_tags
            if tag.strip("[]") in valid_tags
    ]

        used_llm = False
        
    return {
    "answer": answer_text.strip(),
    "used_llm": used_llm,
    "raw_citations_found": [
        tag.strip("[]")
        for tag in valid_citations
    ],
}
        

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json
    from backend.app.rag.context_compression import compress

    MOCK_CHUNKS = [
        {"chunk_id": "abc12345", "text": "Waiting period for pre-existing diseases is 48 months.", "chunk_type": "clause", "score": 0.91, "page_start": 4},
        {"chunk_id": "def67890", "text": "Maternity benefits have a waiting period of 24 months.", "chunk_type": "clause", "score": 0.78, "page_start": 7},
    ]

    query = "What is the waiting period for pre-existing diseases?"
    compressed = compress(query, MOCK_CHUNKS)

    result = generate_answer(query, compressed["context_block"], compressed["used_chunks"])
    print(json.dumps(result, indent=2))