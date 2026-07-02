"""
Context Compressor
===================
Step 4 of the Advanced RAG Pipeline:  Hybrid Retrieval -> Reranking -> **Context
Compression** -> Grounded Prompt -> LLM ...

Purpose
-------
Retrieved + reranked chunks are often redundant (near-duplicate table rows,
overlapping sliding-window text chunks) and can blow past the LLM's context
budget. This module:

  1. Deduplicates near-identical chunks (cheap Jaccard token overlap, no model).
  2. Trims each chunk down to the sentences most relevant to the query
     (keyword overlap scoring — free, no embeddings required).
  3. Packs chunks into a token budget, highest-score first.
  4. Emits a citation-ready block: each chunk gets a stable inline tag like
     [C_a1b2c3d4] that the LLM is instructed to cite, and that the
     citation_checker.py module later verifies against.

No setup required — pure Python, no external services.

Run standalone:
    python -m backend.agents_claude.context_compressor
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple
import hashlib
# =========================================================
# CONFIG
# =========================================================

DEFAULT_TOKEN_BUDGET = 1800          # ~ words budget for packed context
MAX_SENTENCES_PER_CHUNK = 4          # cap per-chunk length after trimming
DEDUP_JACCARD_THRESHOLD = 0.65       # chunks more similar than this are merged
MIN_CHUNK_KEEP_SCORE = 0.0           # sentences scoring <= this get dropped if chunk has alternatives


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter — good enough for policy clause text and
    # table_row pipe-separated text; avoids pulling in nltk/spacy (paid setup
    # / heavy install footprint).
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.?!])\s+|\s*\|\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _dedupe_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    kept_tokens: List[List[str]] = []

    # Highest-scored first so we keep the "best" copy when merging dupes.
    ordered = sorted(chunks, key=lambda c: float(c.get("score", c.get("final_score", 0.0)) or 0.0), reverse=True)

    for c in ordered:
        toks = _tokenize(c.get("text", ""))
        is_dupe = False
        for kt in kept_tokens:
            if _jaccard(toks, kt) >= DEDUP_JACCARD_THRESHOLD:
                is_dupe = True
                break
        if not is_dupe:
            kept.append(c)
            kept_tokens.append(toks)

    return kept


def _trim_chunk_text(text: str, query_tokens: List[str]) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= MAX_SENTENCES_PER_CHUNK:
        return " ".join(sentences)

    scored: List[Tuple[float, int, str]] = []
    qset = set(query_tokens)

    for idx, s in enumerate(sentences):
        stoks = set(_tokenize(s))
        overlap = len(stoks & qset)
        scored.append((overlap, idx, s))
        
    filtered = [
    s for s in scored
    if s[0] > MIN_CHUNK_KEEP_SCORE
]
    if not filtered:
        filtered = scored
        
    top = sorted(
        filtered,
        key=lambda x: x[0],
        reverse=True
    )[:MAX_SENTENCES_PER_CHUNK]

    # Keep top-N by overlap, but restore original order for readability.
    
    top_sorted_by_position = sorted(top, key=lambda x: x[1])
    return " ".join(s for _, _, s in top_sorted_by_position)


def _estimate_tokens(text: str) -> int:
    # Rough word-count proxy for tokens — avoids needing a tokenizer install.
    return max(1, int(len(text.split()) * 1.3))


def compress(
    query: str,
    chunks: List[Dict[str, Any]],
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Dict[str, Any]:
    """
    Returns:
        {
          "context_block": "<formatted text ready to paste into the prompt>",
          "used_chunks": [ {chunk_id, tag, text, chunk_type, score, page_start, ...} ],
          "dropped_count": int,
          "tokens_used": int,
        }
    """

    if not chunks:
        return {"context_block": "", "used_chunks": [], "dropped_count": 0, "tokens_used": 0}

    query_tokens = _tokenize(query)

    deduped = _dedupe_chunks(chunks)
    dropped = len(chunks) - len(deduped)

    # Sentence-level selection across all chunks: score each sentence by
    # query token overlap, then pick top sentences until token_budget is
    # exhausted. This keeps the most relevant sentences while reducing
    # extraneous text and latency.
    sentence_pool: List[Tuple[float, int, Dict[str, Any], str]] = []
    # (score, original_chunk_index, chunk_obj, sentence_text)
    for idx, c in enumerate(deduped):
        raw_text = c.get("text", "") or ""
        if not raw_text.strip():
            continue
        sentences = _split_sentences(raw_text)
        for s in sentences:
            stoks = set(_tokenize(s))
            score = len(stoks & set(query_tokens))
            sentence_pool.append((float(score), idx, c, s))

    # Keep only reasonably scored sentences (allow zero-scored to avoid empty)
    sentence_pool.sort(key=lambda x: x[0], reverse=True)

    used_chunks_map: Dict[int, List[Tuple[int, str]]] = {}
    tokens_used = 0
    lines: List[str] = []

    for score, idx, c, s in sentence_pool:
        entry_tokens = _estimate_tokens(s)
        if tokens_used + entry_tokens > token_budget and lines:
            break
        # Limit sentences per chunk
        lst = used_chunks_map.setdefault(idx, [])
        if len(lst) >= MAX_SENTENCES_PER_CHUNK:
            continue
        lst.append((len(lst), s))
        tokens_used += entry_tokens

    # Build used_chunks and context lines preserving original sentence order
    used_chunks: List[Dict[str, Any]] = []
    for idx, sentences in used_chunks_map.items():
        c = deduped[idx]
        # restore original sentence order for readability
        sentences_sorted = [s for _, s in sorted(sentences, key=lambda x: x[0])]
        trimmed = " ".join(sentences_sorted)

        chunk_id = c.get("chunk_id") or c.get("id") or f"anon_{len(used_chunks)}"
        tag = "C_" + hashlib.md5(str(chunk_id).encode("utf-8")).hexdigest()[:8]

        meta_bits = []
        if c.get("chunk_type"):
            meta_bits.append(c["chunk_type"])
        if c.get("page_start") is not None:
            meta_bits.append(f"page {c.get('page_start')}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""

        lines.append(f"[{tag}]{meta}: {trimmed}")

        chunk_metadata = dict(c)
        chunk_metadata.update({
            "chunk_id": chunk_id,
            "tag": tag,
            "text": trimmed,
        })
        used_chunks.append(chunk_metadata)

    context_block = "\n".join(lines)

    return {
        "context_block": context_block,
        "used_chunks": used_chunks,
        "dropped_count": dropped + (len(deduped) - len(used_chunks)),
        "tokens_used": tokens_used,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json

    MOCK_CHUNKS = [
        {"chunk_id": "abc12345", "text": "Waiting period for pre-existing diseases is 48 months. This applies to all riders.", "chunk_type": "clause", "score": 0.91, "page_start": 4},
        {"chunk_id": "abc12345dup", "text": "Waiting period for pre-existing diseases is 48 months under this policy.", "chunk_type": "clause", "score": 0.85, "page_start": 4},
        {"chunk_id": "def67890", "text": "Maternity benefits have a waiting period of 24 months. Cosmetic surgery is not covered. Dental treatment is excluded unless accidental.", "chunk_type": "exclusion", "score": 0.78, "page_start": 7},
        {"chunk_id": "ghi11122", "text": "Cataract surgery is covered after a 2-year waiting period, subject to sub-limits per the table below.", "chunk_type": "clause", "score": 0.70, "page_start": 9},
    ]

    result = compress("waiting period for pre-existing diseases", MOCK_CHUNKS, token_budget=200)
    print(json.dumps(result, indent=2))
    print(f"\nDeduped {len(MOCK_CHUNKS)} -> {len(result['used_chunks'])} kept chunks, "
          f"{result['dropped_count']} dropped, {result['tokens_used']} tokens used.")