"""
Citation Checker
================
Step 9 of the Advanced RAG Pipeline: Self Reflection -> Response Validator ->
**Citation Checker** -> JSON Output.

Purpose
-------
Two-way verification of the generated answer's citation tags ([C_xxxxxxxx]):

  1. VALID CITATIONS — every tag the answer cites must correspond to a chunk
     that was actually in the packed context (catches hallucinated sources).
  2. CITATION COVERAGE — every sentence carrying a factual claim should end
     with at least one valid citation tag (catches uncited / unsupported
     claims slipped in by the LLM).

Pure Python, no external services, no setup required.

Run standalone:
    python -m backend.agents_claude.citation_checker
"""

from __future__ import annotations

import re
from typing import Dict, Any, List

CITATION_PATTERN = re.compile(r"\[C_[a-zA-Z0-9]+\]")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Split after .?! UNLESS immediately followed by a citation tag (keep tag
    # glued to its sentence), and also split after a citation tag's closing
    # bracket when a new capitalized sentence follows directly after it.
    raw_parts = re.split(r"(?<=[.?!])\s+(?!\[C_)|(?<=\])\s+(?=[A-Z])", text)
    return [p.strip() for p in raw_parts if p.strip()]


def _is_factual_sentence(sentence: str) -> bool:
    """
    Heuristic: short courtesy/caveat sentences ("I'm not sure.",
    "Please consult your policy document.") don't need a citation. Anything
    else with content words does.
    """
    stripped = re.sub(CITATION_PATTERN, "", sentence).strip()
    word_count = len(stripped.split())
    return word_count >= 4


def check_citations(answer: str, used_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns:
        {
          "valid_tags": [...],          # tags cited AND present in context
          "invalid_tags": [...],        # tags cited but NOT in context (hallucinated source)
          "uncited_factual_sentences": [...],
          "citation_coverage": float 0-1,   # fraction of factual sentences with >=1 valid citation
          "has_hallucinated_citation": bool,
        }
    """

    valid_tag_set = {c["tag"] for c in used_chunks if c.get("tag")}
    cited_tags = {m.strip("[]") for m in CITATION_PATTERN.findall(answer)}

    valid_tags = sorted(cited_tags & valid_tag_set)
    invalid_tags = sorted(cited_tags - valid_tag_set)

    sentences = _split_sentences(answer)
    factual_sentences = [s for s in sentences if _is_factual_sentence(s)]

    uncited = []
    cited_count = 0
    for s in factual_sentences:
        tags_in_sentence = {m.strip("[]") for m in CITATION_PATTERN.findall(s)}
        if tags_in_sentence & valid_tag_set:
            cited_count += 1
        else:
            uncited.append(s)

    coverage = round(cited_count / len(factual_sentences), 3) if factual_sentences else 1.0

    return {
        "valid_tags": valid_tags,
        "invalid_tags": invalid_tags,
        "uncited_factual_sentences": uncited,
        "citation_coverage": coverage,
        "has_hallucinated_citation": len(invalid_tags) > 0,
    }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    import json

    used_chunks = [
        {"chunk_id": "abc12345", "tag": "C_abc12345", "text": "Waiting period for pre-existing diseases is 48 months."},
    ]

    cases = {
        "fully_cited": "The waiting period for pre-existing diseases is 48 months. [C_abc12345]",
        "uncited_claim": "The waiting period for pre-existing diseases is 48 months. [C_abc12345] You will also get a free wellness check every year.",
        "hallucinated_source": "The waiting period is 48 months. [C_zzz99999]",
    }

    for label, ans in cases.items():
        print(f"\n--- {label} ---")
        print(json.dumps(check_citations(ans, used_chunks), indent=2))