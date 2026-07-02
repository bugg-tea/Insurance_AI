"""
pdf_cleaner.py

Production-grade text cleaning utilities for PDF extraction.

This module ONLY cleans text.
It does NOT perform OCR, layout detection, or PDF extraction.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import List


# ---------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters.

    Converts ligatures and fancy unicode into standard form.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u00A0": " ",      # non-breaking space
        "\u200B": "",       # zero width space
        "\u200C": "",
        "\u200D": "",
        "\ufeff": "",
        "\x00": "",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "•": "-",
        "▪": "-",
        "■": "-",
        "●": "-",
        "◦": "-",
        "·": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ---------------------------------------------------------
# Remove control characters
# ---------------------------------------------------------

def remove_control_characters(text: str) -> str:
    """
    Remove invisible control characters.
    """

    return "".join(
        ch
        for ch in text
        if unicodedata.category(ch)[0] != "C"
        or ch in ("\n", "\t")
    )


# ---------------------------------------------------------
# Fix broken hyphenated words
# ---------------------------------------------------------

def repair_hyphenated_words(text: str) -> str:
    """
    Hospital-
    ization

    ->
    Hospitalization
    """

    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


# ---------------------------------------------------------
# Normalize line endings
# ---------------------------------------------------------

def normalize_line_endings(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text


# ---------------------------------------------------------
# Normalize whitespace
# ---------------------------------------------------------

def normalize_whitespace(text: str) -> str:

    lines = []

    for line in text.split("\n"):

        line = re.sub(r"[ \t]+", " ", line)

        line = line.strip()

        lines.append(line)

    text = "\n".join(lines)

    return text


# ---------------------------------------------------------
# Remove repeated blank lines
# ---------------------------------------------------------

def remove_extra_blank_lines(text: str) -> str:

    return re.sub(r"\n{3,}", "\n\n", text)


# ---------------------------------------------------------
# Remove standalone page numbers
# ---------------------------------------------------------

def remove_page_numbers(text: str) -> str:
    """
    Removes lines containing only numbers.

    Example:

    14

    becomes removed.
    """

    cleaned = []

    for line in text.split("\n"):

        if re.fullmatch(r"\d+", line.strip()):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------
# Remove duplicated consecutive lines
# ---------------------------------------------------------

def remove_duplicate_lines(text: str) -> str:

    output = []

    previous = None

    for line in text.split("\n"):

        if line == previous:
            continue

        output.append(line)

        previous = line

    return "\n".join(output)


# ---------------------------------------------------------
# Detect repeated headers/footers
# ---------------------------------------------------------

def detect_repeated_lines(
    pages: List[str],
    min_frequency: float = 0.6
) -> List[str]:
    """
    Detect lines that appear on most pages.

    Used for removing headers and footers.
    """

    counter = Counter()

    total_pages = len(pages)

    for page in pages:

        unique = set(page.split("\n"))

        counter.update(unique)

    repeated = []

    threshold = max(2, int(total_pages * min_frequency))

    for line, freq in counter.items():

        if len(line.strip()) < 3:
            continue

        if freq >= threshold:
            repeated.append(line)

    return repeated


# ---------------------------------------------------------
# Remove repeated headers/footers
# ---------------------------------------------------------

def remove_repeated_lines(
    pages: List[str]
) -> List[str]:

    repeated = detect_repeated_lines(pages)

    cleaned_pages = []

    for page in pages:

        lines = []

        for line in page.split("\n"):

            if line in repeated:
                continue

            lines.append(line)

        cleaned_pages.append("\n".join(lines))

    return cleaned_pages


# ---------------------------------------------------------
# Final cleaning pipeline
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Full cleaning pipeline.
    """

    text = normalize_unicode(text)

    text = remove_control_characters(text)

    text = normalize_line_endings(text)

    text = repair_hyphenated_words(text)

    text = normalize_whitespace(text)

    text = remove_duplicate_lines(text)

    text = remove_page_numbers(text)

    text = remove_extra_blank_lines(text)

    return text.strip()


# ---------------------------------------------------------
# Clean all extracted pages
# ---------------------------------------------------------

def clean_pages(pages: List[dict]) -> List[dict]:
    """
    Cleans page-wise extracted text.

    Input:

    [
        {
            page_number,
            text
        }
    ]
    """

    texts = [p["text"] for p in pages]

    texts = remove_repeated_lines(texts)

    cleaned = []

    for page, text in zip(pages, texts):

        cleaned.append({

            "page_number": page["page_number"],

            "text": clean_text(text)

        })

    return cleaned


# ---------------------------------------------------------
# Join cleaned pages
# ---------------------------------------------------------

def pages_to_document(pages: List[dict]) -> str:
    """
    Convert cleaned pages into one document.
    """

    output = []

    for page in pages:

        output.append(
            f"\n========== PAGE {page['page_number']} ==========\n"
        )

        output.append(page["text"])

    return "\n".join(output).strip()