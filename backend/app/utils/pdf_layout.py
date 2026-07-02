"""
pdf_layout.py

Layout-aware text extraction using PyMuPDF.

Features
--------
✓ Detects one-column and two-column layouts
✓ Reads words using coordinates
✓ Preserves natural reading order
✓ Falls back gracefully
"""

from __future__ import annotations
from typing import List

import fitz



# ---------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------

COLUMN_GAP_RATIO = 0.18
LINE_Y_THRESHOLD = 6


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def sort_words(words):
    """
    Sort words first by Y then X.
    """

    return sorted(
        words,
        key=lambda w: (
            round(w["y0"] / LINE_Y_THRESHOLD),
            w["x0"]
        )
    )


# ---------------------------------------------------------
# Group into text lines
# ---------------------------------------------------------

def words_to_lines(words):
    """
    Convert sorted words into lines.

    Returns
    -------
    List[str]
    """

    if not words:
        return []

    lines = []

    current = []

    current_y = words[0]["y0"]

    for word in words:

        if abs(word["y0"] - current_y) <= LINE_Y_THRESHOLD:

            current.append(word)

        else:

            current = sorted(current, key=lambda x: x["x0"])

            lines.append(" ".join(w["text"] for w in current))

            current = [word]

            current_y = word["y0"]

    if current:
        current = sorted(current, key=lambda x: x["x0"])
        lines.append(" ".join(w["text"] for w in current))

    return lines


# ---------------------------------------------------------
# Detect if page is two-column
# ---------------------------------------------------------

def detect_two_columns(words, page_width):
    """
    Simple heuristic using word distribution.
    """

    if len(words) < 50:
        return False

    xs = [w["x0"] for w in words]

    xs.sort()

    middle = page_width / 2

    gap = page_width * COLUMN_GAP_RATIO

    center_words = [
        x
        for x in xs
        if middle - gap < x < middle + gap
    ]

    return len(center_words) < len(xs) * 0.05


# ---------------------------------------------------------
# Split into left/right columns
# ---------------------------------------------------------

def split_columns(words, page_width):

    middle = page_width / 2

    left = []

    right = []

    for word in words:

        if word["x0"] < middle:

            left.append(word)

        else:

            right.append(word)

    return left, right


# ---------------------------------------------------------
# Read one page in natural order
# ---------------------------------------------------------

def extract_page_text(page: fitz.Page) -> str:
    """
    Extract one page with layout awareness.
    """

    raw_words = page.get_text("words")

    if not raw_words:
        return ""

    words = []

    for w in raw_words:

        words.append({
            "x0": w[0],
            "y0": w[1],
            "x1": w[2],
            "y1": w[3],
            "text": w[4]
        })

    page_width = page.rect.width

    if detect_two_columns(words, page_width):

        left, right = split_columns(words, page_width)

        left = sort_words(left)
        right = sort_words(right)

        lines = []

        lines.extend(words_to_lines(left))

        lines.append("")

        lines.extend(words_to_lines(right))

        return "\n".join(lines)

    else:

        words = sort_words(words)

        return "\n".join(words_to_lines(words))


# ---------------------------------------------------------
# Extract complete document
# ---------------------------------------------------------
def extract_document_layout(
    pdf_path: str,
) -> List[dict]:
    """
    Extract every page preserving layout.

    Returns
    -------
    pages : list

    [
        {
            page_number,
            text
        }
    ]
    """

    doc = fitz.open(pdf_path)

    pages = []

    for i, page in enumerate(doc):

        text = extract_page_text(page)

        pages.append({

            "page_number": i + 1,

            "text": text

        })

    doc.close()

    return pages