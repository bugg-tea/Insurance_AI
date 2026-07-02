"""
pdf_enrichment.py

Document enrichment utilities for PDF extraction.

This module performs higher-level document analysis after text
extraction is complete.

Features
--------
✓ PDF metadata extraction
✓ Embedded image detection
✓ Figure caption detection
✓ Page rotation detection
✓ Language detection
✓ OCR confidence statistics

This module NEVER extracts text from PDFs.
That responsibility belongs to pdf_extractor.py.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional

import fitz

try:
    from langdetect import detect, DetectorFactory

    # deterministic language detection
    DetectorFactory.seed = 0

    LANGDETECT_AVAILABLE = True

except ImportError:

    LANGDETECT_AVAILABLE = False


# ==========================================================
# PDF DATE PARSER
# ==========================================================

def parse_pdf_date(date_string: Optional[str]) -> Optional[str]:
    """
    Convert PDF metadata dates into ISO format.

    PDF stores dates like

    D:20250526111409+05'30'

    Returns

    2025-05-26 11:14:09

    If parsing fails, returns original string.
    """

    if not date_string:
        return None

    try:

        date_string = date_string.strip()

        if date_string.startswith("D:"):
            date_string = date_string[2:]

        dt = datetime.strptime(
            date_string[:14],
            "%Y%m%d%H%M%S"
        )

        return dt.strftime("%Y-%m-%d %H:%M:%S")

    except Exception:

        return date_string


# ==========================================================
# PDF METADATA
# ==========================================================

def extract_pdf_metadata(
    doc: fitz.Document
) -> Dict:
    """
    Extract metadata stored inside the PDF.

    Returns
    -------
    {
        title,
        author,
        subject,
        keywords,
        creator,
        producer,
        creation_date,
        modification_date
    }
    """

    metadata = doc.metadata or {}

    return {

        "title":
            metadata.get("title"),

        "author":
            metadata.get("author"),

        "subject":
            metadata.get("subject"),

        "keywords":
            metadata.get("keywords"),

        "creator":
            metadata.get("creator"),

        "producer":
            metadata.get("producer"),

        "creation_date":
            parse_pdf_date(
                metadata.get("creationDate")
            ),

        "modification_date":
            parse_pdf_date(
                metadata.get("modDate")
            )
    }


# ==========================================================
# PAGE ROTATION
# ==========================================================

def extract_page_rotations(
    doc: fitz.Document
) -> Dict[int, int]:
    """
    Extract rotation angle of every page.

    Returns
    -------

    {
        1: 0,
        2: 90,
        3: 0
    }
    """

    rotations = {}

    for page in doc:

        rotations[page.number + 1] = page.rotation

    return rotations


# ==========================================================
# LANGUAGE DETECTION
# ==========================================================

def detect_document_language(
    text: str
) -> str:
    """
    Detect primary language of the document.

    Uses only first few thousand characters for speed.

    Returns
    -------
    "en"

    or

    "unknown"
    """

    if not LANGDETECT_AVAILABLE:

        return "unknown"

    sample = text.strip()

    if not sample:

        return "unknown"

    sample = sample[:5000]

    try:

        return detect(sample)

    except Exception:

        return "unknown"


# ==========================================================
# CAPTION REGEX
# ==========================================================

CAPTION_PATTERN = re.compile(

    r"^\s*"
    r"(Figure|Fig\.?|Image|Chart|Diagram|Table)"
    r"\s*"
    r"([0-9IVXivx\-\.]*)?"
    r"\s*:?"
    r"\s*(.+)?$",

    re.IGNORECASE
)

# ==========================================================
# FIGURE CAPTION DETECTION
# ==========================================================

def extract_figure_captions(
    pages: List[Dict]
) -> List[Dict]:
    """
    Detect figure/table captions from extracted text.

    Returns
    -------
    [
        {
            "page": 2,
            "caption": "Figure 1 Claim Workflow"
        }
    ]
    """

    captions = []

    for page in pages:

        page_number = page["page_number"]

        for line in page["cleaned_text"].splitlines():

            line = line.strip()

            if not line:
                continue

            if CAPTION_PATTERN.match(line):

                captions.append({
                    "page": page_number,
                    "caption": line
                })

    return captions


# ==========================================================
# EMBEDDED IMAGE DETECTION
# ==========================================================

def extract_images(
    doc: fitz.Document
) -> List[Dict]:
    """
    Detect embedded raster images.

    Does NOT perform OCR.

    Returns
    -------
    [
        {
            "page": 2,
            "image_index": 1,
            "width": 900,
            "height": 600,
            "colorspace": 3,
            "xref": 27
        }
    ]
    """

    images = []

    for page in doc:

        page_number = page.number + 1

        page_images = page.get_images(full=True)

        for idx, image in enumerate(page_images, start=1):

            try:

                xref = image[0]

                pix = fitz.Pixmap(doc, xref)

                images.append({

                    "page": page_number,

                    "image_index": idx,

                    "xref": xref,

                    "width": pix.width,

                    "height": pix.height,

                    "colorspace": pix.n

                })

                pix = None

            except Exception:

                continue

    return images

# ==========================================================
# OCR CONFIDENCE ESTIMATION
# ==========================================================

def estimate_ocr_confidence(
    pages: List[Dict]
) -> Dict:
    """
    Estimate OCR usage statistics.

    NOTE
    ----
    This is not Tesseract's real confidence score.

    Since OCR is only invoked as a fallback in pdf_ocr.py,
    we currently report:

    - pages using OCR
    - percentage of document OCR'd

    Real OCR confidence can be added later using
    pytesseract.image_to_data().

    Returns
    -------
    {
        "ocr_pages": 2,
        "ocr_percentage": 16.67
    }
    """

    total_pages = len(pages)

    if total_pages == 0:
        return {
            "ocr_pages": 0,
            "ocr_percentage": 0.0
        }

    ocr_pages = sum(
        1
        for page in pages
        if page.get("used_ocr", False)
    )

    return {

        "ocr_pages": ocr_pages,

        "ocr_percentage": round(
            (ocr_pages / total_pages) * 100,
            2
        )
    }


# ==========================================================
# DOCUMENT ENRICHMENT
# ==========================================================

def enrich_pdf(
    pdf_path: str,
    pages: List[Dict],
    cleaned_text: str
) -> Dict:
    """
    Perform higher-level PDF enrichment.

    Parameters
    ----------
    pdf_path : str

    pages : output from pdf_extractor

    cleaned_text : full cleaned document

    Returns
    -------
    {
        pdf_metadata,
        page_rotations,
        language,
        figures,
        images,
        image_count,
        ocr_statistics
    }
    """

    doc = fitz.open(pdf_path)

    try:

        metadata = extract_pdf_metadata(doc)

        rotations = extract_page_rotations(doc)

        images = extract_images(doc)

        captions = extract_figure_captions(pages)

        language = detect_document_language(
            cleaned_text
        )

        ocr_stats = estimate_ocr_confidence(
            pages
        )

        return {

            "pdf_metadata": metadata,

            "page_rotations": rotations,

            "language": language,

            "figures": captions,

            "images": images,

            "image_count": len(images),

            "ocr_statistics": ocr_stats
        }

    finally:

        doc.close()


# ==========================================================
# STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    import json

    from backend.app.utils.pdf_extractor import extract_pdf

    pdf_path = input("PDF Path: ").strip()

    extracted = extract_pdf(pdf_path)

    enrichment = enrich_pdf(

        pdf_path,

        extracted["pages"],

        extracted["cleaned_text"]
    )

    print("=" * 80)
    print("PDF ENRICHMENT")
    print("=" * 80)

    print(json.dumps(
        enrichment,
        indent=2,
        ensure_ascii=False
    ))