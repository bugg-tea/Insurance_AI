"""
pdf_extract.py

Main PDF extraction engine.

Responsibilities
----------------
✓ Layout-aware extraction
✓ OCR fallback
✓ Table extraction
✓ Duplicate removal
✓ Page-wise metadata
✓ Character statistics
✓ Production-ready error handling

This module DOES NOT clean text.
Cleaning is handled by pdf_cleaner.py.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import fitz
import pdfplumber

from backend.app.utils.pdf_layout import extract_page_text
from backend.app.utils.pdf_tables import extract_tables_from_page
from backend.app.utils.pdf_ocr import get_page_text

from backend.app.utils.pdf_cleaner import (
    clean_text,
    pages_to_document,
)
from backend.app.utils.pdf_enrichment import enrich_pdf
# ============================================================
# Duplicate removal
# ============================================================

def normalize_for_compare(text: str) -> str:
    """
    Normalize text before comparing.

    Used to detect duplicated content coming from
    PyMuPDF + pdfplumber.
    """

    return " ".join(
        text.lower().split()
    )


def remove_duplicate_blocks(blocks: List[str]) -> List[str]:
    """
    Remove duplicated paragraphs.

    Example

    PyMuPDF:
        Hospitalization Cover

    pdfplumber:
        Hospitalization Cover

    Keep only one.
    """

    seen = set()

    output = []

    for block in blocks:

        block = block.strip()

        if not block:
            continue

        key = hashlib.md5(
            normalize_for_compare(block).encode("utf8")
        ).hexdigest()

        if key in seen:
            continue

        seen.add(key)

        output.append(block)

    return output


# ==========================================================
# PAGE EXTRACTION
# ==========================================================
def extract_page(
    fitz_page: fitz.Page,
    plumber_page,
    form_mode: bool = False
) -> Dict:
    """
    Extract one page using all available methods.

    Returns
    -------
    {
        page_number,
        raw_text,
        cleaned_text,
        used_ocr,
        tables,
        table_text,
        raw_char_count,
        clean_char_count
    }
    """

    page_number = fitz_page.number + 1

    # ------------------------------------------------------
    # Layout-aware extraction
    # ------------------------------------------------------

    
    layout_text = extract_page_text(fitz_page)

# pdfplumber extraction
    plumber_text = plumber_page.extract_text() or ""

# Choose the richer extraction
    candidate = layout_text

    if len(plumber_text) > len(candidate):
        candidate = plumber_text

    # ------------------------------------------------------
    # OCR fallback
    # ------------------------------------------------------

    
    final_text = get_page_text(
        fitz_page,
        candidate,
        form_mode=form_mode
)

    
    used_ocr = final_text != layout_text

    # ------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------

    table_result = extract_tables_from_page(plumber_page)

    tables = table_result["tables"]
    table_text = table_result["text"]

    # ------------------------------------------------------
    # Merge page text + tables
    # ------------------------------------------------------

    merged = final_text


    # ------------------------------------------------------
    # Remove duplicate paragraphs
    # ------------------------------------------------------

    
    blocks = remove_duplicate_blocks(
        merged.split("\n\n")
)

    merged = "\n\n".join(blocks)

    # ------------------------------------------------------
    # Clean page
    # ------------------------------------------------------

    cleaned = clean_text(merged)

    return {

        "page_number": page_number,

        "raw_text": merged,

        "cleaned_text": cleaned,

        "used_ocr": used_ocr,

        "tables": tables,

        "table_text": table_text,

        "raw_char_count": len(merged),

        "clean_char_count": len(cleaned)
    }


# ==========================================================
# DOCUMENT EXTRACTION
# ==========================================================

def extract_document(
    pdf_path: str
) -> Dict:
    """
    Extract an entire PDF.

    Returns
    -------
    {
        raw_text,
        cleaned_text,
        pages,
        tables,
        raw_char_count,
        clean_char_count,
        used_ocr_pages
    }
    """

    doc = fitz.open(pdf_path)

    plumber_pdf = pdfplumber.open(pdf_path)
    
    try:

        pages = []

        all_tables = []

        used_ocr_pages = []

        raw_document = []

        cleaned_document = []
        
        form_mode = any(
            keyword in pdf_path.lower()
            for keyword in (
                "claim",
                "preauth"
    )
)

        for fitz_page, plumber_page in zip(
            doc,
            plumber_pdf.pages
    ):

            page = extract_page(
                fitz_page,
                plumber_page,
                form_mode=form_mode
)
            
            pages.append(page)

            raw_document.append(
                f"\n========== PAGE {page['page_number']} ==========\n"
        )
            raw_document.append(page["raw_text"])

            cleaned_document.append(
                f"\n========== PAGE {page['page_number']} ==========\n"
        )
            cleaned_document.append(page["cleaned_text"])

            if page["tables"]:

                for table in page["tables"]:

                    all_tables.append({

                        "page": page["page_number"],

                        "table": table

                })

            if page["used_ocr"]:

                used_ocr_pages.append(
                    page["page_number"]
            )
    # ------------------------------------------------------
    # Close PDFs
    # ------------------------------------------------------
    finally:
        plumber_pdf.close()
        doc.close()

    # ------------------------------------------------------
    # Remove repeated headers/footers across pages
    # ------------------------------------------------------

    cleaned_pages = [
        {
            "page_number": p["page_number"],
            "text": p["cleaned_text"]
        }
        for p in pages
    ]
      

    # Update cleaned page text after header/footer removal
    for page, cleaned in zip(pages, cleaned_pages):
        page["cleaned_text"] = cleaned["text"]
        page["clean_char_count"] = len(cleaned["text"])

    # ------------------------------------------------------
    # Build final cleaned document
    # ------------------------------------------------------

    cleaned_document = pages_to_document(cleaned_pages)

    raw_document = "\n".join(raw_document)
    
    # ------------------------------------------------------
# Document enrichment
# ------------------------------------------------------

    enrichment = enrich_pdf(
        pdf_path,
        pages,
        cleaned_document
)

    return {

        "raw_text": raw_document,

        "cleaned_text": cleaned_document,

        "pages": pages,

        "tables": all_tables,

        "raw_char_count": len(raw_document),

        "clean_char_count": len(cleaned_document),

        "page_count": len(pages),

        "table_count": len(all_tables),

        "used_ocr_pages": used_ocr_pages,
        
        "enrichment": enrichment
    }


# ==========================================================
# PUBLIC API
# ==========================================================

def extract_pdf(pdf_path: str) -> Dict:
    """
    Public extraction function.

    This is the only function that should be imported
    by pdf_loader.py.

    Returns
    -------
    {
        raw_text,
        cleaned_text,
        pages,
        tables,
        raw_char_count,
        clean_char_count,
        page_count,
        table_count,
        used_ocr_pages
    }
    """

    return extract_document(pdf_path)


# ==========================================================
# STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    pdf_path = input("PDF Path: ").strip()

    result = extract_pdf(pdf_path)

    print("=" * 80)
    print("DOCUMENT SUMMARY")
    print("=" * 80)

    print(f"Pages            : {result['page_count']}")
    print(f"Tables           : {result['table_count']}")
    print(f"OCR Pages        : {result['used_ocr_pages']}")
    print(f"Raw Characters   : {result['raw_char_count']}")
    print(f"Clean Characters : {result['clean_char_count']}")

    print("\n")
    print("=" * 80)
    print("FIRST 5000 CHARACTERS")
    print("=" * 80)

    print(result["cleaned_text"][:5000])