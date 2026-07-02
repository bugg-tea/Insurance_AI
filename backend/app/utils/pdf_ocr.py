"""
pdf_ocr.py

OCR utilities for scanned PDF pages.

Uses:
- Tesseract OCR
- PIL
- PyMuPDF

OCR is only used as a fallback when embedded text extraction fails.
"""

from __future__ import annotations

import io
import os
from typing import Optional

import fitz
import pytesseract
from PIL import Image

# ---------------------------------------------------------------------
# Configure Tesseract
# ---------------------------------------------------------------------

DEFAULT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(DEFAULT_WINDOWS_PATH):
    
    
    pytesseract.pytesseract.tesseract_cmd = os.getenv(
        "TESSERACT_CMD",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"   # fallback for local Windows dev
)

# ---------------------------------------------------------------------
# OCR SETTINGS
# ---------------------------------------------------------------------

# OEM 3 = Default LSTM engine
# PSM 6 = Uniform block of text

OCR_CONFIG_TEXT = r"--oem 3 --psm 6"

OCR_CONFIG_FORM = r"--oem 3 --psm 4"

# 300 DPI gives much better OCR than default rendering
OCR_DPI = 300


# ---------------------------------------------------------------------
# Render PDF page to PIL Image
# ---------------------------------------------------------------------

def page_to_image(
    page: fitz.Page,
    dpi: int = OCR_DPI
) -> Image.Image:
    """
    Convert a PDF page into a high-resolution PIL image.
    """

    pix = page.get_pixmap(dpi=dpi, alpha=False)

    img = Image.open(
        io.BytesIO(
            pix.tobytes("png")
        )
    )

    return img

def symbol_ratio(text: str) -> float:

    if not text:
        return 1

    symbols = sum(
        not ch.isalnum() and not ch.isspace()
        for ch in text
    )

    return symbols / len(text)
# ---------------------------------------------------------------------
# OCR a PIL image
# ---------------------------------------------------------------------
def ocr_image(
    image: Image.Image,
    lang: str = "eng",
    form_mode: bool = False
) -> str:

    config = OCR_CONFIG_FORM if form_mode else OCR_CONFIG_TEXT

    try:
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=config
        )

        return text.strip()

    except Exception:
        return ""


# ---------------------------------------------------------------------
# OCR a PDF page
# ---------------------------------------------------------------------


def ocr_page(
    page: fitz.Page,
    lang: str = "eng",
    form_mode: bool = False
) -> str:

    image = page_to_image(page)

    return ocr_image(
        image,
        lang=lang,
        form_mode=form_mode
    )
# ---------------------------------------------------------------------
# Decide whether OCR is needed
# ---------------------------------------------------------------------

def needs_ocr(
    extracted_text: Optional[str],
    threshold: int = 40
) -> bool:
    """
    Determine if OCR should be used.

    OCR is triggered when:

    • no text
    • whitespace only
    • very few characters

    Parameters
    ----------
    extracted_text : str

    threshold : int

    Returns
    -------
    bool
    """

    if extracted_text is None:
        return True

    stripped = extracted_text.strip()

    if len(stripped) < threshold:
        return True
    
    
    
    if symbol_ratio(stripped) > 0.20:
        return True

    return False


# ---------------------------------------------------------------------
# OCR only if required
# ---------------------------------------------------------------------
def get_page_text(
    page,
    extracted_text,
    form_mode=False
):
    """
    Return embedded text if available.

    Otherwise perform OCR.
    """

    if needs_ocr(extracted_text):
        return ocr_page(
            page,
            form_mode=form_mode
)
        
    return extracted_text


# ---------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    sample_pdf = input("PDF Path: ").strip()

    doc = fitz.open(sample_pdf)

    for i, page in enumerate(doc):

        embedded = page.get_text()

        text = get_page_text(page, embedded)

        print("=" * 80)
        print(f"PAGE {i + 1}")
        print("=" * 80)
        print(text[:1500])