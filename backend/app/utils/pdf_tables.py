"""
pdf_tables.py

Extract tables from PDF pages.

Features
--------
- Extracts tables using pdfplumber
- Cleans empty cells
- Removes empty rows
- Returns structured tables
- Converts tables into readable text
- Safe fallback on extraction errors
"""

from __future__ import annotations

from typing import Dict, List
import pdfplumber


# ---------------------------------------------------------------------
# Clean a single table
# ---------------------------------------------------------------------

def clean_table(table: List[List[str]]) -> List[List[str]]:
    """
    Clean a raw table extracted by pdfplumber.

    - Replace None with ""
    - Strip whitespace
    - Remove completely empty rows
    """

    cleaned = []

    for row in table:

        new_row = []

        for cell in row:
            if cell is None:
                new_row.append("")
            else:
                new_row.append(str(cell).strip())

        # Remove rows that are entirely empty
        if any(cell != "" for cell in new_row):
            cleaned.append(new_row)

    return cleaned


# ---------------------------------------------------------------------
# Convert a table into readable text
# ---------------------------------------------------------------------

def table_to_text(table: List[List[str]]) -> str:
    """
    Convert a structured table into text.

    Example

    Name | Age
    John | 30

    becomes

    Name | Age
    John | 30
    """

    lines = []

    for row in table:
        lines.append(" | ".join(row))

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Extract tables from a pdfplumber page
# ---------------------------------------------------------------------
def extract_tables_from_page(
    page: pdfplumber.page.Page,
) -> Dict:
    
    """
    Extract all tables from a page.

    Returns
    -------
    {
        "tables": [...],
        "text": "..."
    }
    """

    structured_tables = []
    text_blocks = []

    try:

        tables = page.extract_tables()

        if not tables:
            return {
                "tables": [],
                "text": ""
            }

        for table in tables:

            cleaned = clean_table(table)

            if not cleaned:
                continue

            structured_tables.append(cleaned)

            text_blocks.append(table_to_text(cleaned))

    except Exception:
        return {
            "tables": [],
            "text": ""
        }

    return {
        "tables": structured_tables,
        "text": "\n\n".join(text_blocks)
    }


# ---------------------------------------------------------------------
# Extract every table from an entire PDF
# ---------------------------------------------------------------------

def extract_tables_from_pdf(pdf_path: str):
    """
    Extract tables from every page.

    Returns
    -------
    {
        "tables": [
            {
                "page": 1,
                "table": [...]
            }
        ],
        "text": "..."
    }
    """

    all_tables = []
    text_blocks = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            result = extract_tables_from_page(page)

            if result["tables"]:

                for table in result["tables"]:
                    all_tables.append({
                        "page": page_number,
                        "table": table
                    })

                text_blocks.append(
                    f"\n===== PAGE {page_number} TABLES =====\n"
                )

                text_blocks.append(result["text"])

    return {
        "tables": all_tables,
        "text": "\n".join(text_blocks)
    }


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    pdf_path = input("PDF Path: ").strip()

    result = extract_tables_from_pdf(pdf_path)

    print("=" * 80)
    print("TOTAL TABLES:", len(result["tables"]))
    print("=" * 80)

    print(result["text"][:5000])