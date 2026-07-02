"""
json_debug.py

Utility to inspect stored extracted JSON files.

Reads an extracted JSON file and recreates the
same debug output as pdf_debug.py without
re-extracting the PDF.
"""

from __future__ import annotations

import json
import os
import sys


def debug_json(json_path: str) -> None:

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        document = json.load(f)

    print("\n" + "=" * 80)
    print("DOCUMENT METADATA")
    print("=" * 80)

    print(f"Insurer          : {document['metadata']['insurer']}")
    print(f"Document Type    : {document['metadata']['document_type']}")
    print(f"File             : {document['metadata']['file_name']}")
    print(f"Pages            : {document['page_count']}")
    print(f"Tables           : {document['table_count']}")
    print(f"OCR Pages        : {document['used_ocr_pages']}")
    print(f"Raw Characters   : {document['raw_char_count']}")
    print(f"Clean Characters : {document['clean_char_count']}")

    print("\n" + "=" * 80)
    print("DOCUMENT ENRICHMENT")
    print("=" * 80)

    enrichment = document["enrichment"]

    print(f"Language         : {enrichment['language']}")
    print(f"Images           : {enrichment['image_count']}")
    print(f"Figure Captions  : {len(enrichment['figures'])}")
    print(f"OCR Percentage   : {enrichment['ocr_statistics']['ocr_percentage']}%")

    print("\nPDF Metadata")

    for key, value in enrichment["pdf_metadata"].items():
        print(f"{key:20}: {value}")

    print("\nPage Rotations")

    for page, rotation in enrichment["page_rotations"].items():
        print(f"Page {page}: {rotation}°")

    print("\n" + "=" * 80)
    print("PAGE SUMMARY")
    print("=" * 80)

    for page in document["pages"]:

        print(
            f"Page {page['page_number']:>3} | "
            f"OCR={page['used_ocr']} | "
            f"Tables={len(page['tables'])} | "
            f"Raw={page['raw_char_count']} | "
            f"Clean={page['clean_char_count']}"
        )

    os.makedirs(
        "debug_output",
        exist_ok=True
    )

    with open(
        "debug_output/raw_text.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(document["raw_text"])

    with open(
        "debug_output/clean_text.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(document["clean_text"])

    with open(
        "debug_output/document.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            document,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved:")
    print("  debug_output/raw_text.txt")
    print("  debug_output/clean_text.txt")
    print("  debug_output/document.json")


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage:")
        print("python json_debug.py path/to/file.json")
        sys.exit(1)

    debug_json(sys.argv[1])