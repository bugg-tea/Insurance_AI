"""
pdf_loader.py

Main PDF loader for the project.

Responsibilities
----------------
✓ Scan Dataset folder
✓ Detect insurer & document type
✓ Call pdf_extract.extract_pdf()
✓ Build production-ready document objects
✓ Return all extracted documents
"""

from __future__ import annotations

import json
import os
from typing import Dict, List
import hashlib
from tqdm import tqdm

from backend.app.utils.pdf_extractor import extract_pdf

# ==========================================================
# DATASET ROOT
# ==========================================================

DATA_PATH = "Dataset"

# ==========================================================
# EXTRACTED JSON STORAGE
# ==========================================================

EXTRACTED_PATH = "backend/data/extracted"
# ==========================================================
# DOCUMENT TYPES
# ==========================================================

SUPPORTED_DOC_TYPES = {

    "Policy": "policy",

    "Claim": "claim",

    "CIS": "cis",

    "Coverage": "coverage",

    "Exclusions": "exclusions",

    "Brochure": "brochure",

    "PreAuth": "preauth",

    "Proposal": "proposal",

    "Policy Usage Guide": "usage_guide"
}


# ==========================================================
# Detect metadata from folder structure
# ==========================================================

def detect_metadata(file_path: str) -> Dict:

    """
    Example

    Dataset/
        Care Health/
            Policy/
                abc.pdf
    """

    parts = os.path.normpath(file_path).split(os.sep)

    insurer = "Unknown"

    folder_type = "Unknown"

    if len(parts) >= 3:

        insurer = parts[1]

        folder_type = parts[2]

    return {

        "insurer": insurer,

        "document_type": SUPPORTED_DOC_TYPES.get(
            folder_type,
            "other"
        ),

        "folder_type": folder_type,

        "file_name": os.path.basename(file_path),

        "file_path": file_path
    }


# ==========================================================
# JSON STORAGE
# ==========================================================

def save_extracted_json(document: Dict) -> None:
    """
    Save one extracted document as JSON.

    Each PDF is stored separately.

    backend/data/extracted/

        Care_Health_policy_xxxxx.json
    """

    os.makedirs(
        EXTRACTED_PATH,
        exist_ok=True
    )

    metadata = document["metadata"]

    insurer = metadata["insurer"].replace(" ", "_")

    

    document_type = metadata["document_type"]

    # Stable document id using file path
    
    document_id = metadata["document_id"]

    json_name = (
        f"{insurer}_"
        f"{document_type}_"
        f"{document_id}.json"
    )

    output_path = os.path.join(
        EXTRACTED_PATH,
        json_name
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            document,
            f,
            indent=2,
            ensure_ascii=False
        )
        
    print(f"Saved: {output_path}")
        
# ==========================================================
# Load one PDF
# ==========================================================

def load_pdf(file_path: str) -> Dict:

    """
    Load a single PDF.
    """

    
    metadata = detect_metadata(file_path)
    
    with open(metadata["file_path"], "rb") as f:
        file_bytes = f.read()

    metadata["document_id"] = hashlib.sha256(
        file_bytes
    ).hexdigest()

    metadata["file_size"] = len(file_bytes)
 
    metadata["last_modified"] = os.path.getmtime(
        metadata["file_path"]
)



    

    extracted = extract_pdf(file_path)

    document = {

        "metadata": metadata,

        "raw_text": extracted["raw_text"],

        "clean_text": extracted["cleaned_text"],

        "pages": extracted["pages"],

        "tables": extracted["tables"],

        "raw_char_count": extracted["raw_char_count"],

        "clean_char_count": extracted["clean_char_count"],

        "page_count": extracted["page_count"],

        "table_count": extracted["table_count"],

        "used_ocr_pages": extracted["used_ocr_pages"],
        
        "enrichment": extracted["enrichment"]
    }
    
    save_extracted_json(document)
    return document


# ==========================================================
# Scan Dataset
# ==========================================================

def load_all_pdfs(
    data_path: str = DATA_PATH
) -> List[Dict]:

    """
    Scan Dataset recursively.

    Returns

    List[Document]
    """

    documents = []

    pdf_files = []

    for root, _, files in os.walk(data_path):

        for file in files:

            if file.lower().endswith(".pdf"):

                pdf_files.append(
                    os.path.join(root, file)
                )

    print(f"\nFound {len(pdf_files)} PDF files.\n")

    for pdf in tqdm(pdf_files, desc="Extracting PDFs"):

        try:

            document = load_pdf(pdf)

            documents.append(document)

        except Exception as e:

            print()

            print(f"❌ Failed : {pdf}")

            print(e)

    print()

    print("=" * 60)

    print("PDF LOADING COMPLETE")

    print("=" * 60)

    print(f"Loaded PDFs : {len(documents)}")

    return documents


# ==========================================================
# Save sample output
# ==========================================================

def save_debug_output(
    documents,
    output_path="data/metadata/raw_docs.json",
    sample_size=3
):

    """
    Saves sample extracted documents.

    Useful for inspection.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents[:sample_size],
            f,
            indent=2,
            ensure_ascii=False
        )
        
        

    print()

    print(f"Saved sample to {output_path}")


# ==========================================================
# Summary
# ==========================================================

def print_summary(documents):

    print()

    print("=" * 60)

    print("DATASET SUMMARY")

    print("=" * 60)

    total_pages = sum(
        d["page_count"]
        for d in documents
    )

    total_tables = sum(
        d["table_count"]
        for d in documents
    )

    total_raw = sum(
        d["raw_char_count"]
        for d in documents
    )

    total_clean = sum(
        d["clean_char_count"]
        for d in documents
    )

    print(f"Documents        : {len(documents)}")
    print(f"Pages            : {total_pages}")
    print(f"Tables           : {total_tables}")
    print(f"Raw Characters   : {total_raw:,}")
    print(f"Clean Characters : {total_clean:,}")

    print("=" * 60)


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    docs = load_all_pdfs()

    print_summary(docs)

    save_debug_output(docs)