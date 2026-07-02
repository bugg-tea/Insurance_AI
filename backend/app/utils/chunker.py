import os
import json
import uuid
from typing import List, Dict, Any

# =========================
# CONFIG (UPDATED)
# =========================

DATA_FOLDER = r"C:\Users\DELL LATITUDE\Desktop\project\backend\data\extracted"
CHUNK_OUTPUT_FOLDER = r"C:\Users\DELL LATITUDE\Desktop\project\backend\data\chunks"

# =========================
# FILE STATE CACHE (avoid reprocessing)
# =========================

def get_file_signature(doc: Dict[str, Any]) -> str:
    """
    Used to detect if file changed
    """
    meta = doc.get("metadata", {})
    return f"{meta.get('document_id')}_{meta.get('last_modified')}"

# =========================
# CHECK IF ALREADY PROCESSED
# =========================

def is_already_chunked(file_name: str, signature: str) -> bool:
    """
    Skip processing if unchanged file exists
    """

    os.makedirs(CHUNK_OUTPUT_FOLDER, exist_ok=True)

    expected_file = os.path.join(
        CHUNK_OUTPUT_FOLDER,
        f"{file_name}_chunks.json"
    )

    if not os.path.exists(expected_file):
        return False

    try:
        with open(expected_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("signature") == signature

    except:
        return False

# =========================
# SAVE CHUNKS PER PDF
# =========================

def save_pdf_chunks(file_name: str, signature: str, chunks: List[Dict[str, Any]]):
    """
    Store chunks per PDF (NOT global)
    """

    os.makedirs(CHUNK_OUTPUT_FOLDER, exist_ok=True)

    output_path = os.path.join(
        CHUNK_OUTPUT_FOLDER,
        f"{file_name}_chunks.json"
    )

    payload = {
        "file_name": file_name,
        "signature": signature,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved chunks: {output_path}")
# =========================
# UTIL: ID GENERATOR
# =========================

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# =========================
# 1. LOAD DOCUMENT
# =========================

def load_document(file_path: str) -> Dict[str, Any]:
    """
    Load extracted JSON document
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 2. GET ALL DOCUMENT FILES
# =========================

def get_all_documents(folder_path: str) -> List[str]:
    """
    Returns list of JSON file paths
    """
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            files.append(os.path.join(folder_path, file))
    return files


# =========================
# 3. SPLIT PAGES
# =========================

def split_pages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts raw document into page-based structure
    """
    pages = doc.get("pages", [])

    structured_pages = []

    for p in pages:
        structured_pages.append({
            "page_number": p.get("page_number"),
            "raw_text": p.get("raw_text", ""),
            "cleaned_text": p.get("cleaned_text", ""),
            "tables": p.get("tables", []),
        })

    return structured_pages


# =========================
# 4. DOCUMENT NORMALIZER
# =========================

def normalize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardizes document format
    """
    return {
        "metadata": doc.get("metadata", {}),
        "pages": split_pages(doc)
    }


# =========================
# 6. TABLE PROCESSING ENGINE
# =========================


def extract_tables_from_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract raw tables from page
    """
    return page.get("tables", [])


# -------------------------
# FLATTEN TABLE ROWS
# -------------------------

def flatten_table(table: List[Any]) -> List[List[str]]:
    """
    Converts raw table structure into clean row format
    """
    rows = []

    for row in table:
        if isinstance(row, list):
            cleaned_row = []
            for cell in row:
                if isinstance(cell, list):
                    cleaned_row.append(" ".join([str(x) for x in cell]))
                else:
                    cleaned_row.append(str(cell))
            rows.append(cleaned_row)

    return rows


# -------------------------
# PARENT ROW DETECTION
# -------------------------

def is_parent_row(row: List[str]) -> bool:
    """
    Heuristic to detect header/parent rows
    """

    if not row:
        return False

    text = " ".join(row).strip()

    # Rule 1: very short left + descriptive right
    if len(row) >= 2:
        left, right = row[0], row[1]

        if len(left.split()) <= 4 and len(right.split()) >= 6:
            return True

    # Rule 2: all caps header style
    if text.isupper() and len(text.split()) <= 8:
        return True

    # Rule 3: category-like keywords
    keywords = [
        "coverage", "benefit", "plan", "eligibility",
        "policy", "details", "sum insured"
    ]

    if any(k in text.lower() for k in keywords):
        if len(text.split()) <= 12:
            return True

    return False


# -------------------------
# TABLE ROW STRUCTURING
# -------------------------

def structure_table_rows(table: List[Any], page_number: int):
    """
    Converts raw table into structured rows with IDs
    """

    rows = flatten_table(table)

    structured_rows = []
    parent_context = None

    table_id = generate_id("T")

    for i, row in enumerate(rows):
        row_id = f"{table_id}_R{i}"

        if is_parent_row(row):
            parent_context = " ".join(row)

        structured_rows.append({
            "table_id": table_id,
            "row_id": row_id,
            "page_number": page_number,
            "cells": row,
            "text": " | ".join(row),
            "parent_row": parent_context
        })

    return structured_rows


# -------------------------
# PROCESS ALL TABLES IN PAGE
# -------------------------

def process_page_tables(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Main entry for table processing per page
    """

    page_number = page.get("page_number")
    tables = extract_tables_from_page(page)

    all_rows = []

    for table in tables:
        structured_rows = structure_table_rows(table, page_number)
        all_rows.extend(structured_rows)

    return all_rows

# =========================
# 7. TEXT / SEMANTIC ENGINE
# =========================


import re
from typing import Tuple


# -------------------------
# SECTION DETECTION
# -------------------------

SECTION_HEADERS = [
    "eligibility criteria",
    "plan details",
    "about us",
    "wait periods",
    "policy details",
    "optional benefits",
    "coverage",
    "benefits"
]


def detect_section(text: str) -> str:
    """
    Detect nearest section heading
    """
    text_lower = text.lower()

    for section in SECTION_HEADERS:
        if section in text_lower:
            return section.title()

    return "General"


# -------------------------
# SPLIT TEXT INTO LINES
# -------------------------

def split_lines(text: str) -> List[str]:
    """
    Clean and split text into meaningful lines
    """
    if not text:
        return []

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        line = line.strip()

        if len(line) < 2:
            continue

        cleaned.append(line)

    return cleaned


# -------------------------
# CLAUSE DETECTOR
# -------------------------

def is_policy_clause(line: str) -> bool:
    """
    Detect policy clause patterns
    """

    patterns = [
        r"up to",
        r"coverage",
        r"sum insured",
        r"benefit",
        r"shall",
        r"applicable",
        r"limit",
        r"deductible",
        r"co[- ]?payment"
    ]

    text = line.lower()

    return any(re.search(p, text) for p in patterns)


# -------------------------
# EXCLUSION DETECTOR
# -------------------------

def is_exclusion(line: str) -> bool:
    """
    Detect exclusions
    """

    keywords = [
        "not covered",
        "excluded",
        "does not cover",
        "waiting period",
        "pre-existing"
    ]

    text = line.lower()

    return any(k in text for k in keywords)


# -------------------------
# CLAIM RULE DETECTOR
# -------------------------

def is_claim_rule(line: str) -> bool:
    """
    Detect claim-related rules
    """

    keywords = [
        "co-payment",
        "deductible",
        "claim",
        "reimbursement",
        "limit per year"
    ]

    text = line.lower()

    return any(k in text for k in keywords)


# -------------------------
# SEMANTIC CHUNK MERGER
# -------------------------

def merge_small_chunks(lines: List[str], window: int = 4, overlap: int = 2) -> List[str]:
    """
    Sliding window overlap chunking for better retrieval
    """

    if not lines:
        return []

    chunks = []
    i = 0

    while i < len(lines):

        window_lines = lines[i:i + window]

        if not window_lines:
            break

        chunks.append(" ".join(window_lines))

        i += (window - overlap)

    return chunks


# -------------------------
# MAIN TEXT CHUNKER
# -------------------------

def process_text_block(text: str, page_number: int) -> List[Dict[str, Any]]:
    """
    Converts raw page text into structured semantic chunks
    """

    lines = split_lines(text)
    merged_lines = merge_small_chunks(lines)

    chunks = []

    for line in merged_lines:

        section = detect_section(line)

        if is_exclusion(line):
            chunk_type = "exclusion"
        elif is_claim_rule(line):
            chunk_type = "claim_rule"
        elif is_policy_clause(line):
            chunk_type = "policy_clause"
        else:
            chunk_type = "general_info"

        chunks.append({
            "chunk_type": chunk_type,
            "page_number": page_number,
            "section": section,
            "text": line
        })

    return chunks

# =========================
# 8. FINAL CHUNK ASSEMBLY ENGINE
# =========================


def build_final_chunk(
    chunk_type: str,
    text: str,
    page_number: int,
    metadata: Dict[str, Any],
    document_type: str = None,
    enrichment: Dict[str, Any] = None,   # ✅ NEW
    table_id: str = None,
    row_id: str = None,
    section: str = None
) -> Dict[str, Any]:

    return {
        "chunk_id": generate_id("C"),
        "chunk_type": chunk_type,

        # document context
        "document_type": document_type,
        "insurer": metadata.get("insurer"),

        # page context
        "page_number": page_number,

        # structure context
        "table_id": table_id,
        "row_id": row_id,
        "section": section,

        # text
        "text": text,

        # ✅ NEW: PDF-level enrichment (VERY IMPORTANT)
        "enrichment": enrichment,

        # full metadata
        "metadata": metadata
    }
# =========================
# 9. UNIFIED CHUNK PIPELINE
# =========================

def process_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Combines table + text pipelines into final chunks
    (NOW INCLUDES enrichment + document-level metadata properly)
    """

    final_chunks = []

    metadata = doc.get("metadata", {})
    pages = doc.get("pages", [])

    # ✅ NEW: document-level fields
    document_type = metadata.get("document_type", "unknown")
    enrichment = doc.get("enrichment", {})   # ⭐ FIXED HERE

    for page in pages:

        page_number = page["page_number"]

        # =========================
        # TABLE CHUNKS
        # =========================
        table_rows = process_page_tables(page)

        for row in table_rows:

            final_chunks.append(
                build_final_chunk(
                    chunk_type="table_row",
                    text=row["text"],
                    page_number=page_number,
                    metadata=metadata,
                    document_type=document_type,
                    enrichment=enrichment,   # ✅ FIX ADDED
                    table_id=row.get("table_id"),
                    row_id=row.get("row_id"),
                    section="Table Data"
                )
            )

        # =========================
        # TEXT CHUNKS
        # =========================
        text_chunks = process_text_block(page["cleaned_text"], page_number)

        for t in text_chunks:

            final_chunks.append(
                build_final_chunk(
                    chunk_type=t["chunk_type"],
                    text=t["text"],
                    page_number=page_number,
                    metadata=metadata,
                    document_type=document_type,
                    enrichment=enrichment,   # ✅ FIX ADDED
                    section=t.get("section", "General")
                )
            )

    return final_chunks


# =========================
# 10. SAVE OUTPUT
# =========================

def save_chunks(chunks: List[Dict[str, Any]], file_path: str):
    """
    Save final chunks to JSON file
    """

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


# =========================
# 11. UPDATED PIPELINE RUNNER
# =========================

def process_all_documents(folder_path: str):
    """
    Production pipeline:
    - per PDF chunking
    - per-file output
    - skip unchanged files
    """

    files = get_all_documents(folder_path)

    print(f"Found {len(files)} documents")

    for file_path in files:

        print(f"\nProcessing: {file_path}")

        doc = load_document(file_path)
        doc = normalize_document(doc)

        metadata = doc.get("metadata", {})
        file_name = metadata.get("file_name", "unknown_file")

        signature = get_file_signature(doc)

        # -------------------------
        # SKIP IF ALREADY DONE
        # -------------------------
        if is_already_chunked(file_name, signature):
            print(f"⏭ Skipping (already processed): {file_name}")
            continue

        print(f"Pages: {len(doc['pages'])}")

        # -------------------------
        # GENERATE CHUNKS
        # -------------------------
        chunks = process_document(doc)

        print(f"Generated Chunks: {len(chunks)}")

        # -------------------------
        # SAVE PER FILE
        # -------------------------
        save_pdf_chunks(
            file_name=file_name.replace(".pdf", ""),
            signature=signature,
            chunks=chunks
        )

# =========================
# RUN
# =========================

if __name__ == "__main__":
    process_all_documents(DATA_FOLDER)