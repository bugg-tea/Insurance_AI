import json
from pathlib import Path

from backend.app.graph.entity_extractor import InsuranceEntityExtractor

CHUNK_FOLDER = Path("backend/data/chunks")

extractor = InsuranceEntityExtractor()

for json_file in CHUNK_FOLDER.glob("*.json"):

    print("=" * 100)
    print(f"Processing: {json_file.name}")
    print("=" * 100)

    with open(json_file, "r", encoding="utf8") as f:
        data = json.load(f)

    chunks = data["chunks"]

    chunks = extractor.extract_document(chunks)

    # Print first 5 chunks only
    for chunk in chunks[:5]:
        print("-" * 80)
        print(chunk["text"])
        print()
        print(chunk["entities"])

