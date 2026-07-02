from __future__ import annotations

from typing import List, Dict, Any
from collections import defaultdict

from tqdm import tqdm
from gliner import GLiNER


# ============================================================
# LABELS
# ============================================================

ENTITY_LABELS = [
    # Insurance
    "Policy",
    "Coverage",
    "Benefit",
    "Exclusion",
    "Waiting Period",
    "Premium",
    "Sum Insured",
    "Deductible",
    "Claim",

    # Medical
    "Disease",
    "Procedure",
    "Treatment",
    "Hospital",
    "Medicine",
    "Doctor",

    # Administrative
    "Document",
    "Person",
    "Organization",
    "Location"
]


# ============================================================
# ENTITY EXTRACTOR
# ============================================================

class InsuranceEntityExtractor:
    """
    Production-ready batched GLiNER entity extraction.

    Output format remains identical to your previous version,
    but processing is significantly faster.
    """

    def __init__(
        self,
        model_name: str = "urchade/gliner_small-v2.1",
        threshold: float = 0.45,
        batch_size: int = 16,
    ):

        print("Loading GLiNER model...")

        self.model = GLiNER.from_pretrained(model_name)

        self.threshold = threshold
        self.batch_size = batch_size

    # -------------------------------------------------------

    @staticmethod
    def _clean_predictions(predictions):

        entities = []
        seen = set()

        for p in predictions:

            value = p["text"].strip()
            label = p["label"]

            key = (value.lower(), label)

            if key in seen:
                continue

            seen.add(key)

            entities.append({
                "text": value,
                "label": label,
                "score": round(float(p["score"]), 4)
            })

        return entities

    # -------------------------------------------------------

    def extract_document(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        texts = [c.get("text", "") for c in chunks]

        total = len(texts)

        for start in tqdm(
            range(0, total, self.batch_size),
            desc="Extracting entities"
        ):

            end = min(start + self.batch_size, total)

            batch_texts = texts[start:end]

            # Batch inference
            batch_predictions = self.model.batch_predict_entities(
                batch_texts,
                ENTITY_LABELS,
                threshold=self.threshold,
            )

            for chunk, preds in zip(chunks[start:end], batch_predictions):

                chunk["entities"] = self._clean_predictions(preds)

        return chunks

    # -------------------------------------------------------

    def extract_chunk(
        self,
        chunk: Dict[str, Any]
    ) -> Dict[str, Any]:

        text = chunk.get("text", "")

        preds = self.model.predict_entities(
            text,
            ENTITY_LABELS,
            threshold=self.threshold,
        )

        chunk["entities"] = self._clean_predictions(preds)

        return chunk

    # -------------------------------------------------------

    def build_entity_index(
        self,
        chunks: List[Dict[str, Any]]
    ):

        entity_map = defaultdict(list)

        for chunk in chunks:

            cid = chunk["chunk_id"]

            for ent in chunk.get("entities", []):

                entity_map[
                    ent["text"].lower()
                ].append(cid)

        return dict(entity_map)