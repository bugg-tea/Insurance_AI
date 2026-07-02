from __future__ import annotations

from typing import List, Dict, Any

from backend.app.retrieval.hybrid_search import build_hybrid_engine
from backend.app.retrieval.reranker import build_reranker


# =========================================================
# QUERY ROUTER
# =========================================================

import re


class QueryRouter:
    def __init__(self):
        self.patterns = {
            "exclusion": r"\b(exclude|exclusion|not covered|not payable|denied)\b",
            "coverage": r"\b(cover|coverage|covered|benefit|eligible)\b",
            "claim": r"\b(claim|claim process|how to claim|settlement)\b",
            "premium": r"\b(premium|cost|price|amount)\b",
            "waiting_period": r"\b(waiting period|wait time|cooling)\b",
            "definition": r"\b(what is|define|meaning of)\b",
        }

    def detect_intent(self, query: str) -> Dict[str, float]:
        q = query.lower()
        intent = {k: 0.0 for k in self.patterns}

        matched = False

        for k, p in self.patterns.items():
            if re.search(p, q):
                intent[k] = 1.0
                matched = True

        if not matched:
            intent["general"] = 1.0

        return intent


# =========================================================
# ADAPTIVE SCORER (PURE FUNCTION STYLE)
# =========================================================

class AdaptiveScorer:

    def apply(self, intent: Dict[str, float], docs: List[Dict[str, Any]]):

        updated = []

        for d in docs:
            boost = 0.0
            text = d.get("text", "").lower()
            ctype = d.get("chunk_type", "")

            if intent.get("exclusion"):
                if ctype == "exclusion":
                    boost += 0.35
                if "not covered" in text:
                    boost += 0.25

            if intent.get("coverage"):
                if ctype in ("table_row", "table_meta"):
                    boost += 0.15
                if "covered" in text:
                    boost += 0.10

            if intent.get("claim") and "claim" in text:
                boost += 0.25

            if intent.get("premium") and "premium" in text:
                boost += 0.25

            if intent.get("waiting_period") and "waiting" in text:
                boost += 0.30

            new_doc = dict(d)
            new_doc["intent_boost"] = boost

            updated.append(new_doc)

        return updated


# =========================================================
# FINAL RETRIEVAL PIPELINE
# =========================================================

class RetrievalPipeline:

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.router = QueryRouter()
        self.scorer = AdaptiveScorer()

        self.hybrid = build_hybrid_engine(chunks)
        self.reranker = build_reranker(use_cross_encoder=True)

    # -----------------------------------------------------
    # MAIN ENTRY
    # -----------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:

    # -------------------------------------------------
    # STEP 1 : Detect intent
    # -------------------------------------------------

        intent = self.router.detect_intent(query)

    # -------------------------------------------------
    # STEP 2 : Hybrid Retrieval
    # -------------------------------------------------

        candidates = self.hybrid.search(
            query,
            top_k=top_k * 3
    )

        print("AFTER HYBRID:", len(candidates))

    # -------------------------------------------------
    # STEP 3 : Apply Intent Boost
    # -------------------------------------------------

        candidates = self.scorer.apply(intent, candidates)

        for c in candidates:
            c["intent_boost"] = c.get("intent_boost", 0.0)

            c["final_score"] = float(c["final_score"]) + c["intent_boost"]
        
        print("AFTER INTENT BOOST:", len(candidates))

    # -------------------------------------------------
    # STEP 4 : Cross Encoder
    # -------------------------------------------------

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k * 2
    )

        if not reranked:
            return self._pack_context(candidates[:top_k])

    # -------------------------------------------------
    # STEP 5 : Blend Hybrid + CrossEncoder
    # -------------------------------------------------
    

        

        for c in reranked:
            hybrid = c.get("retrieval_score", 0.0)
            rerank = c.get("rerank_score", hybrid)
            intent = c.get("intent_boost", 0.0)

            c["final_score"] = (
                0.5 * hybrid +
                0.4 * rerank +
                0.1 * intent
    )
        

    # -------------------------------------------------
    # STEP 6 : Reconstruct split table rows
    # -------------------------------------------------

        reranked = self._reconstruct_rows(reranked)

    # -------------------------------------------------
    # STEP 7 : Pack surrounding context
    # -------------------------------------------------

        reranked = self._pack_related_context(reranked)

    # -------------------------------------------------
    # STEP 8 : Final sort
    # -------------------------------------------------

        reranked.sort(
            key=lambda x: x["final_score"],
            reverse=True
    )

    # -------------------------------------------------
    # STEP 9 : Return
    # -------------------------------------------------

        return self._pack_context(
            reranked[:top_k]
    )
        

        

    # -----------------------------------------------------
    # CONTEXT PACKER
    # -----------------------------------------------------

    def _pack_context(
        self,
        docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        packed = []

        for d in docs:

            packed.append({

            "chunk_id": d.get("chunk_id"),

            "text": d.get("text"),

            
            "final_score": d.get("final_score", 0.0),
            "retrieval_score": d.get("final_score", 0.0),
            "score": d.get("final_score", 0.0),
            "rerank_score": d.get("rerank_score"),

            "chunk_type": d.get("chunk_type"),

            "section": d.get("section"),

            "subsection": d.get("subsection"),

            "page_start": d.get("page_start"),

            "page_end": d.get("page_end"),

            "table_id": d.get("table_id"),

            "row_number": d.get("row_number"),

            # Will simply be None until you regenerate chunks
            "hierarchy": d.get("hierarchy_path"),

            "context_hint": self._build_hint(d)

        })

        return packed
    
    
    def _build_hint(self, d: Dict[str, Any]) -> str:
        hints = []

        if d.get("table_id"):
            hints.append("table_context")

        if d.get("chunk_type") == "exclusion":
            hints.append("high_priority_exclusion")

        if d.get("chunk_type") == "clause":
            hints.append("legal_clause")

        if d.get("row_number") is not None:
            hints.append(f"row_{d['row_number']}")

        return "|".join(hints)

    # -----------------------------------------------------
    # RECONSTRUCT COMPLETE TABLE ROWS
    # -----------------------------------------------------

    def _reconstruct_rows(
        self,
        docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        grouped = {}
        output = []

        for doc in docs:

            if doc.get("chunk_type") != "table_row_part":
                output.append(doc)
                continue

            key = (
            doc.get("table_id"),
            doc.get("row_number")
        )

            grouped.setdefault(key, []).append(doc)

        for _, parts in grouped.items():

            parts.sort(
            key=lambda x: x.get("part_number", 0)
        )

            merged = dict(parts[0])

            merged["text"] = "\n".join(
            p.get("text", "")
            for p in parts
        )

            merged["chunk_type"] = "table_row"

            merged["final_score"] = max(
            p.get("final_score", 0.0)
            for p in parts
        )

            output.append(merged)

        return output
    
    # -----------------------------------------------------
    # PACK RELATED CONTEXT
    # -----------------------------------------------------

    def _pack_related_context(
        self,
        docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        packed = []

        grouped = {}

        for doc in docs:

            key = (

            doc.get("table_id"),

            doc.get("section"),

            doc.get("page_start")

        )

            grouped.setdefault(key, []).append(doc)

        for _, group in grouped.items():

            group.sort(
                key=lambda x: (
                    x.get("row_number", -1),
                    x.get("part_number", 0)
            )
        )

            merged = dict(group[0])

            merged["text"] = "\n\n".join(
            d.get("text", "")
            for d in group
        )

            merged["final_score"] = max(
            d.get("final_score", 0.0)
            for d in group
        )

            packed.append(merged)

        return packed
# =========================================================
# FACTORY
# =========================================================

def build_retrieval_pipeline(all_chunks: List[Dict[str, Any]]):
    return RetrievalPipeline(all_chunks)