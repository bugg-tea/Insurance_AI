from __future__ import annotations

from typing import List, Dict, Any, Tuple
import re

from backend.app.retrieval.hybrid_search import build_hybrid_engine
from backend.app.retrieval.reranker import build_reranker


# =========================================================
# 1. QUERY INTENT DETECTOR (INSURANCE SPECIALIZED)
# =========================================================

class QueryRouter:
    """
    Converts raw user query into structured intent signals.
    This is the MOST important upgrade for insurance QA.
    """

    def __init__(self):
        # regex-based lightweight classifier (fast, no LLM needed)
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

        scores = {}

        for intent, pattern in self.patterns.items():
            if re.search(pattern, q):
                scores[intent] = 1.0
            else:
                scores[intent] = 0.0

        # default fallback
        if sum(scores.values()) == 0:
            scores["general"] = 1.0

        return scores


# =========================================================
# 2. ADAPTIVE SCORING RULES (THE MAGIC LAYER)
# =========================================================

class AdaptiveScorer:
    """
    Applies query-aware boosts to retrieval candidates.
    This is what makes insurance QA feel "intelligent".
    """

    def __init__(self):
        pass
    
    def apply(self, intent: Dict[str, float], docs: List[Dict[str, Any]]):

        updated = []

        for d in docs:
            boost = 0.0
            text = d.get("text", "").lower()
            ctype = d.get("chunk_type", "")

            if intent.get("exclusion"):
                boost += 0.35 if ctype == "exclusion" else 0.0
                boost += 0.25 if "not covered" in text else 0.0

            if intent.get("coverage"):
                boost += 0.15 if ctype in ("table_row", "table_meta") else 0.0
                boost += 0.10 if "covered" in text else 0.0

            if intent.get("claim"):
                boost += 0.25 if "claim" in text else 0.0

            if intent.get("premium"):
                boost += 0.25 if "premium" in text else 0.0

            if intent.get("waiting_period"):
                boost += 0.30 if "waiting" in text else 0.0

            new_doc = dict(d)
            new_doc["intent_boost"] = boost
            new_doc["final_score"] = new_doc.get("final_score", 0.0) + boost

            updated.append(new_doc)

        return updated   

    

# =========================================================
# 3. FINAL RETRIEVAL PIPELINE
# =========================================================

class RetrievalPipeline:
    """
    Orchestrates:
    - Query understanding
    - Hybrid retrieval (Part 1)
    - Adaptive scoring (Part 3)
    - Cross encoder reranking (Part 2)
    """

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.router = QueryRouter()
        self.scorer = AdaptiveScorer()

        self.hybrid = build_hybrid_engine(chunks)
        self.reranker = build_reranker(use_cross_encoder=True)

    # -----------------------------------------------------
    # MAIN ENTRY
    # -----------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:

        # ----------------------------
        # STEP 1: INTENT DETECTION
        # ----------------------------
        intent = self.router.detect_intent(query)

        # ----------------------------
        # STEP 2: HYBRID RETRIEVAL
        # ----------------------------
        candidates = self.hybrid.search(query, top_k=top_k * 3)

        # ----------------------------
        # STEP 3: APPLY INTENT BOOSTS
        # ----------------------------
        candidates = self.scorer.apply(intent, candidates)

        # merge intent boost into final score
        for c in candidates:
            base = c.get("final_score", 0.0)
            c["final_score"] = base + c.get("intent_boost", 0.0)

        # ----------------------------
        # STEP 4: RERANKING
        # ----------------------------
        reranked = self.reranker.rerank(query, candidates, top_k=top_k * 2)

        # ----------------------------
        # STEP 5: CONTEXT PACKING
        # ----------------------------
        final = self._pack_context(reranked[:top_k])

        return final

    # -----------------------------------------------------
    # CONTEXT PACKER (IMPORTANT FOR LLM QUALITY)
    # -----------------------------------------------------

    def _pack_context(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        packed = []

        for d in docs:
            packed.append({
                "chunk_id": d.get("chunk_id"),
                "text": d.get("text"),
                "score": d.get("final_score"),
                "chunk_type": d.get("chunk_type"),
                "section": d.get("section"),
                "page_start": d.get("page_start"),
                "page_end": d.get("page_end"),
                "table_id": d.get("table_id"),
                "row_number": d.get("row_number"),
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
            hints.append(f"row_{d.get('row_number')}")

        return "|".join(hints)


# =========================================================
# 4. FACTORY FUNCTION (PRODUCTION ENTRY POINT)
# =========================================================

def build_retrieval_pipeline(all_chunks: List[Dict[str, Any]]):
    return RetrievalPipeline(all_chunks)