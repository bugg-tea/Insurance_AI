
# Optional dependency (recommended)
from __future__ import annotations

from typing import List, Dict, Any, Tuple

import numpy as np
import math

# Optional dependency — imported lazily inside CrossEncoderReranker.__init__
# so that SimpleReranker (the free, no-model fallback) still works even if
# sentence-transformers / torch are not installed.
_CROSS_ENCODER_IMPORT_ERROR = None
try:
    from sentence_transformers import CrossEncoder
except Exception as e:  # pragma: no cover
    CrossEncoder = None
    _CROSS_ENCODER_IMPORT_ERROR = e


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
# =========================================================
# CONFIG
# =========================================================

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Insurance-specific boosts AFTER rerank
EXTRA_BOOSTS = {
    "exclusion": 0.05,
    "clause": 0.03,
    "table_row": 0.02,
}


# =========================================================
# CROSS ENCODER RERANKER
# =========================================================

class CrossEncoderReranker:
    """
    Re-ranks FAISS + BM25 candidates using a cross-encoder
    (query, document) scoring model.

    This is MUCH more accurate than embedding similarity because:
    - it performs full attention between query and chunk
    - understands negation ("not covered", "excluded", etc.)
    - crucial for insurance QA
    """
    
    

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not installed, so the cross-encoder "
                f"reranker is unavailable ({_CROSS_ENCODER_IMPORT_ERROR}). "
                "Run: pip install sentence-transformers --break-system-packages "
                "or use build_reranker(use_cross_encoder=False) for the free "
                "SimpleReranker fallback."
            )
        self.model = CrossEncoder(model_name)

    # -----------------------------------------------------
    # SCORE PAIRS
    # -----------------------------------------------------

    def score(self, query: str, docs: List[Dict[str, Any]]) -> List[float]:
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = self.model.predict(pairs)

        return scores.tolist()
    
    
    # -----------------------------------------------------
    # MAIN RERANK FUNCTION
    # -----------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:

        if not candidates:
            return []

        scores = self.score(query, candidates)
        

        reranked = []

        for doc, score in zip(candidates, scores):

            boost = EXTRA_BOOSTS.get(doc.get("chunk_type", ""), 0.0)

            
            
            
            
            
            


            
            base = float(score)   # cross encoder raw

# normalize to 0–1
            base_norm = 1 / (1 + math.exp(-base))

            rerank_score = base_norm + boost
            
            reranked.append({
                **doc,
                "cross_encoder_score": float(score),
                "rerank_boost": boost,

    # FIXED
                "rerank_score": float(rerank_score),
                
            })

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# =========================================================
# LIGHTWEIGHT FALLBACK (NO MODEL ENV)
# =========================================================

class SimpleReranker:
    """
    Fallback if CrossEncoder is not available.
    Uses heuristic scoring only.
    """

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:

        def score(doc: Dict[str, Any]) -> float:
            text = doc.get("text", "").lower()
            q = query.lower()

            base = 0.0

            # keyword overlap (very weak signal but stable fallback)
            for w in q.split():
                if w in text:
                    base += 0.1

            base += EXTRA_BOOSTS.get(doc.get("chunk_type", ""), 0.0)

            return base

        scored = []

        for d in candidates:
            scored.append({
                **d,
                "final_score": score(d)
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)

        return scored[:top_k]


# =========================================================
# FACTORY
# =========================================================


def build_reranker(use_cross_encoder: bool = True):
    if use_cross_encoder:
        try:
            return CrossEncoderReranker()
        except Exception as e:
            print(f"⚠️  Cross-encoder unavailable, falling back to SimpleReranker: {e}")
            return SimpleReranker()
    return SimpleReranker()