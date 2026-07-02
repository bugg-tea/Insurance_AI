from __future__ import annotations

import math
import collections
from typing import List, Dict, Any, Optional, Set

from backend.app.utils.embeddings import search as faiss_search


# =========================================================
# CONFIG (TUNE THESE — THESE MATTER A LOT IN INSURANCE QA)
# =========================================================

VECTOR_WEIGHT = 0.70
BM25_WEIGHT = 0.30

# Chunk-type importance boosts (insurance domain tuned)
CHUNK_TYPE_BOOST = {
    "table_row": 0.10,
    "table_row_part": 0.10,
    "table_meta": 0.05,
    "clause": 0.12,
    "exclusion": 0.20,     # VERY IMPORTANT in insurance QA
    "definition": 0.08,
    "list": 0.05,
    "narrative": 0.00,
}

# Structural grouping boosts
TABLE_GROUP_BOOST = 0.08
ROW_NEIGHBOR_BOOST = 0.05


# =========================================================
# BM25 (lightweight but stable)
# =========================================================

class BM25:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.docs = documents
        self.k1 = 1.5
        self.b = 0.75

        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0

        self._build()

    def _tokenize(self, text: str):
        return text.lower().split()

    def _build(self):
        df = {}
        freq = []
        doc_len = []

        for doc in self.docs:
            tokens = self._tokenize(doc.get("text", ""))
            doc_len.append(len(tokens))

            f = collections.Counter(tokens)
            freq.append(f)

            for w in f:
                df[w] = df.get(w, 0) + 1

        self.doc_freqs = freq
        self.doc_len = doc_len
        self.avgdl = sum(doc_len) / max(len(doc_len), 1)

        N = len(self.docs)

        for word, f in df.items():
            self.idf[word] = math.log((N - f + 0.5) / (f + 0.5) + 1)

    def score(self, query: str) -> List[float]:
        q_tokens = self._tokenize(query)
        scores = []

        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            dl = self.doc_len[i]

            for q in q_tokens:
                if q not in doc_freq:
                    continue

                f = doc_freq[q]
                idf = self.idf.get(q, 0.0)

                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-6))

                score += idf * (num / (den + 1e-6))

            scores.append(score)

        return scores


# =========================================================
# CORE HYBRID ENGINE
# =========================================================

class HybridSearchEngine:
    """
    Context-aware hybrid retrieval:
    - FAISS semantic search
    - BM25 lexical search
    - structural boosting (tables, rows, exclusions)
    - neighborhood expansion (row-aware retrieval)
    """

    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.bm25 = BM25(docs)

        # fast lookup maps
        
        self.by_chunk_id = {}

        for d in docs:
            cid = d.get("chunk_id")
            if cid:
                self.by_chunk_id[cid] = d

        print("TOTAL DOCS INDEXED:", len(self.by_chunk_id))

        self.table_map = self._build_table_map()

    # -----------------------------------------------------
    # STRUCTURE INDEXING
    # -----------------------------------------------------

    def _build_table_map(self):
        """
        Groups chunks by table_id so we can:
        - boost table-level coherence
        - retrieve related rows
        """
        table_map = {}

        for d in self.docs:
            tid = d.get("table_id")
            if not tid:
                continue
            table_map.setdefault(tid, []).append(d)

        return table_map

    # -----------------------------------------------------
    # NEIGHBOR EXPANSION (VERY IMPORTANT)
    # -----------------------------------------------------

    def _expand_neighbors(self, chunk: Dict[str, Any]) -> Set[str]:
        """
        Adds:
        - previous_row
        - next_row
        - same table siblings
        """
        related = set()

        cid = chunk.get("chunk_id")

        # direct links (from chunker)
        if chunk.get("previous_row"):
            related.add(chunk["previous_row"])

        if chunk.get("next_row"):
            related.add(chunk["next_row"])

        # same table expansion
        tid = chunk.get("table_id")
        if tid and tid in self.table_map:
            for c in self.table_map[tid]:
                if c.get("chunk_id") != cid:
                    related.add(c.get("chunk_id"))

        return related

    # -----------------------------------------------------
    # STRUCTURAL SCORING BOOST
    # -----------------------------------------------------

    def _structure_boost(self, doc: Dict[str, Any]) -> float:
        boost = 0.0

        ctype = doc.get("chunk_type", "")
        boost += CHUNK_TYPE_BOOST.get(ctype, 0.0)

        # table coherence boost
        if doc.get("table_id"):
            boost += TABLE_GROUP_BOOST

        return boost

    # -----------------------------------------------------
    # MAIN SEARCH
    # -----------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:

    # ----------------------------
    # 1. VECTOR SEARCH
    # ----------------------------
        vector_results = faiss_search(query, top_k=top_k * 3)
        print("DEBUG VECTOR RESULTS:", len(vector_results))

    # ----------------------------
    # 2. BM25 SCORES
    # ----------------------------
        bm25_scores = self.bm25.score(query)
        bm25_norm = self._normalize(bm25_scores)
        
        bm25_map = {}

        for i, doc in enumerate(self.docs):
            cid = doc.get("chunk_id")
            if cid:
                bm25_map[cid] = bm25_norm[i]

        
    # ----------------------------
    # 3. CORE FUSION
    # ----------------------------
        results = []

        for v in vector_results:

            cid = v.get("chunk_id")

            doc = self.by_chunk_id.get(cid)

        # =====================================================
        # FIX: fallback to FAISS payload instead of dropping
        # =====================================================
            if not doc:
                doc = {
                    "chunk_id": cid,
                    "text": v.get("text", ""),
                    "chunk_type": v.get("chunk_type", "unknown"),
                    "page_start": v.get("page_start", 0),
                    "page_end": v.get("page_end", 0),
                    "table_id": v.get("table_id"),
                    "row_number": v.get("row_number"),
                    "final_score": 0.0
            }

            
            vector_score = float(v.get("score", 0.0))
            bm25_score = bm25_map.get(cid, 0.0)

            base_score = (
    VECTOR_WEIGHT * vector_score +
    BM25_WEIGHT * bm25_score
)

            struct_boost = self._structure_boost(doc)

            final_score = base_score + struct_boost

            
            results.append({
                **v,
                "bm25_score": bm25_score,
                "structure_boost": struct_boost,
                "retrieval_score": float(final_score),
    # INITIAL FINAL SCORE (SAME AS BASE HERE)
                "final_score": final_score,
        })

    # ----------------------------
    # 4. SORT
    # ----------------------------
        
        results.sort(key=lambda x: x["retrieval_score"], reverse=True)

    # ----------------------------
    # 5. CONTEXT EXPANSION
    # ----------------------------
        expanded = self._expand_results(results[:top_k * 2])
        expanded = self._rescore_expanded(expanded)

        expanded.sort(key=lambda x: x["final_score"], reverse=True)

        return expanded[:top_k]

    # -----------------------------------------------------
    # EXPANSION PIPELINE
    # -----------------------------------------------------

    def _expand_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        expanded = []

        for r in results:
            cid = r.get("chunk_id")
            if cid and cid in seen:
                continue

            if cid:
                seen.add(cid)

            expanded.append(r)

            doc = self.by_chunk_id.get(cid)
            if not doc:
                continue

            neighbors = self._expand_neighbors(doc)

            for nid in neighbors:
                if nid and nid in seen:
                    continue

                ndoc = self.by_chunk_id.get(nid)
                if not ndoc:
                    continue

                expanded.append({
                    "chunk_id": nid,
                    "text": ndoc.get("text"),
                    "chunk_type": ndoc.get("chunk_type"),
                    "page_start": ndoc.get("page_start"),
                    "page_end": ndoc.get("page_end"),
                    "table_id": ndoc.get("table_id"),
                    "row_number": ndoc.get("row_number"),
                    "expanded": True,
                    "final_score": r.get("retrieval_score", 0.0) 
                })

                if nid:
                    seen.add(nid)

        return expanded

    # -----------------------------------------------------
    # RE-SCORING AFTER EXPANSION
    # -----------------------------------------------------

    def _rescore_expanded(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for d in docs:
            base = d.get("final_score", 0.0)

            doc = self.by_chunk_id.get(d.get("chunk_id"))
            if not doc:
                continue

            d["final_score"] = base + self._structure_boost(doc)

        return docs

    # -----------------------------------------------------
    # UTIL
    # -----------------------------------------------------

    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return scores

        mx = max(scores)
        mn = min(scores)

        return [(s - mn) / (mx - mn + 1e-6) for s in scores]


# =========================================================
# FACTORY
# =========================================================


def build_hybrid_engine(all_chunks: List[Dict[str, Any]]):
    """
    Production entry point
    """
    return HybridSearchEngine(all_chunks)