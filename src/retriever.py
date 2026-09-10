"""
Hybrid Retrieval Engine module for CLAUSE Enterprise RAG.

Combines FAISS dense vector retrieval and BM25 sparse keyword retrieval using
Reciprocal Rank Fusion (RRF).
Does not perform direct arithmetic addition of unscaled scores.
Returns rich RetrievalResult objects with full source attribution, ranks, and metadata.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from src.config import DEFAULT_RRF_K, DEFAULT_CANDIDATE_TOP_N, DEFAULT_FINAL_TOP_K
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.keyword_search import BM25SearchIndex
from src.models import DocumentChunk, RetrievalResult


class RetrieverError(Exception):
    """Custom exception raised for retrieval errors."""
    pass


class HybridRetriever:
    """
    Orchestrates dense (FAISS), sparse (BM25), and hybrid (RRF) retrieval strategies.
    Supports simple metadata filtering (e.g. document_name, department).
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        bm25_index: BM25SearchIndex,
        embedding_service: EmbeddingService,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_top_n: int = DEFAULT_CANDIDATE_TOP_N,
        default_top_k: int = DEFAULT_FINAL_TOP_K,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_service = embedding_service
        self.rrf_k = rrf_k
        self.candidate_top_n = candidate_top_n
        self.default_top_k = default_top_k

    def _matches_filters(self, chunk: DocumentChunk, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a chunk satisfies optional metadata filters (e.g., document_name, department).

        Args:
            chunk: Candidate DocumentChunk instance.
            filters: Dictionary of field -> expected_value requirements.

        Returns:
            True if all filter criteria are met; False otherwise.
        """
        if not filters:
            return True

        for filter_key, filter_val in filters.items():
            if filter_val is None:
                continue

            # Check direct chunk attributes
            if hasattr(chunk, filter_key):
                chunk_val = getattr(chunk, filter_key)
                if chunk_val != filter_val:
                    return False
            # Check nested metadata dict
            elif filter_key in chunk.metadata:
                if chunk.metadata[filter_key] != filter_val:
                    return False
            else:
                return False

        return True

    def retrieve_dense(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform dense vector-only retrieval using FAISS.

        Args:
            query: User search query string.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of RetrievalResult objects ranked by dense cosine score.
        """
        k = top_k or self.default_top_k
        if not query or not query.strip():
            return []

        query_vec = self.embedding_service.encode_query(query)
        # Fetch larger candidate pool if filters present
        search_k = self.candidate_top_n if filters else k
        raw_results = self.vector_store.search(query_vec, top_k=search_k)

        results: List[RetrievalResult] = []
        rank_counter = 1

        for chunk, score, orig_rank in raw_results:
            if not self._matches_filters(chunk, filters):
                continue

            res = RetrievalResult(
                chunk=chunk,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                section=chunk.section,
                dense_score=score,
                bm25_score=0.0,
                dense_rank=orig_rank,
                bm25_rank=None,
                fusion_score=score,
                final_rank=rank_counter,
                retrieval_methods=["dense"],
            )
            results.append(res)
            rank_counter += 1
            if len(results) >= k:
                break

        return results

    def retrieve_keyword(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform BM25 sparse keyword-only retrieval.

        Args:
            query: User search query string.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of RetrievalResult objects ranked by BM25 score.
        """
        k = top_k or self.default_top_k
        if not query or not query.strip():
            return []

        search_k = self.candidate_top_n if filters else k
        raw_results = self.bm25_index.search(query, top_k=search_k)

        results: List[RetrievalResult] = []
        rank_counter = 1

        for chunk, score, orig_rank in raw_results:
            if not self._matches_filters(chunk, filters):
                continue

            res = RetrievalResult(
                chunk=chunk,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                section=chunk.section,
                dense_score=0.0,
                bm25_score=score,
                dense_rank=None,
                bm25_rank=orig_rank,
                fusion_score=score,
                final_rank=rank_counter,
                retrieval_methods=["keyword"],
            )
            results.append(res)
            rank_counter += 1
            if len(results) >= k:
                break

        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_top_n: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval combining FAISS dense and BM25 sparse results using RRF.

        RRF Score Formula:
            score(d) = sum_{m in {dense, keyword}} (1 / (k_rrf + rank_m(d)))

        Args:
            query: User search query string.
            top_k: Final top-K results to return.
            candidate_top_n: Number of candidates to fetch from each retriever.
            filters: Optional metadata filters.

        Returns:
            List of RetrievalResult objects sorted by RRF fusion score.
        """
        final_k = top_k or self.default_top_k
        cand_n = candidate_top_n or self.candidate_top_n

        if not query or not query.strip():
            return []

        # 1. Fetch dense candidates
        query_vec = self.embedding_service.encode_query(query)
        dense_candidates = self.vector_store.search(query_vec, top_k=cand_n)

        # 2. Fetch BM25 candidates
        bm25_candidates = self.bm25_index.search(query, top_k=cand_n)

        # Map chunk_id -> tracking info
        candidate_map: Dict[str, Dict[str, Any]] = {}

        # Process dense candidates
        for chunk, score, rank in dense_candidates:
            if not self._matches_filters(chunk, filters):
                continue
            candidate_map[chunk.chunk_id] = {
                "chunk": chunk,
                "dense_score": score,
                "dense_rank": rank,
                "bm25_score": 0.0,
                "bm25_rank": None,
                "methods": ["dense"],
            }

        # Process BM25 candidates
        for chunk, score, rank in bm25_candidates:
            if not self._matches_filters(chunk, filters):
                continue
            cid = chunk.chunk_id
            if cid in candidate_map:
                candidate_map[cid]["bm25_score"] = score
                candidate_map[cid]["bm25_rank"] = rank
                if "keyword" not in candidate_map[cid]["methods"]:
                    candidate_map[cid]["methods"].append("keyword")
            else:
                candidate_map[cid] = {
                    "chunk": chunk,
                    "dense_score": 0.0,
                    "dense_rank": None,
                    "bm25_score": score,
                    "bm25_rank": rank,
                    "methods": ["keyword"],
                }

        if not candidate_map:
            return []

        # Calculate Reciprocal Rank Fusion (RRF) scores
        fused_items = []
        for cid, info in candidate_map.items():
            rrf_score = 0.0
            if info["dense_rank"] is not None:
                rrf_score += 1.0 / (self.rrf_k + info["dense_rank"])
            if info["bm25_rank"] is not None:
                rrf_score += 1.0 / (self.rrf_k + info["bm25_rank"])

            fused_items.append((cid, rrf_score, info))

        # Sort by fusion_score descending; break ties by chunk_id for determinism
        fused_items.sort(key=lambda item: (-item[1], item[0]))

        # Assemble final RetrievalResult objects
        results: List[RetrievalResult] = []
        for final_rank, (cid, fusion_score, info) in enumerate(fused_items[:final_k], start=1):
            chunk = info["chunk"]
            res = RetrievalResult(
                chunk=chunk,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                section=chunk.section,
                dense_score=info["dense_score"],
                bm25_score=info["bm25_score"],
                dense_rank=info["dense_rank"],
                bm25_rank=info["bm25_rank"],
                fusion_score=fusion_score,
                final_rank=final_rank,
                retrieval_methods=info["methods"],
            )
            results.append(res)

        return results

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Unified retrieval entry point supporting 'hybrid', 'dense', and 'keyword' modes.

        Args:
            query: User search query string.
            mode: Retrieval mode ('hybrid', 'dense', 'keyword'). Default: 'hybrid'.
            top_k: Number of top results to return.
            filters: Optional metadata filter dictionary.

        Returns:
            List of RetrievalResult objects.
        """
        mode_str = mode.lower().strip()
        if mode_str == "hybrid":
            return self.retrieve_hybrid(query=query, top_k=top_k, filters=filters)
        elif mode_str == "dense":
            return self.retrieve_dense(query=query, top_k=top_k, filters=filters)
        elif mode_str == "keyword" or mode_str == "bm25":
            return self.retrieve_keyword(query=query, top_k=top_k, filters=filters)
        else:
            raise RetrieverError(
                f"Unsupported retrieval mode '{mode}'. Expected one of ['hybrid', 'dense', 'keyword']."
            )
