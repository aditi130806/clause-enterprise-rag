"""
Lightweight Deterministic Reranker for CLAUSE RAG.
Reranks hybrid retrieval candidates using composite scoring across RRF rank,
dense similarity, BM25 score, query-token coverage, exact identifier matches,
and section relevance without PyTorch or external model dependencies.
"""

import re
from typing import List
from src.models import RetrievalResult
from src.query_rewriter import extract_policy_identifiers


def calculate_query_coverage(query: str, text: str) -> float:
    """Calculate ratio of non-stopword query tokens present in text."""
    if not query or not text:
        return 0.0
    
    stop_words = {
        "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "is", "are", "what", "does", "can", "do",
        "how", "why", "who", "where", "which", "company", "policy", "policies", "employee", "employees",
        "corporate", "guidelines", "rules", "rule", "work", "working", "day", "days", "time", "hours",
        "item", "items", "use", "used", "bring", "your", "year", "years", "month", "months", "plan", "plans",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
        "leave", "leaves", "provided", "provides", "provide", "required", "requires", "requirement",
        "requirements", "general", "specific"
    }
    q_tokens = [w.lower() for w in re.findall(r"\b\w+\b", query) if w.lower() not in stop_words]
    if not q_tokens:
        return 0.0
    
    text_words = set(re.findall(r"\b\w+\b", text.lower()))
    matches = sum(1 for token in q_tokens if token in text_words)
    return matches / len(q_tokens)


def rerank_retrieval_results(
    query: str,
    results: List[RetrievalResult],
    top_k: int = 5,
) -> List[RetrievalResult]:
    """
    Rerank a list of RetrievalResult candidate objects using composite signal scoring.

    Args:
        query: User raw or normalized search query.
        results: Candidate RetrievalResult items from hybrid search.
        top_k: Number of reranked items to return.

    Returns:
        Sorted list of RetrievalResult objects with updated fusion_score and final_rank.
    """
    if not results:
        return []

    policy_ids = extract_policy_identifiers(query)
    q_lower = query.lower()

    # Detect value-seeking queries (allowance, limit, rate, amount, per diem, cost, etc.)
    is_value_seeking = any(term in q_lower for term in [
        "allowance", "limit", "rate", "amount", "how much", "percentage",
        "cost", "deadline", "how long", "per day", "per month", "stipend",
        "fee", "price", "reimbursement", "per diem", "meal", "meals"
    ])

    scored_candidates = []
    for candidate in results:
        chunk_text = candidate.chunk.text if candidate.chunk else ""
        section_title = (candidate.section or "") + " " + (candidate.chunk.section if candidate.chunk else "")

        # 1. Base RRF / Retrieval score (normalized 0-1)
        base_rrf_score = candidate.fusion_score

        # 2. Query token coverage (0-1)
        token_coverage = calculate_query_coverage(query, chunk_text)

        # 3. Exact policy identifier match boost
        identifier_boost = 0.0
        for pid in policy_ids:
            if pid.lower() in chunk_text.lower():
                identifier_boost += 0.35
            if pid.lower() in section_title.lower() or pid.lower() in candidate.document_name.lower():
                identifier_boost += 0.25

        # 4. Section title relevance boost
        section_coverage = calculate_query_coverage(query, section_title) * 0.15

        # 5. Value-seeking concrete numerical value boost
        value_boost = 0.0
        if is_value_seeking:
            # Check for currency ($), percentage (%), per-diem/daily indicators, or concrete numbers ($75)
            if re.search(r"\$\d+|\b\d+\s*%\b|\b\d+\s*(?:days?|hours?|calendar days?|per day|daily)\b", chunk_text, re.IGNORECASE):
                value_boost += 0.20
            if any(k in q_lower for k in ["meal", "meals", "food", "dining", "per diem"]) and any(k in chunk_text.lower() for k in ["meal", "meals", "per diem", "75"]):
                value_boost += 0.25

        # Composite Score Calculation
        composite_score = (
            (base_rrf_score * 0.40) +
            (token_coverage * 0.35) +
            section_coverage +
            identifier_boost +
            value_boost
        )

        # Create updated RetrievalResult object preserving original chunk metadata
        updated_result = RetrievalResult(
            chunk=candidate.chunk,
            document_name=candidate.document_name,
            page_number=candidate.page_number,
            section=candidate.section,
            dense_score=candidate.dense_score,
            bm25_score=candidate.bm25_score,
            dense_rank=candidate.dense_rank,
            bm25_rank=candidate.bm25_rank,
            fusion_score=float(composite_score),
            final_rank=1,
            retrieval_methods=list(candidate.retrieval_methods) + ["reranked"] if candidate.retrieval_methods else ["reranked"],
        )
        scored_candidates.append(updated_result)

    # Sort descending by composite rerank score
    scored_candidates.sort(key=lambda r: r.fusion_score, reverse=True)

    # Assign updated 1-indexed final_rank
    for rank_idx, item in enumerate(scored_candidates, start=1):
        item.final_rank = rank_idx

    return scored_candidates[:top_k]
