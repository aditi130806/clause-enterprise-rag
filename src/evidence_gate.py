"""
Evidence Sufficiency Gate module for CLAUSE Enterprise RAG.

Evaluates retrieved candidate document chunks prior to LLM generation.
Prevents ungrounded queries from reaching the LLM when retrieval evidence is weak or absent.
"""

from typing import List, Tuple, Optional, Dict, Any

from src.config import SUFFICIENCY_MIN_SCORE, SUFFICIENCY_MIN_CHUNKS
from src.models import RetrievalResult

INSUFFICIENT_INFORMATION_REFUSAL = (
    "I could not find sufficient information in the provided documents to answer this question."
)


def evaluate_evidence_sufficiency(
    retrieved_results: List[RetrievalResult],
    min_score: float = SUFFICIENCY_MIN_SCORE,
    min_chunks: int = SUFFICIENCY_MIN_CHUNKS,
) -> Tuple[bool, float, Optional[str], Dict[str, Any]]:
    """
    Evaluate whether retrieved chunks contain sufficient evidence to answer the user question.

    Args:
        retrieved_results: List of RetrievalResult objects from HybridRetriever.
        min_score: Minimum required top retrieval score (fusion_score, dense_score, or bm25_score).
        min_chunks: Minimum number of candidate chunks required.

    Returns:
        Tuple of:
        - is_sufficient (bool): True if evidence meets threshold criteria; False otherwise.
        - confidence (float): Estimated heuristic evidence confidence score [0.0, 1.0].
        - refusal_reason (Optional[str]): Reason description if insufficient; None if sufficient.
        - diagnostics (Dict[str, Any]): Detailed diagnostic metadata for inspection.
    """
    total_retrieved = len(retrieved_results)

    diagnostics: Dict[str, Any] = {
        "total_retrieved": total_retrieved,
        "min_score_threshold": min_score,
        "min_chunks_threshold": min_chunks,
        "top_fusion_score": 0.0,
        "top_dense_score": 0.0,
        "top_bm25_score": 0.0,
        "gate_passed": False,
        "refusal_reason": None,
    }

    if total_retrieved == 0:
        diagnostics["refusal_reason"] = "Zero document chunks retrieved for query."
        return False, 0.0, diagnostics["refusal_reason"], diagnostics

    if total_retrieved < min_chunks:
        diagnostics["refusal_reason"] = f"Retrieved chunk count ({total_retrieved}) below minimum ({min_chunks})."
        return False, 0.0, diagnostics["refusal_reason"], diagnostics

    top_result = retrieved_results[0]
    top_fusion = top_result.fusion_score
    top_dense = top_result.dense_score
    top_bm25 = top_result.bm25_score

    diagnostics["top_fusion_score"] = top_fusion
    diagnostics["top_dense_score"] = top_dense
    diagnostics["top_bm25_score"] = top_bm25

    # Max score across metrics
    best_score = max(top_fusion, top_dense, top_bm25)

    if best_score < min_score:
        reason = f"Top retrieval score ({best_score:.4f}) below sufficiency threshold ({min_score:.4f})."
        diagnostics["refusal_reason"] = reason
        return False, 0.1, reason, diagnostics

    # Compute heuristic confidence
    confidence = min(1.0, max(0.3, best_score * 2.0))
    diagnostics["gate_passed"] = True

    return True, confidence, None, diagnostics
