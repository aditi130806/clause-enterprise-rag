"""
Focused Unit Tests for Prompt 5: Advanced RAG + Evidence Guard.
Tests lightweight reranking, deterministic query rewriting, bounded corrective retrieval,
Evidence Guard diagnostics, conflict detection, and exact refusal behavior.
"""

import pytest
from unittest.mock import MagicMock

from src.models import DocumentChunk, RetrievalResult, SourceReference, RAGResponse
from src.query_rewriter import normalize_query, expand_query_for_corrective_retrieval, extract_policy_identifiers
from src.reranker import rerank_retrieval_results, calculate_query_coverage
from src.evidence_guard import EvidenceGuard, detect_evidence_conflicts
from src.rag_pipeline import answer_question, INSUFFICIENT_INFORMATION_REFUSAL


@pytest.fixture
def sample_chunk_a():
    return DocumentChunk(
        chunk_id="chk_a",
        document_id="doc_rw",
        document_name="Remote Work Policy.pdf",
        page_number=2,
        text="Employees on probation may request remote work approval under HR-RW-017.",
        section="Remote Work Eligibility",
        chunk_index=0,
    )


@pytest.fixture
def sample_chunk_b():
    return DocumentChunk(
        chunk_id="chk_b",
        document_id="doc_sec",
        document_name="Information Security Policy.pdf",
        page_number=5,
        text="Multi-factor authentication (MFA) is mandatory for all remote access.",
        section="Authentication Standards",
        chunk_index=1,
    )


@pytest.fixture
def sample_retrieval_results(sample_chunk_a, sample_chunk_b):
    res_a = RetrievalResult(
        chunk=sample_chunk_a,
        document_name="Remote Work Policy.pdf",
        page_number=2,
        section="Remote Work Eligibility",
        fusion_score=0.25,
        final_rank=1,
        retrieval_methods=["dense", "bm25"],
    )
    res_b = RetrievalResult(
        chunk=sample_chunk_b,
        document_name="Information Security Policy.pdf",
        page_number=5,
        section="Authentication Standards",
        fusion_score=0.15,
        final_rank=2,
        retrieval_methods=["dense"],
    )
    return [res_a, res_b]


# ============================================================================
# 1. LIGHTWEIGHT RERANKING TESTS
# ============================================================================

def test_query_coverage_calculation():
    score = calculate_query_coverage("probation remote work", "Employees on probation may request remote work")
    assert score > 0.5


def test_rerank_retrieval_results_preserves_metadata(sample_retrieval_results):
    query = "HR-RW-017 probation WFH"
    reranked = rerank_retrieval_results(query, sample_retrieval_results, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].final_rank == 1
    assert reranked[1].final_rank == 2
    # Exact identifier HR-RW-017 should boost sample_chunk_a to top rank
    assert reranked[0].document_name == "Remote Work Policy.pdf"
    assert "reranked" in reranked[0].retrieval_methods


def test_rerank_deterministic_ordering(sample_retrieval_results):
    query = "MFA authentication security"
    reranked = rerank_retrieval_results(query, sample_retrieval_results, top_k=2)

    # MFA chunk should get highest boost for security query
    assert reranked[0].document_name == "Information Security Policy.pdf"
    assert reranked[0].final_rank == 1


# ============================================================================
# 2. QUERY REWRITING TESTS
# ============================================================================

def test_normalize_query_abbreviations():
    raw_query = "Can an employee WFH during probation?"
    normalized = normalize_query(raw_query)

    assert "work from home remote work" in normalized
    assert "probationary employee eligibility" in normalized
    assert "Can an employee" not in normalized


def test_normalize_query_preserves_exact_policy_identifiers():
    raw = "What does HR-RW-017 say about MFA?"
    normalized = normalize_query(raw)

    assert "HR-RW-017" in normalized
    assert "multi-factor authentication MFA" in normalized


def test_expand_query_for_corrective_retrieval():
    expanded = expand_query_for_corrective_retrieval("WFH probation policy")
    assert "work" in expanded
    assert "probation" in expanded
    assert "remote" in expanded


# ============================================================================
# 3. EVIDENCE GUARD & CONFLICT DETECTION TESTS
# ============================================================================

def test_evidence_guard_high_confidence(sample_retrieval_results):
    # Artificially set high score for testing
    sample_retrieval_results[0].fusion_score = 0.45

    guard = EvidenceGuard()
    diag = guard.evaluate_retrieval_evidence("probation", sample_retrieval_results)

    assert diag.confidence_label == "High"
    assert diag.evidence_status == "sufficient"
    assert not diag.refusal_recommended
    assert diag.source_diversity == 2


def test_evidence_guard_insufficient_confidence():
    guard = EvidenceGuard(min_sufficiency_score=0.10)
    diag = guard.evaluate_retrieval_evidence("unsupported query", [])

    assert diag.confidence_label == "Insufficient"
    assert diag.evidence_status == "insufficient"
    assert diag.refusal_recommended
    assert "No document chunks retrieved" in diag.reasons[0]


def test_detect_evidence_conflicts():
    chunk1 = DocumentChunk("c1", "d1", "Policy A", 1, "Remote work is allowed for all staff.", "Sec 1", 0)
    chunk2 = DocumentChunk("c2", "d2", "Policy B", 2, "Remote work is prohibited during training.", "Sec 2", 1)

    r1 = RetrievalResult(chunk1, "Policy A", 1, "Sec 1", fusion_score=0.3)
    r2 = RetrievalResult(chunk2, "Policy B", 2, "Sec 2", fusion_score=0.25)

    has_conflict, reasons = detect_evidence_conflicts([r1, r2])
    assert has_conflict
    assert len(reasons) > 0
    assert "allowed" in reasons[0] and "prohibited" in reasons[0]


# ============================================================================
# 4. RAG PIPELINE INTEGRATION & STRICT REFUSAL TESTS
# ============================================================================

def test_answer_question_unsupported_refusal(sample_retrieval_results):
    mock_retriever = MagicMock()
    # Return empty results to trigger refusal
    mock_retriever.retrieve.return_value = []

    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-3.8-flash"

    response: RAGResponse = answer_question(
        question="Does the company provide free sports cars?",
        retriever=mock_retriever,
        llm_service=mock_llm,
    )

    assert response.refused
    assert response.answer == INSUFFICIENT_INFORMATION_REFUSAL
    assert response.confidence_label == "Insufficient"
    assert response.evidence_status == "insufficient"
    assert len(response.sources) == 0
    # Gemini should NOT be called when evidence is insufficient
    mock_llm.generate.assert_not_called()


def test_corrective_retrieval_executes_at_most_once(sample_retrieval_results):
    mock_retriever = MagicMock()
    # Weak chunk text ensures coverage is 0.0 and fusion score stays < 0.08
    unrelated_chunk = DocumentChunk("chk_u", "doc_u", "Doc.pdf", 1, "Unrelated content about unrelated topic.", "Sec", 0)
    weak_res = [RetrievalResult(unrelated_chunk, "Doc.pdf", 1, "Sec", fusion_score=0.01)]
    strong_res = [RetrievalResult(sample_retrieval_results[0].chunk, "Doc.pdf", 1, "Sec", fusion_score=0.40)]

    mock_retriever.retrieve.side_effect = [weak_res, strong_res]

    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-3.8-flash"
    mock_llm.generate.return_value = "Employees on probation can work from home [S1]."

    response: RAGResponse = answer_question(
        question="Can I WFH on probation?",
        retriever=mock_retriever,
        llm_service=mock_llm,
    )

    # Exactly 2 retrieve calls: 1 original + 1 bounded corrective
    assert mock_retriever.retrieve.call_count == 2
    assert response.retrieval_corrected
    assert response.confidence_label in ("Moderate", "High")
