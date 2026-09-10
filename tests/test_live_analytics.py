"""
Unit tests for CLAUSE Live Usage Analytics, Persistence, and Benchmark Refusal Regression.
"""

import json
import pytest
from pathlib import Path

from src.models import RAGResponse, SourceReference, DocumentChunk, RetrievalResult
from src.live_analytics import (
    log_live_query_event,
    load_live_query_events,
    compute_live_analytics_summary,
    clear_live_query_events,
    LIVE_USAGE_FILE,
)
from src.evidence_guard import EvidenceGuard


@pytest.fixture(autouse=True)
def clean_live_analytics_file():
    """Ensure clean live analytics state before and after each test."""
    clear_live_query_events()
    yield
    clear_live_query_events()


def test_ask_request_appends_live_analytics_event():
    """Test that logging a RAGResponse appends a structured event record."""
    src = SourceReference(
        source_id="[S1]",
        document_id="doc1",
        document_name="Remote Work Policy.docx",
        page_number=1,
        section="Probationary Eligibility",
        chunk_id="chk1",
        excerpt="Employees on probation may work remotely with manager approval.",
        retrieval_rank=1,
        retrieval_score=0.9,
    )
    resp = RAGResponse(
        question="Can an employee work from home while on probation?",
        answer="Yes, with manager approval. [S1]",
        sources=[src],
        retrieved_results=[],
        grounded=True,
        refused=False,
        confidence_label="High",
        metadata={"total_ms": 150.0},
    )

    log_live_query_event(resp)

    events = load_live_query_events()
    assert len(events) == 1
    assert events[0]["question"] == "Can an employee work from home while on probation?"
    assert events[0]["mode"] == "Gemini"
    assert events[0]["confidence"] == "High"
    assert events[0]["sources"] == 1
    assert events[0]["citation_valid"] is True
    assert events[0]["refused"] is False


def test_total_queries_increments_and_metrics_update():
    """Test query counts and metric aggregations increment dynamically."""
    # 1. Answered Gemini query
    r1 = RAGResponse(
        question="Query 1",
        answer="Ans 1 [S1]",
        sources=[SourceReference(source_id="[S1]", document_id="d1", document_name="doc.docx", page_number=1, section="Sec 1", chunk_id="chk1", excerpt="valid excerpt", retrieval_rank=1, retrieval_score=0.9)],
        grounded=True,
        refused=False,
        confidence_label="High",
        metadata={"total_ms": 100.0},
    )
    # 2. Refusal query
    r2 = RAGResponse(
        question="Query 2",
        answer="I cannot answer this based on the available documents.",
        sources=[],
        grounded=False,
        refused=True,
        confidence_label="Insufficient",
        metadata={"total_ms": 50.0},
    )
    # 3. Fallback query
    r3 = RAGResponse(
        question="Query 3",
        answer="Extractive fallback answer. [S1]",
        sources=[SourceReference(source_id="[S1]", document_id="d1", document_name="doc.docx", page_number=1, section="Sec 1", chunk_id="chk1", excerpt="valid excerpt", retrieval_rank=1, retrieval_score=0.8)],
        grounded=True,
        refused=False,
        confidence_label="Moderate",
        metadata={"fallback": True, "total_ms": 200.0},
    )

    log_live_query_event(r1)
    summary1 = compute_live_analytics_summary()
    assert summary1["total_queries"] == 1
    assert summary1["answered_queries"] == 1
    assert summary1["refused_queries"] == 0

    log_live_query_event(r2)
    log_live_query_event(r3)

    summary3 = compute_live_analytics_summary()
    assert summary3["total_queries"] == 3
    assert summary3["answered_queries"] == 2
    assert summary3["refused_queries"] == 1
    assert summary3["fallback_responses"] == 1
    assert summary3["gemini_responses"] == 1
    assert summary3["citation_valid_responses"] == 2
    assert summary3["avg_latency_ms"] == 116.7


def test_clearing_live_analytics_works():
    """Test clearing live analytics truncates live_usage.jsonl without removing files."""
    r1 = RAGResponse(question="Q1", answer="A1", sources=[], refused=True)
    log_live_query_event(r1)
    assert len(load_live_query_events()) == 1

    clear_live_query_events()
    assert len(load_live_query_events()) == 0


def test_benchmark_dataset_remains_fixed_at_24():
    """Test that benchmark dataset questions count remains fixed at 24 and is not modified by live queries."""
    from src.evaluation import load_benchmark_questions
    dataset = load_benchmark_questions()
    assert len(dataset) == 24
    
    # Adding live query must not touch benchmark dataset
    r1 = RAGResponse(question="Arbitrary user question on Ask page", answer="Ans", sources=[])
    log_live_query_event(r1)

    dataset_after = load_benchmark_questions()
    assert len(dataset_after) == 24


def test_refusal_regression_fixed_100_percent():
    """Test EvidenceGuard key subject term coverage refuses completely unsupported queries (100% refusal rate)."""
    guard = EvidenceGuard()

    # Unsupported Q21: Tuition reimbursement for master's degree
    unsupported_query_21 = "What is the company policy for tuition reimbursement for university master's degrees?"
    
    chunk_expense = DocumentChunk(
        chunk_id="chk_exp",
        document_id="doc_exp",
        document_name="Expense Reimbursement Policy.docx",
        page_number=1,
        text="Section 2. Reimbursable Expenses\nItemized receipts are strictly required for all business expenses exceeding $25.",
        section="2. Reimbursable Expenses",
        chunk_index=2,
    )
    res_expense = RetrievalResult(
        chunk=chunk_expense,
        document_name="Expense Reimbursement Policy.docx",
        page_number=1,
        section="2. Reimbursable Expenses",
        fusion_score=0.20,
    )

    result_21 = guard.evaluate_retrieval_evidence(unsupported_query_21, [res_expense])
    assert result_21.refusal_recommended is True
    assert result_21.confidence_label == "Insufficient"

    # Unsupported Q23: Sabbatical leaves exceeding 2 years
    unsupported_query_23 = "What are the company guidelines for sabbatical leaves exceeding 2 years?"
    chunk_pto = DocumentChunk(
        chunk_id="chk_pto",
        document_id="doc_pto",
        document_name="Annual Leave & PTO Policy.docx",
        page_number=1,
        text="Section 1. Annual Leave\nFull-time employees receive 15 PTO days annually.",
        section="1. Annual Leave",
        chunk_index=1,
    )
    res_pto = RetrievalResult(
        chunk=chunk_pto,
        document_name="Annual Leave & PTO Policy.docx",
        page_number=1,
        section="1. Annual Leave",
        fusion_score=0.20,
    )

    result_23 = guard.evaluate_retrieval_evidence(unsupported_query_23, [res_pto])
    assert result_23.refusal_recommended is True
    assert result_23.confidence_label == "Insufficient"
