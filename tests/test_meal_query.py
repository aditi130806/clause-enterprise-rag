"""
Unit tests for CLAUSE Meal Allowance Query, Evidence Panel Safety, and Citation Alignment.
"""

import pytest
import streamlit as st
from unittest.mock import MagicMock

from src.models import (
    DocumentChunk,
    RetrievalResult,
    SourceReference,
    RAGResponse,
)
from src.ui.components import render_evidence_panel
from src.extractive_answer import generate_extractive_answer
from src.reranker import rerank_retrieval_results
from src.query_rewriter import normalize_query, expand_query_for_corrective_retrieval


def test_evidence_panel_uses_valid_attributes_without_crashing():
    """Test Evidence panel uses SourceReference and RetrievalResult valid fields without AttributeError."""
    src = SourceReference(
        source_id="[S1]",
        document_id="doc_exp",
        document_name="Business Travel & Expense Reimbursement Policy.docx",
        page_number=None,
        section="3. Meals & Daily Per Diem",
        chunk_id="chk_meal_01",
        excerpt="The standard per diem limit for business meals is $75 per day per employee.",
        retrieval_rank=1,
        retrieval_score=0.92,
    )
    chunk = DocumentChunk(
        chunk_id="chk_meal_01",
        document_id="doc_exp",
        document_name="Business Travel & Expense Reimbursement Policy.docx",
        page_number=None,
        text="Section 3. Meals & Daily Per Diem\nThe standard per diem limit for business meals is $75 per day per employee.",
        section="3. Meals & Daily Per Diem",
        chunk_index=3,
    )
    res = RetrievalResult(
        chunk=chunk,
        document_name="Business Travel & Expense Reimbursement Policy.docx",
        page_number=None,
        section="3. Meals & Daily Per Diem",
        fusion_score=0.92,
        final_rank=1,
        retrieval_methods=["bm25", "dense"],
    )
    response = RAGResponse(
        question="What is the daily meal allowance during business travel?",
        answer="The standard per diem limit for business meals is $75 per day per employee. [S1]",
        sources=[src],
        retrieved_results=[res],
        grounded=True,
        refused=False,
    )

    markdown_calls = []
    st.markdown = lambda body, unsafe_allow_html=False: markdown_calls.append(body)

    # Must run cleanly without raising AttributeError: 'RetrievalResult' object has no attribute 'text'
    render_evidence_panel(response)
    assert len(markdown_calls) >= 2
    assert "Business Travel &amp; Expense Reimbursement Policy" in markdown_calls[-1]
    assert "3. Meals &amp; Daily Per Diem" in markdown_calls[-1]
    assert "$75 per day" in markdown_calls[-1]


def test_evidence_panel_renderer_handles_unexpected_error_gracefully(monkeypatch):
    """Test evidence renderer logs error and shows clean neutral UI state on unexpected error."""
    response = RAGResponse(
        question="Test query",
        answer="Test answer",
        sources=[MagicMock(side_effect=Exception("Corrupt object"))],
        retrieved_results=[],
    )

    markdown_calls = []
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: markdown_calls.append(body))

    render_evidence_panel(response)

    assert len(markdown_calls) >= 2
    assert "Evidence details unavailable." in markdown_calls[-1]
    assert "Traceback" not in markdown_calls[-1]


def test_meal_query_expansion():
    """Test query expansion for meal allowance / per diem equivalence."""
    expanded = expand_query_for_corrective_retrieval("What is the daily meal allowance during business travel?")
    assert "per diem" in expanded.lower()
    assert "meal" in expanded.lower()


def test_meal_extractive_answer_and_topic_exclusion():
    """Test meal query extractive fallback selects $75 per day sentence and excludes Attendance / Purpose & Scope."""
    c_scope = DocumentChunk(
        chunk_id="c_scope",
        document_id="doc_exp",
        document_name="Expense Policy",
        page_number=None,
        text="1. Purpose & Scope\nThis policy defines the rules for employee business travel and expense reimbursement across all departments.",
        section="1. Purpose & Scope",
        chunk_index=1,
    )
    c_attendance = DocumentChunk(
        chunk_id="c_att",
        document_id="doc_att",
        document_name="Attendance Policy",
        page_number=None,
        text="3. Flexible Scheduling Options\nFull-time employees must work 8 daily hours and log daily attendance.",
        section="3. Flexible Scheduling Options",
        chunk_index=3,
    )
    c_meal = DocumentChunk(
        chunk_id="c_meal",
        document_id="doc_exp",
        document_name="Expense Policy",
        page_number=None,
        text="3. Meals & Daily Per Diem\nThe standard per diem limit for business meals is $75 per day per employee.",
        section="3. Meals & Daily Per Diem",
        chunk_index=4,
    )

    res_scope = RetrievalResult(chunk=c_scope, document_name="Expense Policy", page_number=None, section="1. Purpose & Scope", fusion_score=0.5)
    res_att = RetrievalResult(chunk=c_attendance, document_name="Attendance Policy", page_number=None, section="3. Flexible Scheduling Options", fusion_score=0.6)
    res_meal = RetrievalResult(chunk=c_meal, document_name="Expense Policy", page_number=None, section="3. Meals & Daily Per Diem", fusion_score=0.85)

    results = [res_meal, res_att, res_scope]
    query = "What is the daily meal allowance during business travel?"

    ans, sources = generate_extractive_answer(query, results)

    assert ans is not None
    assert "$75 per day" in ans
    assert "Flexible Scheduling" not in ans
    assert "8 daily hours" not in ans
    assert "Purpose & Scope" not in ans
    assert len(sources) == 1
    assert sources[0].section == "3. Meals & Daily Per Diem"
    assert sources[0].source_id == "[S1]"
