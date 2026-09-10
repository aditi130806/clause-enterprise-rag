"""
Tests for Ask-page UX polish: Insufficient-evidence panel consistency,
immediate loading state & session state lifecycle, and sidebar logo balance.
"""

import pytest
import streamlit as st
from src.models import RAGResponse, SourceReference, RetrievalResult, DocumentChunk
from src.ui.components import render_evidence_panel
from src.ui.shell import load_logo_svg, render_sidebar


def test_insufficient_evidence_panel_hides_unrelated_chunks(monkeypatch):
    """Test that refused/insufficient response hides unrelated retrieval candidates."""
    chunk = DocumentChunk(
        chunk_id="chk_1",
        document_id="doc_1",
        document_name="Expense_Policy.docx",
        page_number=1,
        text="Unrelated expense info",
        section="1. Expenses",
        chunk_index=0,
    )
    res = RetrievalResult(
        chunk=chunk,
        dense_score=0.8,
        bm25_score=5.0,
        fusion_score=0.9,
        document_name="Expense_Policy.docx",
        page_number=1,
        section="1. Expenses",
        final_rank=1,
    )

    insufficient_resp = RAGResponse(
        question="What is the Martian vacation policy?",
        answer="I could not find sufficient information in the provided documents to answer this question.",
        sources=[],  # Zero validated cited sources
        retrieved_results=[res],
        grounded=False,
        refused=True,
        confidence=0.0,
        confidence_label="Insufficient",
        evidence_status="insufficient",
        conflict_detected=False,
        retrieval_corrected=False,
        evidence_reasons=["No matching evidence found"],
        generation_model="gemini-3.8-flash",
        retrieval_mode="hybrid",
    )

    rendered_markdown = []

    def mock_markdown(body, unsafe_allow_html=False):
        rendered_markdown.append(body)

    monkeypatch.setattr(st, "markdown", mock_markdown)

    render_evidence_panel(insufficient_resp)

    combined_output = "".join(rendered_markdown)
    assert "No supporting evidence was found for this question." in combined_output
    assert "Expense_Policy.docx" not in combined_output
    assert "Retrieval rank" not in combined_output


def test_supported_evidence_panel_shows_sources(monkeypatch):
    """Test that supported response with cited sources renders evidence normally."""
    src = SourceReference(
        source_id="S1",
        document_id="doc_2",
        document_name="Employee_Benefits_Policy.pdf",
        page_number=2,
        section="2. Healthcare Coverage",
        chunk_id="chk_2",
        excerpt="The company pays 85% of monthly healthcare insurance premiums.",
        retrieval_rank=1,
        retrieval_score=0.95,
    )
    supported_resp = RAGResponse(
        question="How much insurance premium is covered?",
        answer="The company pays 85% of monthly healthcare insurance premiums. [S1]",
        sources=[src],
        retrieved_results=[],
        grounded=True,
        refused=False,
        confidence=0.95,
        confidence_label="High",
        evidence_status="grounded",
        conflict_detected=False,
        retrieval_corrected=False,
        evidence_reasons=[],
        generation_model="gemini-3.8-flash",
        retrieval_mode="hybrid",
    )

    rendered_markdown = []

    def mock_markdown(body, unsafe_allow_html=False):
        rendered_markdown.append(body)

    monkeypatch.setattr(st, "markdown", mock_markdown)

    render_evidence_panel(supported_resp)

    combined_output = "".join(rendered_markdown)
    assert "Employee_Benefits_Policy.pdf" in combined_output
    assert "85%" in combined_output
    assert "2. Healthcare Coverage" in combined_output


def test_logo_rendering():
    """Test that load_logo_svg returns valid SVG with proper brand dimensions and tight viewBox."""
    svg = load_logo_svg()
    assert "<svg" in svg
    assert "Clause" in svg
    assert 'viewBox="8 8 135 38"' in svg or 'width="135"' in svg

