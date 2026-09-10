"""
Unit tests for MFA Extractive Fallback & Suggested Queries Curation.
"""

import pytest
from src.document_loader import load_docx
from src.chunking import chunk_documents
from src.extractive_answer import generate_extractive_answer
from src.models import RetrievalResult


def test_mfa_fallback_returns_actual_requirement_not_heading_only():
    pages = load_docx("data/documents/Information Security Policy.docx")
    chunks = chunk_documents(pages)

    retrieved_results = [
        RetrievalResult(
            chunk=c,
            document_name=c.document_name,
            page_number=c.page_number,
            section=c.section,
            dense_score=0.9,
            bm25_score=10.0,
            final_rank=i + 1,
            fusion_score=0.9 - (i * 0.05),
        )
        for i, c in enumerate(chunks)
    ]

    result = generate_extractive_answer("What are the MFA requirements?", retrieved_results)
    assert result is not None, "Extractive answer should not be None"

    answer_text, sources = result

    # 1. Heading-only check: answer must not be just the section heading
    assert answer_text.strip() != "Multi-Factor Authentication (MFA) Requirements. [S1]"
    assert "mandatory" in answer_text.lower() or "tokens" in answer_text.lower()

    # 2. Disclosure text check: disclosure text must be excluded
    assert "DEMONSTRATION POLICY DISCLOSURE" not in answer_text
    for s in sources:
        assert "DEMONSTRATION POLICY DISCLOSURE" not in s.excerpt

    # 3. Section & Evidence mapping check
    assert len(sources) > 0
    mfa_source = sources[0]
    assert mfa_source.document_name == "Information Security Policy.docx"
    assert "Multi-Factor Authentication" in mfa_source.section
    assert "Purpose & Scope" not in mfa_source.section


def test_suggested_queries_curation():
    from src.ui.pages.ask import render_ask_page
    import inspect

    source = inspect.getsource(render_ask_page)

    expected_queries = [
        "What percentage of healthcare insurance premiums does the company cover for employees and dependents?",
        "What receipts are required for expenses over $25, and when must I submit the claim?",
        "Can employees work from home during probation, and what conditions apply?",
    ]

    for q in expected_queries:
        assert q in source, f"Expected query chip missing from ask.py: {q}"
