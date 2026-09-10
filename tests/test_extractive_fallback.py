"""
Tests for Section-Aware Evidence, Centered Excerpts, Boilerplate Exclusion, and Extractive Fallback Answer Mode.
"""

import pytest
from unittest.mock import MagicMock

from src.rag_pipeline import answer_question
from src.models import RetrievalResult, DocumentChunk, RAGResponse, DocumentPage
from src.extractive_answer import generate_extractive_answer, score_sentence_relevance, is_boilerplate_sentence
from src.chunking import chunk_page
from src.llm import LLMError


@pytest.fixture
def multi_chunk_retrieved_results():
    chunk_flex = DocumentChunk(
        chunk_id="chk_flex_101",
        document_id="doc_workplace",
        document_name="workplace_policy.pdf",
        page_number=3,
        text="3. Flexible Scheduling Options. Employees may request flexible working hours subject to manager approval.",
        section="3. Flexible Scheduling Options",
        chunk_index=0,
    )
    res_flex = RetrievalResult(
        chunk=chunk_flex,
        document_name="workplace_policy.pdf",
        page_number=3,
        section="3. Flexible Scheduling Options",
        dense_score=0.88,
        bm25_score=0.85,
        fusion_score=0.89,
        final_rank=1,
        retrieval_methods=["faiss", "bm25"],
    )

    chunk_absence = DocumentChunk(
        chunk_id="chk_abs_102",
        document_id="doc_workplace",
        document_name="workplace_policy.pdf",
        page_number=4,
        text="4. Unexcused Absences & Tardiness. Unexcused absences or habitual tardiness exceeding 3 occurrences in a quarter are subject to HR review and progressive performance management.",
        section="4. Unexcused Absences & Tardiness",
        chunk_index=1,
    )
    res_absence = RetrievalResult(
        chunk=chunk_absence,
        document_name="workplace_policy.pdf",
        page_number=4,
        section="4. Unexcused Absences & Tardiness",
        dense_score=0.85,
        bm25_score=0.92,
        fusion_score=0.87,
        final_rank=2,
        retrieval_methods=["faiss", "bm25"],
    )

    chunk_expense = DocumentChunk(
        chunk_id="chk_exp_103",
        document_id="doc_expense",
        document_name="expense_policy.pdf",
        page_number=1,
        text="Receipts are required for all travel expenses exceeding $25.",
        section="Expense Rules",
        chunk_index=2,
    )
    res_expense = RetrievalResult(
        chunk=chunk_expense,
        document_name="expense_policy.pdf",
        page_number=1,
        section="Expense Rules",
        dense_score=0.40,
        bm25_score=0.30,
        fusion_score=0.45,
        final_rank=3,
        retrieval_methods=["bm25"],
    )

    chunk_boilerplate = DocumentChunk(
        chunk_id="chk_bp_104",
        document_id="doc_workplace",
        document_name="workplace_policy.pdf",
        page_number=1,
        text="DEMONSTRATION POLICY DISCLOSURE: This document is a synthetic demonstration policy version 1.0.",
        section="Document Control",
        chunk_index=3,
    )
    res_boilerplate = RetrievalResult(
        chunk=chunk_boilerplate,
        document_name="workplace_policy.pdf",
        page_number=1,
        section="Document Control",
        dense_score=0.35,
        bm25_score=0.20,
        fusion_score=0.35,
        final_rank=4,
        retrieval_methods=["bm25"],
    )

    return [res_flex, res_absence, res_expense, res_boilerplate]


@pytest.fixture
def healthcare_retrieved_results():
    chunk_purpose = DocumentChunk(
        chunk_id="chk_health_purpose_001",
        document_id="doc_benefits",
        document_name="benefits_policy.pdf",
        page_number=1,
        text="1. Purpose & Scope. DEMONSTRATION POLICY DISCLOSURE: This document outlines employee health benefits version 1.0.",
        section="1. Purpose & Scope",
        chunk_index=0,
    )
    res_purpose = RetrievalResult(
        chunk=chunk_purpose,
        document_name="benefits_policy.pdf",
        page_number=1,
        section="1. Purpose & Scope",
        dense_score=0.70,
        bm25_score=0.65,
        fusion_score=0.72,
        final_rank=1,
        retrieval_methods=["faiss", "bm25"],
    )

    chunk_health = DocumentChunk(
        chunk_id="chk_health_cov_002",
        document_id="doc_benefits",
        document_name="benefits_policy.pdf",
        page_number=2,
        text="2. Healthcare Insurance Plan Coverage. The company pays 85% of monthly healthcare insurance premiums for full-time employees and 50% for enrolled dependents. Coverage takes effect on the 1st of the month following start date.",
        section="2. Healthcare Insurance Plan Coverage",
        chunk_index=1,
    )
    res_health = RetrievalResult(
        chunk=chunk_health,
        document_name="benefits_policy.pdf",
        page_number=2,
        section="2. Healthcare Insurance Plan Coverage",
        dense_score=0.94,
        bm25_score=0.96,
        fusion_score=0.95,
        final_rank=2,
        retrieval_methods=["faiss", "bm25"],
    )

    chunk_wfh = DocumentChunk(
        chunk_id="chk_wfh_003",
        document_id="doc_benefits",
        document_name="benefits_policy.pdf",
        page_number=5,
        text="5. Remote Work & Telecommuting. Employees approved for WFH receive a one-time remote setup stipend.",
        section="5. Remote Work & Telecommuting",
        chunk_index=2,
    )
    res_wfh = RetrievalResult(
        chunk=chunk_wfh,
        document_name="benefits_policy.pdf",
        page_number=5,
        section="5. Remote Work & Telecommuting",
        dense_score=0.45,
        bm25_score=0.40,
        fusion_score=0.48,
        final_rank=3,
        retrieval_methods=["bm25"],
    )

    return [res_purpose, res_health, res_wfh]


def test_healthcare_premium_query_returns_percentages(healthcare_retrieved_results):
    """Test 1, 4 & 5: Healthcare premium query returns 85% and 50%, excludes Purpose & Scope and WFH."""
    query = "How much of the healthcare insurance premium does the company pay for employees and their dependents?"
    res = generate_extractive_answer(query, healthcare_retrieved_results)

    assert res is not None
    answer_text, sources = res

    # 1. Answer text contains 85% and 50%
    assert "85%" in answer_text
    assert "50%" in answer_text
    assert "[S1]" in answer_text

    # 2. Excludes WFH and Purpose & Scope boilerplate
    assert "Purpose & Scope" not in answer_text
    assert "DEMONSTRATION POLICY" not in answer_text
    assert "WFH" not in answer_text
    assert "remote setup stipend" not in answer_text


def test_healthcare_section_metadata_and_centered_excerpt(healthcare_retrieved_results):
    """Test 2 & 3: Evidence section is Healthcare Insurance Plan Coverage and excerpt contains supporting premium sentence."""
    query = "How much of the healthcare insurance premium does the company pay for employees and their dependents?"
    res = generate_extractive_answer(query, healthcare_retrieved_results)

    assert res is not None
    _, sources = res

    # Check cited source metadata and centered excerpt
    assert len(sources) == 1
    src = sources[0]
    assert src.source_id == "[S1]"
    assert src.section == "2. Healthcare Insurance Plan Coverage"
    assert "85%" in src.excerpt
    assert "50%" in src.excerpt


def test_unused_citations_pruned_for_factual_query(healthcare_retrieved_results):
    """Test 6: Simple factual query produces single cited source [S1] and prunes unused candidate citations."""
    query = "How much of the healthcare insurance premium does the company pay for employees and their dependents?"
    res = generate_extractive_answer(query, healthcare_retrieved_results)

    assert res is not None
    answer_text, sources = res

    assert "[S1]" in answer_text
    assert "[S2]" not in answer_text
    assert len(sources) == 1


def test_section_aware_chunk_boundary_splitting():
    """Test 1: Heading-aware chunking enforces section splits so Section 2 text does not inherit Section 1 metadata."""
    page_text = (
        "## 1. Purpose & Scope\n"
        "This policy outlines employee benefit plans and eligibility requirements.\n\n"
        "## 2. Healthcare Insurance Plan Coverage\n"
        "The company pays 85% of monthly healthcare insurance premiums for full-time employees and 50% for enrolled dependents."
    )
    doc_page = DocumentPage(
        document_id="doc_ben",
        document_name="benefits_policy.pdf",
        file_type="pdf",
        page_number=2,
        text=page_text,
        source_path="benefits_policy.pdf",
    )

    chunks = chunk_page(doc_page, chunk_size=200, chunk_overlap=30)
    health_chunks = [c for c in chunks if "85%" in c.text]

    assert len(health_chunks) > 0
    assert health_chunks[0].section == "2. Healthcare Insurance Plan Coverage"
    assert "1. Purpose & Scope" not in health_chunks[0].text


def test_unexcused_absences_query_selects_absence_evidence(multi_chunk_retrieved_results):
    """Test 7: Query about unexcused absences selects absence chunk and maps [S1] to it."""
    query = "What are the rules for unexcused absences?"
    res = generate_extractive_answer(query, multi_chunk_retrieved_results)

    assert res is not None
    answer_text, sources = res

    assert "Unexcused absences" in answer_text
    assert "[S1]" in answer_text
    assert len(sources) == 1
    assert sources[0].source_id == "[S1]"
    assert sources[0].chunk_id == "chk_abs_102"
    assert sources[0].section == "4. Unexcused Absences & Tardiness"


def test_unsupported_question_refusal_unchanged():
    """Test 8: Unsupported question refusal remains unchanged."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-3.8-flash"

    query = "What is the policy for alien spacecraft landing?"
    response: RAGResponse = answer_question(
        question=query,
        retriever=mock_retriever,
        llm_service=mock_llm,
    )

    assert response.refused is True
    assert response.answer == "I could not find sufficient information in the provided documents to answer this question."


def test_redundant_heading_fragments_removed():
    """Test cleanup 1: Standalone heading lines without verbs are filtered out from answer text."""
    from src.extractive_answer import is_standalone_heading
    assert is_standalone_heading("3. Annual Wellness Stipend.") is True
    assert is_standalone_heading("Flexible Scheduling Options.") is True
    assert is_standalone_heading("The company pays 85% of premiums.") is False


def test_evidence_excerpt_clipped_at_next_section():
    """Test cleanup 2: Evidence excerpt is clipped before bleeding into the next section heading."""
    from src.extractive_answer import build_centered_excerpt
    chunk_text = (
        "3. Annual Wellness Stipend\n"
        "Full-time employees receive up to $500 per year for eligible wellness expenses.\n\n"
        "4. 401(k) Retirement Savings Match\n"
        "The company matches up to 4% of employee salary contributions."
    )
    sent = "Full-time employees receive up to $500 per year for eligible wellness expenses."
    excerpt = build_centered_excerpt(chunk_text, sent)

    assert "Annual Wellness Stipend" in excerpt or "$500" in excerpt
    assert "401(k) Retirement Savings Match" not in excerpt
    assert "matches up to 4%" not in excerpt


def test_compound_expense_query_multi_fact_answering():
    """Test compound expense query returns receipt rule, credit card rejection, and 30 calendar days rule with aligned section metadata."""
    chunk_receipts = DocumentChunk(
        chunk_id="chk_rec_201",
        document_id="doc_expense_pol",
        document_name="expense_policy.pdf",
        page_number=2,
        text=(
            "2. Receipt & Documentation Requirements\n"
            "Itemized receipts are strictly required for all reimbursable expenses exceeding $25. "
            "Summary credit card slips without itemized line items are not acceptable. "
            "Claims must be submitted via the expense portal within 30 calendar days."
        ),
        section="2. Receipt & Documentation Requirements",
        chunk_index=0,
    )
    res_receipts = RetrievalResult(
        chunk=chunk_receipts,
        document_name="expense_policy.pdf",
        page_number=2,
        section="1. Purpose & Scope",
        dense_score=0.92,
        bm25_score=0.95,
        fusion_score=0.94,
        final_rank=1,
        retrieval_methods=["faiss", "bm25"],
    )

    query = "What receipts are required for reimbursable expenses over $25, and how soon must I submit the claim?"
    res = generate_extractive_answer(query, [res_receipts])

    assert res is not None
    answer_text, sources = res

    # 1. Multi-part answering: contains all 3 key facts
    assert "Itemized receipts" in answer_text
    assert "not acceptable" in answer_text or "credit card slips" in answer_text
    assert "30 calendar days" in answer_text

    # 2. Grouped citation: single [S1] at the end
    assert answer_text.endswith("[S1]") or "[S1]" in answer_text
    assert answer_text.count("[S1]") == 1

    # 3. Section metadata alignment between source row and RetrievalResult
    assert len(sources) == 1
    src = sources[0]
    assert src.section == "2. Receipt & Documentation Requirements"
    assert res_receipts.section == "2. Receipt & Documentation Requirements"


