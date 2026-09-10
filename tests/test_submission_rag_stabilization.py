"""
Regression test suite for Clause RAG final submission stabilization.
Verifies correct answer generation for supported compound queries and proper refusal for unsupported queries.
"""

import pytest
from src.rag_pipeline import load_retrieval_engine, answer_question_with_timeout
from src.extractive_answer import generate_extractive_answer
from src.reranker import rerank_retrieval_results


@pytest.fixture(scope="module")
def retriever():
    return load_retrieval_engine()


def test_receipts_and_submission_deadline_query(retriever):
    """
    Verify compound query regarding receipt requirements over $25 and submission timeframe.
    Must NOT be refused.
    Must return answers grounded in Expense Reimbursement Policy containing $25 and 30 calendar days.
    """
    query = "What receipts are required for expenses over $25, and when must I submit the claim?"
    
    # 1. Test offline reranker & extractive engine directly
    raw_res = retriever.retrieve(query, top_k=5)
    reranked = rerank_retrieval_results(query, raw_res, top_k=5)
    
    assert len(reranked) > 0
    fb_res = generate_extractive_answer(query, reranked)
    assert fb_res is not None
    
    ext_answer, sources = fb_res
    assert "$25" in ext_answer or "itemized" in ext_answer.lower()
    assert "30" in ext_answer or "days" in ext_answer.lower()
    assert len(sources) > 0
    assert any("Expense" in s.document_name for s in sources)

    # 2. Test full answer_question_with_timeout pipeline
    response = answer_question_with_timeout(question=query, retriever=retriever)
    assert response.refused is False
    assert response.confidence_label in ("High", "Moderate")
    assert len(response.sources) > 0
    assert any("Expense" in s.document_name for s in response.sources)


def test_unsupported_stock_options_refusal(retriever):
    """
    Verify that unsupported equity/stock options query is strictly refused.
    """
    query = "Does the company offer employee stock options or equity grants?"
    response = answer_question_with_timeout(question=query, retriever=retriever)
    
    assert response.refused is True
    assert "insufficient" in response.evidence_status.lower() or response.confidence_label == "Insufficient"


def test_daily_meal_allowance_query(retriever):
    """
    Verify daily meal per diem query returns $75 per day.
    """
    query = "What is the daily meal allowance during business travel?"
    raw_res = retriever.retrieve(query, top_k=5)
    reranked = rerank_retrieval_results(query, raw_res, top_k=5)
    
    fb_res = generate_extractive_answer(query, reranked)
    assert fb_res is not None
    ext_answer, sources = fb_res
    assert "$75" in ext_answer or "75" in ext_answer
    assert any("Expense" in s.document_name for s in sources)


def test_mfa_requirements_query(retriever):
    """
    Verify MFA requirements query returns substantive policy sentences rather than section title alone.
    """
    query = "What are the MFA requirements?"
    raw_res = retriever.retrieve(query, top_k=5)
    reranked = rerank_retrieval_results(query, raw_res, top_k=5)
    
    fb_res = generate_extractive_answer(query, reranked)
    assert fb_res is not None
    ext_answer, sources = fb_res
    assert len(ext_answer) > 30
    assert "Multi-Factor Authentication" not in ext_answer or len(ext_answer.split()) > 8
