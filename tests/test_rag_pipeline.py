"""
Unit tests for src/rag_pipeline.py (QA Orchestration & Refusal handling).
"""

from unittest.mock import MagicMock
from src.models import DocumentChunk, RetrievalResult
from src.retriever import HybridRetriever
from src.rag_pipeline import answer_question
from src.evidence_gate import INSUFFICIENT_INFORMATION_REFUSAL


def test_answer_question_empty_query():
    mock_retriever = MagicMock()
    res = answer_question("", retriever=mock_retriever)

    assert res.refused is True
    assert res.grounded is True
    assert res.answer == INSUFFICIENT_INFORMATION_REFUSAL
    assert len(res.sources) == 0


def test_answer_question_insufficient_evidence_refusal():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    res = answer_question("Does the company provide employees with free company cars?", retriever=mock_retriever)

    assert res.refused is True
    assert res.answer == INSUFFICIENT_INFORMATION_REFUSAL
    assert res.metadata["refusal_stage"] == "evidence_sufficiency_gate"
    assert len(res.sources) == 0


def test_answer_question_mocked_llm_success():
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc_remote",
        document_name="remote_policy.pdf",
        page_number=1,
        text="Remote work guidelines HR-RW-017.",
        section="Overview",
        chunk_index=0,
    )
    result = RetrievalResult(c1, "remote_policy.pdf", 1, "Overview", fusion_score=0.032, final_rank=1)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [result]

    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-3.6-flash"
    mock_llm.generate.return_value = "Employees may work remotely per policy HR-RW-017 [S1]."

    res = answer_question(
        "What does HR-RW-017 say?",
        retriever=mock_retriever,
        llm_service=mock_llm,
    )

    assert res.refused is False
    assert res.grounded is True
    assert len(res.sources) == 1
    assert res.sources[0].source_id == "S1"
    assert res.sources[0].document_name == "remote_policy.pdf"
    assert "HR-RW-017" in res.answer
