"""
Tests for Ask Request Timeout Safety and Timings.
"""

import time
import pytest
from unittest.mock import MagicMock

from src.rag_pipeline import answer_question_with_timeout, answer_question
from src.models import RAGResponse


def test_ask_timeout_triggers_refusal(monkeypatch):
    """Test that answer_question_with_timeout safely abandons slow requests after timeout."""
    # Mock answer_question to simulate a hanging/slow function (sleeping 2.0s)
    def mock_slow_answer(*args, **kwargs):
        time.sleep(2.0)
        return RAGResponse(
            question="slow query",
            answer="Slow answer",
            sources=[],
            retrieved_results=[],
            grounded=True,
            refused=False,
            confidence=1.0,
            confidence_label="High",
            evidence_status="sufficient",
            conflict_detected=False,
            retrieval_corrected=False,
            evidence_reasons=[],
            generation_model="gemini-3.8-flash",
            retrieval_mode="hybrid",
            metadata={},
        )

    monkeypatch.setattr("src.rag_pipeline.answer_question", mock_slow_answer)

    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-3.8-flash"

    # Set hard timeout to 0.2s (less than the 2.0s sleep)
    response = answer_question_with_timeout(
        question="What is the remote work policy?",
        retriever=mock_retriever,
        llm_service=mock_llm,
        timeout_seconds=0.2,
    )

    assert response.refused is True
    assert response.answer == "Clause could not complete the request in time. Please try again."
    assert response.metadata["refusal_stage"] == "ask_timeout_exceeded"


def test_ask_fast_execution_returns_response(monkeypatch):
    """Test that normal fast requests return standard RAGResponse within timeout."""
    mock_response = RAGResponse(
        question="fast query",
        answer="Fast answer",
        sources=[],
        retrieved_results=[],
        grounded=True,
        refused=False,
        confidence=1.0,
        confidence_label="High",
        evidence_status="sufficient",
        conflict_detected=False,
        retrieval_corrected=False,
        evidence_reasons=[],
        generation_model="gemini-3.8-flash",
        retrieval_mode="hybrid",
        metadata={},
    )

    monkeypatch.setattr("src.rag_pipeline.answer_question", lambda *a, **k: mock_response)

    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    mock_llm.model_name = "gemini-3.8-flash"

    response = answer_question_with_timeout(
        question="fast query",
        retriever=mock_retriever,
        llm_service=mock_llm,
        timeout_seconds=5.0,
    )

    assert response.refused is False
    assert response.answer == "Fast answer"
