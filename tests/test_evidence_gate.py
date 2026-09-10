"""
Unit tests for src/evidence_gate.py.
"""

from src.evidence_gate import evaluate_evidence_sufficiency, INSUFFICIENT_INFORMATION_REFUSAL
from src.models import RetrievalResult, DocumentChunk


def test_evidence_gate_empty_results():
    is_sufficient, confidence, reason, diag = evaluate_evidence_sufficiency([])
    assert is_sufficient is False
    assert confidence == 0.0
    assert "Zero document chunks" in reason


def test_evidence_gate_low_score_refusal():
    chunk = DocumentChunk(
        chunk_id="chk1",
        document_id="doc1",
        document_name="test.pdf",
        page_number=1,
        text="Irrelevant content snippet.",
        section="Intro",
        chunk_index=0,
    )
    result = RetrievalResult(
        chunk=chunk,
        document_name="test.pdf",
        page_number=1,
        section="Intro",
        fusion_score=0.001,
        dense_score=0.001,
        bm25_score=0.001,
    )

    is_sufficient, confidence, reason, diag = evaluate_evidence_sufficiency(
        [result], min_score=0.015
    )
    assert is_sufficient is False
    assert confidence == 0.1
    assert "below sufficiency threshold" in reason


def test_evidence_gate_sufficient_evidence():
    chunk = DocumentChunk(
        chunk_id="chk_remote",
        document_id="doc1",
        document_name="remote.pdf",
        page_number=1,
        text="Employees may work remotely up to two days.",
        section="Policy",
        chunk_index=0,
    )
    result = RetrievalResult(
        chunk=chunk,
        document_name="remote.pdf",
        page_number=1,
        section="Policy",
        fusion_score=0.032,
        dense_score=0.45,
        bm25_score=7.5,
    )

    is_sufficient, confidence, reason, diag = evaluate_evidence_sufficiency(
        [result], min_score=0.015
    )
    assert is_sufficient is True
    assert confidence > 0.5
    assert reason is None
    assert diag["gate_passed"] is True
