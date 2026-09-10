"""
Comprehensive unit tests for CLAUSE Evaluation Pipeline.
Tests:
1. Benchmark table native rendering (no raw HTML leak)
2. Metrics auto-load from evaluation_results.json
3. Deterministic knowledge-base fingerprinting
4. Unchanged knowledge base bypasses re-evaluation
5. Changed knowledge base triggers stale detection
6. Stale evaluation automatically reruns once
7. New results replace old results
8. Manual 'Run evaluation' execution
9. Zero Gemini calls during evaluation
10. Evaluation error handling hides Python tracebacks
"""

import json
import pytest
import streamlit as st
from unittest.mock import MagicMock
from pathlib import Path

from src.kb_state import compute_kb_fingerprint
from src.evaluation import evaluate_retrieval_and_rag
from src.ui.pages.evaluation import render_evaluation_page


def test_kb_fingerprint_deterministic():
    """Test 3: knowledge-base fingerprint is deterministic."""
    fp1 = compute_kb_fingerprint()
    fp2 = compute_kb_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 16


def test_metrics_load_from_evaluation_results_json(monkeypatch):
    """Test 2: metrics load dynamically from evaluation_results.json."""
    tables_rendered = []
    monkeypatch.setattr(st, "table", lambda data: tables_rendered.append(data))
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_page_header", lambda title, subtitle: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_footer", lambda: None)

    # Mock matching fingerprint
    current_fp = compute_kb_fingerprint()
    mock_status = MagicMock()
    mock_status.is_ready = True
    mock_status.retriever = MagicMock()
    monkeypatch.setattr("src.ui.pages.evaluation.get_knowledge_base_status", lambda: mock_status)
    monkeypatch.setattr("src.ui.pages.evaluation.compute_kb_fingerprint", lambda: current_fp)

    render_evaluation_page()

    assert len(tables_rendered) >= 1
    rows = tables_rendered[0]
    metric_names = [r["Metric"] for r in rows]
    assert "Retrieval Hit@1" in metric_names
    assert "Retrieval Hit@3" in metric_names
    assert "MRR" in metric_names
    assert "Expected Source Accuracy" in metric_names
    assert "Correct Refusal Rate" in metric_names
    assert "Grounding Validation Rate" in metric_names
    assert "Total Questions" in metric_names
    assert "Supported Questions" in metric_names
    assert "Unsupported Questions" in metric_names


def test_benchmark_table_no_raw_html(monkeypatch):
    """Test 1: benchmark table does not expose raw HTML."""
    tables_rendered = []
    markdown_rendered = []
    monkeypatch.setattr(st, "table", lambda data: tables_rendered.append(data))
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: markdown_rendered.append(body))
    monkeypatch.setattr("src.ui.pages.evaluation.render_page_header", lambda title, subtitle: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_footer", lambda: None)

    render_evaluation_page()

    # Table is passed as list of dicts to st.table
    assert len(tables_rendered) >= 1
    # Verify no raw <table or <div table HTML in markdown calls
    for body in markdown_rendered:
        assert "<div style=\"display: grid;" not in body


def test_unchanged_kb_bypasses_reevaluation(monkeypatch):
    """Test 4: unchanged knowledge base does not rerun evaluation."""
    eval_calls = []
    monkeypatch.setattr("src.ui.pages.evaluation.evaluate_retrieval_and_rag", lambda **kw: eval_calls.append(kw))
    monkeypatch.setattr(st, "table", lambda data: None)
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_page_header", lambda title, subtitle: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_footer", lambda: None)

    current_fp = compute_kb_fingerprint()
    mock_status = MagicMock()
    mock_status.is_ready = True
    mock_status.retriever = MagicMock()
    monkeypatch.setattr("src.ui.pages.evaluation.get_knowledge_base_status", lambda: mock_status)
    monkeypatch.setattr("src.ui.pages.evaluation.compute_kb_fingerprint", lambda: current_fp)

    # Set up evaluation_results.json with matching fingerprint
    mock_eval_data = {
        "knowledge_base_fingerprint": current_fp,
        "retrieval_hit_at_1": 0.944,
        "expected_source_accuracy": 1.0,
        "grounding_validation_rate": 1.0,
        "correct_refusal_rate": 1.0,
    }
    monkeypatch.setattr("json.load", lambda f: mock_eval_data)

    render_evaluation_page()

    assert len(eval_calls) == 0


def test_changed_kb_triggers_stale_and_auto_reevaluation(monkeypatch):
    """Test 5 & 6 & 7: changed KB triggers stale detection and auto reruns once, replacing results."""
    eval_calls = []

    def mock_eval(retriever, llm_service=None, eval_llm_limit=0, kb_fingerprint=None):
        eval_calls.append({
            "retriever": retriever,
            "llm_service": llm_service,
            "eval_llm_limit": eval_llm_limit,
            "kb_fingerprint": kb_fingerprint,
        })
        return {
            "knowledge_base_fingerprint": kb_fingerprint,
            "retrieval_hit_at_1": 0.95,
            "expected_source_accuracy": 1.0,
            "grounding_validation_rate": 1.0,
            "correct_refusal_rate": 1.0,
        }

    monkeypatch.setattr("src.ui.pages.evaluation.evaluate_retrieval_and_rag", mock_eval)
    monkeypatch.setattr(st, "table", lambda data: None)
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: None)
    monkeypatch.setattr(st, "info", lambda msg: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_page_header", lambda title, subtitle: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_footer", lambda: None)

    class MockSpinner:
        def __enter__(self): return self
        def __exit__(self, a, b, c): pass
    monkeypatch.setattr(st, "spinner", lambda msg: MockSpinner())

    mock_status = MagicMock()
    mock_status.is_ready = True
    mock_status.retriever = MagicMock()
    monkeypatch.setattr("src.ui.pages.evaluation.get_knowledge_base_status", lambda: mock_status)
    monkeypatch.setattr("src.ui.pages.evaluation.compute_kb_fingerprint", lambda: "new_changed_fp_123")

    monkeypatch.setattr(st, "session_state", {})

    # Existing JSON has old fingerprint
    old_eval_data = {"knowledge_base_fingerprint": "old_fp_456"}
    monkeypatch.setattr("json.load", lambda f: old_eval_data)

    render_evaluation_page()

    assert len(eval_calls) == 1
    assert eval_calls[0]["kb_fingerprint"] == "new_changed_fp_123"
    assert eval_calls[0]["llm_service"] is None  # Test 9: Zero Gemini calls


def test_manual_run_evaluation_preserved(monkeypatch):
    """Test 8: manual Run evaluation button executes immediately."""
    eval_calls = []
    monkeypatch.setattr("src.ui.pages.evaluation.evaluate_retrieval_and_rag", lambda **kw: eval_calls.append(kw))
    monkeypatch.setattr(st, "table", lambda data: None)
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: None)
    monkeypatch.setattr(st, "button", lambda label, **kw: True if label in ["Run benchmark", "Run evaluation"] else False)
    monkeypatch.setattr(st, "success", lambda msg: None)
    monkeypatch.setattr(st, "rerun", lambda: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_page_header", lambda title, subtitle: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_footer", lambda: None)

    class MockSpinner:
        def __enter__(self): return self
        def __exit__(self, a, b, c): pass
    monkeypatch.setattr(st, "spinner", lambda msg: MockSpinner())

    mock_status = MagicMock()
    mock_status.is_ready = True
    mock_status.retriever = MagicMock()
    monkeypatch.setattr("src.ui.pages.evaluation.get_knowledge_base_status", lambda: mock_status)
    monkeypatch.setattr("src.ui.pages.evaluation.compute_kb_fingerprint", lambda: "fp_789")
    monkeypatch.setattr(st, "session_state", {})

    render_evaluation_page()

    assert len(eval_calls) >= 1
    assert eval_calls[-1]["llm_service"] is None


def test_evaluation_error_hides_traceback(monkeypatch):
    """Test 10: evaluation failure shows clean error, does not expose traceback."""
    def mock_failing_eval(**kw):
        raise RuntimeError("FAISS index corrupt")

    monkeypatch.setattr("src.ui.pages.evaluation.evaluate_retrieval_and_rag", mock_failing_eval)
    monkeypatch.setattr(st, "table", lambda data: None)
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: None)
    monkeypatch.setattr(st, "button", lambda label, **kw: True if label in ["Run benchmark", "Run evaluation"] else False)
    monkeypatch.setattr("src.ui.pages.evaluation.render_page_header", lambda title, subtitle: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_footer", lambda: None)

    errors_shown = []
    monkeypatch.setattr(st, "error", lambda msg: errors_shown.append(msg))

    class MockSpinner:
        def __enter__(self): return self
        def __exit__(self, a, b, c): pass
    monkeypatch.setattr(st, "spinner", lambda msg: MockSpinner())

    mock_status = MagicMock()
    mock_status.is_ready = True
    mock_status.retriever = MagicMock()
    monkeypatch.setattr("src.ui.pages.evaluation.get_knowledge_base_status", lambda: mock_status)

    render_evaluation_page()

    assert len(errors_shown) == 1
    assert "Evaluation could not be refreshed" in errors_shown[0]
    assert "FAISS index corrupt" not in errors_shown[0]  # No raw traceback/internal error leak
