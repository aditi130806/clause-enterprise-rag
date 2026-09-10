"""
Backward-compatible wrapper for evaluation page tests.
"""

import streamlit as st
from unittest.mock import MagicMock
from src.ui.pages.evaluation import render_evaluation_page
from tests.test_evaluation_pipeline import (
    test_metrics_load_from_evaluation_results_json,
    test_manual_run_evaluation_preserved,
)


def test_evaluation_page_auto_loads_json(monkeypatch):
    """Wrapper verifying evaluation results auto load."""
    test_metrics_load_from_evaluation_results_json(monkeypatch)


def test_run_evaluation_button_executes_pipeline(monkeypatch):
    """Wrapper verifying manual evaluation execution."""
    test_manual_run_evaluation_preserved(monkeypatch)


def test_evaluation_page_empty_state(monkeypatch, tmp_path):
    """Test empty state when evaluation_results.json is missing and auto-eval cannot run."""
    monkeypatch.setattr("src.ui.pages.evaluation.BASE_DIR", tmp_path)
    monkeypatch.setattr("src.ui.pages.evaluation.render_page_header", lambda title, subtitle: None)
    monkeypatch.setattr("src.ui.pages.evaluation.render_footer", lambda: None)
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: None)
    monkeypatch.setattr(st, "table", lambda data: None)

    empty_state_calls = []
    monkeypatch.setattr("src.ui.pages.evaluation.render_empty_state", lambda title, subtitle: empty_state_calls.append((title, subtitle)))

    mock_status = MagicMock()
    mock_status.is_ready = False
    mock_status.retriever = None
    monkeypatch.setattr("src.ui.pages.evaluation.get_knowledge_base_status", lambda: mock_status)

    render_evaluation_page()

    assert len(empty_state_calls) >= 1
    assert any("No evaluation results available." in c[0] for c in empty_state_calls)
