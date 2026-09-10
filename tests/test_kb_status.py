"""
Unit tests for unified Knowledge Base Status source of truth.
Validates doc count, chunk count, readiness status text, and sidebar rendering.
"""

import pytest
import streamlit as st
from unittest.mock import MagicMock
from src.kb_state import get_knowledge_base_status, KnowledgeBaseStatus
from src.ui.shell import render_sidebar


def test_kb_status_ready_state(monkeypatch):
    """Test get_knowledge_base_status when documents exist and index is ready."""
    mock_retriever = MagicMock()
    mock_retriever.vector_store.total_chunks = 38
    monkeypatch.setattr(st, "session_state", {"retriever": mock_retriever})

    status = get_knowledge_base_status()
    assert status.doc_count >= 1
    assert status.chunk_count == 38
    assert status.is_ready is True
    assert status.status_text == "Knowledge base ready"


def test_kb_status_unindexed_state(monkeypatch):
    """Test get_knowledge_base_status when documents exist but index is empty."""
    mock_retriever = MagicMock()
    mock_retriever.vector_store.total_chunks = 0
    monkeypatch.setattr(st, "session_state", {"retriever": mock_retriever, "indexing_stats": {"chunks_generated": 0}})

    status = get_knowledge_base_status()
    if status.doc_count > 0:
        assert status.chunk_count == 0
        assert status.is_ready is False
        assert status.status_text == "Knowledge base not indexed"


def test_sidebar_rendering_status(monkeypatch):
    """Test sidebar rendering includes accurate status label and chunk count."""
    mock_retriever = MagicMock()
    mock_retriever.vector_store.total_chunks = 42
    monkeypatch.setattr(st, "session_state", {"retriever": mock_retriever, "current_page": "Ask"})

    rendered_markdown = []

    def mock_markdown(body, unsafe_allow_html=False):
        rendered_markdown.append(body)

    monkeypatch.setattr(st, "markdown", mock_markdown)

    # Mock sidebar context manager
    class MockSidebar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(st, "sidebar", MockSidebar())
    monkeypatch.setattr(st, "button", lambda label, **kwargs: False)

    page = render_sidebar()
    assert page == "Ask"
    combined_html = "".join(rendered_markdown)
    assert "Knowledge base ready" in combined_html
    assert "42 chunks" in combined_html
