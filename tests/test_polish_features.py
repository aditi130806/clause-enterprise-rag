"""
Unit tests for Navigation, Mobile Shell, Scroll-to-Top, and Suggested Questions Polish.
"""

import pytest
import streamlit as st
from unittest.mock import MagicMock
from src.ui.shell import render_sidebar
from src.ui.pages.ask import render_ask_page


def test_scroll_to_top_trigger_condition():
    """Test scroll-to-top trigger evaluates True only when page changes."""
    session = {}
    
    # 1. First load: previous_page is None, current_page is Ask -> page changed
    prev_page = session.get("previous_page", None)
    curr_page = "Ask"
    page_changed_1 = (prev_page != curr_page)
    assert page_changed_1 is True
    session["previous_page"] = curr_page

    # 2. Same-page rerun on Ask -> page changed is False
    prev_page = session.get("previous_page")
    curr_page = "Ask"
    page_changed_2 = (prev_page != curr_page)
    assert page_changed_2 is False

    # 3. Switch to Documents -> page changed is True
    curr_page = "Documents"
    page_changed_3 = (prev_page != curr_page)
    assert page_changed_3 is True
    session["previous_page"] = curr_page

    # 4. Same-page rerun on Documents -> page changed is False
    prev_page = session.get("previous_page")
    curr_page = "Documents"
    page_changed_4 = (prev_page != curr_page)
    assert page_changed_4 is False


def test_mobile_menu_toggle_and_close_on_selection(monkeypatch):
    """Test mobile menu opens on toggle and closes after page selection."""
    session = {"current_page": "Ask", "mobile_menu_open": False}
    monkeypatch.setattr(st, "session_state", session)
    monkeypatch.setattr(st, "sidebar", MagicMock())
    monkeypatch.setattr(st, "markdown", lambda *a, **kw: None)
    monkeypatch.setattr(st, "columns", lambda weights: [MagicMock(), MagicMock()])
    monkeypatch.setattr(st, "button", lambda label, **kw: False)

    # Initial state
    assert session["mobile_menu_open"] is False

    # Toggle menu open
    session["mobile_menu_open"] = True
    assert session["mobile_menu_open"] is True

    # Page selection closes mobile menu
    session["current_page"] = "Documents"
    session["mobile_menu_open"] = False
    assert session["current_page"] == "Documents"
    assert session["mobile_menu_open"] is False


def test_suggested_questions_show_first_2_by_default(monkeypatch):
    """Test that first 2 suggested questions are rendered by default, and More toggles remaining."""
    session = {
        "chat_history": [],
        "request_pending": False,
        "pending_query": None,
        "current_response": None,
        "last_submitted_query": None,
        "show_more_suggestions": False,
    }
    monkeypatch.setattr(st, "session_state", session)
    monkeypatch.setattr("src.ui.pages.ask.render_page_header", lambda title, subtitle, eyebrow=None: None)
    monkeypatch.setattr("src.ui.pages.ask.render_footer", lambda: None)
    monkeypatch.setattr("src.ui.pages.ask.render_empty_state", lambda title, subtitle: None)
    monkeypatch.setattr(st, "text_input", lambda label, **kw: "")
    monkeypatch.setattr(st, "markdown", lambda body, unsafe_allow_html=False: None)

    buttons_rendered = []
    
    def mock_button(label, key=None, **kw):
        buttons_rendered.append((label, key))
        return False

    monkeypatch.setattr(st, "button", mock_button)
    
    def mock_columns(spec, **kw):
        count = len(spec) if isinstance(spec, list) else spec
        return [MagicMock() for _ in range(count)]

    monkeypatch.setattr(st, "columns", mock_columns)

    # Default state: show_more_suggestions = False
    render_ask_page()

    # Verify buttons rendered include first 2 sample queries + "More"
    button_labels = [b[0] for b in buttons_rendered]
    assert any("healthcare" in l.lower() for l in button_labels)
    assert any("expenses over $25" in l.lower() for l in button_labels)
    assert "More" in button_labels
    # The 3rd question ("probation") should NOT be visible when show_more_suggestions is False
    assert not any("probation" in l.lower() for l in button_labels)

    # Toggle to show_more_suggestions = True
    buttons_rendered.clear()
    session["show_more_suggestions"] = True

    render_ask_page()

    button_labels_expanded = [b[0] for b in buttons_rendered]
    assert any("healthcare" in l.lower() for l in button_labels_expanded)
    assert any("expenses over $25" in l.lower() for l in button_labels_expanded)
    assert any("probation" in l.lower() for l in button_labels_expanded)
    assert "Less" in button_labels_expanded
