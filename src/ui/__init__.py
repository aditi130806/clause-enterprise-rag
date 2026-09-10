"""
UI Modular Package for CLAUSE Enterprise RAG.
"""

from src.ui.styles import inject_custom_css
from src.ui.shell import render_sidebar, render_page_header, render_footer
from src.ui.pages import (
    render_ask_page,
    render_documents_page,
    render_evaluation_page,
    render_system_page,
)

__all__ = [
    "inject_custom_css",
    "render_sidebar",
    "render_page_header",
    "render_footer",
    "render_ask_page",
    "render_documents_page",
    "render_evaluation_page",
    "render_system_page",
]
