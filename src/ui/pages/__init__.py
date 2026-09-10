"""
Page Modules Package for CLAUSE.
"""

from src.ui.pages.ask import render_ask_page
from src.ui.pages.documents import render_documents_page
from src.ui.pages.evaluation import render_evaluation_page
from src.ui.pages.system import render_system_page

__all__ = [
    "render_ask_page",
    "render_documents_page",
    "render_evaluation_page",
    "render_system_page",
]
