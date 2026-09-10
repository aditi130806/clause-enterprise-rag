"""
Application Shell, Navigation Sidebar, and Header Components for CLAUSE.
Strictly eliminates emojis, duplicate logos, raw HTML leaks, and overflow.
"""

from pathlib import Path
import streamlit as st
from src.config import BASE_DIR

# Monochrome Outline SVG Icons (1.5px stroke, 18px size, no emojis)
ICON_ASK_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:middle;"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>"""

ICON_DOCS_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:middle;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>"""

ICON_EVAL_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:middle;"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>"""

ICON_SYS_SVG = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:middle;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>"""


def load_logo_svg() -> str:
    """Load clean SVG logo asset or return fallback SVG string."""
    svg_path = BASE_DIR / "assets" / "logo" / "clause_logo.svg"
    if svg_path.exists():
        try:
            return svg_path.read_text(encoding="utf-8")
        except Exception:
            pass

    return """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="8 8 135 38" width="135" height="38">
      <g transform="translate(4, 5)">
        <path d="M 18 36 C 8 30 4 18 8 9 C 14 5 25 9 22 22 C 21 28 19 33 18 36 Z" fill="#24533F"/>
        <path d="M 19 26 C 21 19 28 13 33 11 C 35 16 32 25 25 30 C 22 32 20 33 19 26 Z" fill="#24533F"/>
        <path d="M 18 36 C 17 38 16 40 15 41 C 14 42 14 41 15 40 C 16 38 17 37 18 36 Z" fill="#24533F"/>
      </g>
      <text x="48" y="33" font-family="'Playfair Display', Georgia, serif" font-size="28" font-weight="700" fill="#24533F" letter-spacing="0.5">Clause</text>
    </svg>
    """


from typing import Optional


def render_sidebar(doc_count: Optional[int] = None, chunk_count: Optional[int] = None) -> str:
    """
    Render cohesive left sidebar with active knowledge base status for desktop,
    and compact top navigation header for mobile.

    Returns:
        Selected page name ('Ask', 'Documents', 'Evaluation', 'System').
    """
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Ask"
    if "mobile_menu_open" not in st.session_state:
        st.session_state["mobile_menu_open"] = False

    from src.kb_state import get_knowledge_base_status
    kb_status = get_knowledge_base_status()

    final_doc_count = kb_status.doc_count if doc_count is None else doc_count
    final_chunk_count = kb_status.chunk_count if chunk_count is None else chunk_count

    if kb_status.is_ready and final_chunk_count > 0:
        dot_html = '<span style="color: #4F7C63; font-size: 10px;">●</span>'
        status_label = "Knowledge base ready"
    elif final_doc_count > 0:
        dot_html = '<span style="color: #949692; font-size: 10px;">○</span>'
        status_label = "Knowledge base not indexed"
    else:
        dot_html = '<span style="color: #949692; font-size: 10px;">○</span>'
        status_label = "No documents uploaded"

    # 1. Desktop Left Sidebar (>= 769px)
    with st.sidebar:
        # Top Logo Only (No giant banners, no duplicate icons)
        logo_svg = load_logo_svg().replace("\n", "").strip()
        st.markdown(
            f'<div style="height: 72px; display: flex; align-items: center; justify-content: flex-start; padding-left: 24px; padding-right: 20px; margin-top: 20px; margin-bottom: 8px;">{logo_svg}</div>',
            unsafe_allow_html=True,
        )

        # Navigation Links
        nav_items = [
            ("Ask", "Ask"),
            ("Documents", "Documents"),
            ("Evaluation", "Evaluation"),
            ("System", "System"),
        ]

        for page_key, label in nav_items:
            is_active = st.session_state["current_page"] == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{page_key}", type=btn_type, use_container_width=True):
                st.session_state["current_page"] = page_key
                st.rerun()

        # Spacing to push status module to bottom
        st.markdown('<div style="margin-top: 100px;"></div>', unsafe_allow_html=True)

        # Bottom Sidebar Status Module (Restrained, dynamic KB status)
        st.markdown(
            f'<div style="padding-top: 0.85rem; border-top: 1px solid #E3E3DE; font-size: 13px; color: #6E716D;"><div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px; font-weight: 600; color: #1C1E1C;">{dot_html} {status_label}</div><div style="padding-left: 12px; color: #6E716D; font-size: 12px;">{final_doc_count} documents · {final_chunk_count:,} chunks</div></div>',
            unsafe_allow_html=True,
        )

    # 2. Mobile Top Navigation Bar (<= 768px)
    st.markdown('<div class="mobile-header-bar">', unsafe_allow_html=True)
    m_col1, m_col2 = st.columns([4, 1])
    with m_col1:
        logo_svg = load_logo_svg().replace("\n", "").strip()
        st.markdown(f'<div class="mobile-logo-wrap">{logo_svg}</div>', unsafe_allow_html=True)
    with m_col2:
        btn_label = "✕ Menu" if st.session_state["mobile_menu_open"] else "☰ Menu"
        if st.button(btn_label, key="mob_menu_toggle_btn", use_container_width=True):
            st.session_state["mobile_menu_open"] = not st.session_state["mobile_menu_open"]
            st.rerun()

    if st.session_state["mobile_menu_open"]:
        st.markdown('<div class="mobile-menu-card">', unsafe_allow_html=True)
        nav_items = ["Ask", "Documents", "Evaluation", "System"]
        for page_key in nav_items:
            is_active = st.session_state["current_page"] == page_key
            btn_type = "primary" if is_active else "secondary"
            if st.button(page_key, key=f"mob_nav_{page_key}", type=btn_type, use_container_width=True):
                st.session_state["current_page"] = page_key
                st.session_state["mobile_menu_open"] = False
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    return st.session_state["current_page"]


def render_page_header(title: str, subtitle: str, eyebrow: str = None):
    """Render compact, unified top header area across all pages."""
    eyebrow_html = f"<div class='eyebrow-text'>{eyebrow}</div>" if eyebrow else ""
    header_html = f'<div style="margin-bottom: 1.25rem;">{eyebrow_html}<h1 class="page-title">{title}</h1><div class="page-subtitle">{subtitle}</div></div>'
    st.markdown(header_html, unsafe_allow_html=True)


def render_footer():
    """Render restrained 12px text footer without giant logos."""
    st.markdown('<div class="clause-minimal-footer">Clause · Enterprise Document Intelligence</div>', unsafe_allow_html=True)
