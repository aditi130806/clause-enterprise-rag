"""
Reusable UI Components for CLAUSE Enterprise RAG.
Strictly valid HTML strings, no raw tag leaks, no emojis, brand palette compliant.
"""

import html
import re
from typing import Optional
import streamlit as st
from src.models import RAGResponse, SourceReference, RetrievalResult


def status_badge_html(text: str, variant: str = "success") -> str:
    """Generate HTML string for status badge (success/warning/error)."""
    css_class = f"badge-pill badge-{variant}"
    escaped_text = html.escape(text)
    return f'<span class="{css_class}">● {escaped_text}</span>'


def render_empty_state(title: str, subtitle: str):
    """Render compact 120-150px centered empty state box."""
    html_code = f'<div class="empty-state-box"><div style="font-size: 15px; font-weight: 600; color: #1C1E1C; margin-bottom: 4px;">{html.escape(title)}</div><div style="font-size: 13px; color: #6E716D;">{html.escape(subtitle)}</div></div>'
    st.markdown(html_code, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = None):
    """Render compact metric card with restrained typography."""
    sub_html = f"<div style='font-size: 12px; color: #6E716D; margin-top: 2px;'>{html.escape(subtext)}</div>" if subtext else ""
    html_code = f'<div class="clause-compact-card"><div style="font-size: 12px; font-weight: 600; color: #6E716D; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 4px;">{html.escape(label)}</div><div style="font-size: 26px; font-weight: 700; color: #1C1E1C; line-height: 1.1;">{html.escape(value)}</div>{sub_html}</div>'
    st.markdown(html_code, unsafe_allow_html=True)


def format_citations(answer_text: str) -> str:
    """Format inline citations [S1], [S2] into brand citation badges."""
    escaped = html.escape(answer_text)
    pattern = r"\[S(\d+)\]"
    formatted = re.sub(pattern, r'<span class="citation-tag">[S\1]</span>', escaped)
    paragraphs = formatted.split("\n\n")
    return "".join(f"<p style='margin-bottom: 10px; line-height: 1.55;'>{p}</p>" for p in paragraphs if p.strip())


def render_question_answer_card(response: RAGResponse, timestamp: str = "Today, 10:24 AM"):
    """Render Question and Clause Answer cards."""
    # 1. Question Card
    q_html = f'<div class="clause-card" style="margin-bottom: 12px;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;"><span style="font-size: 12px; font-weight: 600; color: #6E716D; text-transform: uppercase;">Your question</span><span style="font-size: 12px; color: #6E716D;">{html.escape(timestamp)}</span></div><div style="font-size: 16px; font-weight: 600; color: #1C1E1C;">{html.escape(response.question)}</div></div>'
    st.markdown(q_html, unsafe_allow_html=True)

    # 2. Answer Card
    if response.refused and not response.metadata.get("fallback_used"):
        badge_html = status_badge_html("Insufficient evidence", "error")
    elif response.evidence_status == "Evidence-backed fallback" or response.metadata.get("fallback_used"):
        badge_html = status_badge_html("Evidence-backed fallback", "warning")
    else:
        label = response.confidence_label or "High"
        badge_html = status_badge_html(f"{label} confidence", "success")

    formatted_body = format_citations(response.answer)
    fallback_note_html = ""
    if response.metadata.get("fallback_note"):
        note = html.escape(response.metadata["fallback_note"])
        fallback_note_html = f'<div style="font-size: 12px; color: #6E716D; font-style: italic; margin-top: 8px; margin-bottom: 12px; padding: 8px 12px; background-color: #F7F6F2; border-left: 3px solid #D97706; border-radius: 0 4px 4px 0;">{note}</div>'

    a_html = f'<div class="clause-card"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #E3E3DE; padding-bottom: 10px;"><span style="font-size: 15px; font-weight: 700; color: #1C1E1C;">Clause answer</span>{badge_html}</div><div style="font-size: 15px; color: #1C1E1C; margin-bottom: 12px;">{formatted_body}</div>{fallback_note_html}<div style="font-size: 13px; font-weight: 600; color: #6E716D; margin-bottom: 8px; border-top: 1px solid #E3E3DE; padding-top: 12px;">CITED SOURCES</div></div>'
    st.markdown(a_html, unsafe_allow_html=True)

    # Sources List Accordion
    if response.sources:
        for idx, src in enumerate(response.sources, start=1):
            page_str = f"Page {src.page_number}" if src.page_number else "N/A"
            sec_str = src.section or "General"
            with st.expander(f"{idx}. {src.document_name}  |  {page_str}  |  {sec_str}"):
                st.markdown(f"**Excerpt:** *\"{html.escape(src.excerpt)}\"*")
    else:
        st.caption("No sources cited for this response.")


import logging
logger = logging.getLogger(__name__)


def render_evidence_panel(response: RAGResponse):
    """Render right-side Evidence panel matching desktop grid column safely without crashing."""
    try:
        st.markdown('<div style="font-size: 15px; font-weight: 700; color: #1C1E1C; margin-bottom: 12px;">Evidence</div>', unsafe_allow_html=True)

        is_insufficient = (
            response.refused
            or response.confidence_label == "Insufficient"
            or not response.sources
        )

        if is_insufficient:
            st.markdown(
                '<div class="clause-card" style="padding: 16px;"><div style="font-size: 13px; color: #6E716D;">No supporting evidence was found for this question.</div></div>',
                unsafe_allow_html=True,
            )
            return

        # Use response.sources[0] as primary validated source of truth
        first_src: SourceReference = response.sources[0]
        doc_name = first_src.document_name
        page_num = first_src.page_number
        page_str = f"Page {page_num}" if page_num is not None else "N/A"
        sec_str = first_src.section or "General Section"
        excerpt_text = first_src.excerpt or ""

        if len(excerpt_text) > 220:
            excerpt_text = excerpt_text[:220] + "..."

        rank_val = first_src.retrieval_rank or 1
        score_val = first_src.retrieval_score or 0.0

        # Retrieve method from matched result if available safely
        methods_str = "HYBRID (RRF)"
        if response.retrieved_results:
            for res in response.retrieved_results:
                if res.chunk and res.chunk.chunk_id == first_src.chunk_id:
                    if res.retrieval_methods:
                        methods_str = ", ".join(res.retrieval_methods).upper()
                    break

        ev_html = (
            f'<div class="clause-card" style="padding: 16px;">'
            f'<div style="font-size: 14px; font-weight: 600; color: #1C1E1C; margin-bottom: 4px;">{html.escape(doc_name)}</div>'
            f'<div style="font-size: 12px; color: #6E716D; margin-bottom: 2px;">{html.escape(page_str)}</div>'
            f'<div style="font-size: 12px; color: #6E716D; margin-bottom: 12px;">{html.escape(sec_str)}</div>'
            f'<div style="background-color: #F7F6F2; border-left: 2px solid #24533F; padding: 10px 12px; font-size: 13px; font-style: italic; color: #1C1E1C; margin-bottom: 12px; border-radius: 0 4px 4px 0; line-height: 1.45;">"{html.escape(excerpt_text)}"</div>'
            f'<div style="border-top: 1px solid #E3E3DE; padding-top: 10px;">'
            f'<div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;"><span style="color: #6E716D;">Retrieval rank</span><span style="font-weight: 600; color: #1C1E1C;">#{rank_val}</span></div>'
            f'<div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;"><span style="color: #6E716D;">Relevance score</span><span style="font-weight: 600; color: #1C1E1C;">{score_val:.2f}</span></div>'
            f'<div style="display: flex; justify-content: space-between; font-size: 12px;"><span style="color: #6E716D;">Retrieval method</span><span style="font-weight: 600; color: #1C1E1C;">{html.escape(methods_str)}</span></div>'
            f'</div></div>'
        )
        st.markdown(ev_html, unsafe_allow_html=True)
    except Exception as exc:
        logger.error(f"Error rendering evidence panel: {exc}", exc_info=True)
        st.markdown(
            '<div class="clause-card" style="padding: 16px;"><div style="font-size: 13px; color: #6E716D;">Evidence details unavailable.</div></div>',
            unsafe_allow_html=True,
        )


def render_document_inventory_table(files):
    """Render Document Inventory Table with strict HTML string safety."""
    if not files:
        render_empty_state("No documents indexed yet", "Upload PDF or DOCX files above to create your knowledge base.")
        return

    rows_html = ""
    for f in files:
        ext = f.suffix.upper().replace(".", "")
        size_kb = f.stat().st_size / 1024
        rows_html += f"<tr><td><strong>{html.escape(f.name)}</strong></td><td>{html.escape(ext)}</td><td>{size_kb:.1f} KB</td><td>{status_badge_html('Indexed', 'success')}</td></tr>"

    table_html = f'<table class="clause-table"><thead><tr><th>Document Name</th><th>Type</th><th>Size</th><th>Status</th></tr></thead><tbody>{rows_html}</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)


def render_architecture_pipeline():
    """Render horizontal visual architecture pipeline without emojis or oversized cards."""
    steps = [
        ("1. Documents", "PDF & DOCX Ingest"),
        ("2. Text Cleaning", "Header Normalization"),
        ("3. Chunking", "700 Char Window"),
        ("4. Embeddings", "MiniLM-L6-v2 ONNX"),
        ("5. Hybrid Search", "FAISS + BM25"),
        ("6. RRF Fusion", "Reciprocal Rank"),
        ("7. Evidence Gate", "Sufficiency Score"),
        ("8. Gemini LLM", "Grounded QA"),
        ("9. Citation Check", "Strict Attribution"),
    ]

    cols = st.columns(3)
    for idx, (title, desc) in enumerate(steps):
        with cols[idx % 3]:
            card_html = f'<div class="clause-compact-card"><div style="font-size: 13px; font-weight: 700; color: #24533F; margin-bottom: 2px;">{html.escape(title)}</div><div style="font-size: 12px; color: #6E716D;">{html.escape(desc)}</div></div>'
            st.markdown(card_html, unsafe_allow_html=True)


def render_configuration_table(config_items):
    """Render configuration key-value table."""
    rows_html = ""
    for key, val in config_items:
        rows_html += f'<tr><td style="font-weight: 600; width: 30%;">{html.escape(key)}</td><td>{html.escape(str(val))}</td></tr>'

    table_html = f'<table class="clause-table"><thead><tr><th>System Setting</th><th>Active Configuration</th></tr></thead><tbody>{rows_html}</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)
