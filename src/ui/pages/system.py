"""
Page 4: System - Pipeline Architecture & Active Configuration.
"""

import streamlit as st
from src.config import EMBEDDING_MODEL_NAME, GEMINI_MODEL
from src.ui.shell import render_page_header, render_footer
from src.ui.components import render_architecture_pipeline, render_configuration_table


def render_system_page():
    """Render System architecture and configuration page."""
    render_page_header(
        title="System",
        subtitle="Architecture, components, and active configuration.",
    )

    # Pipeline Section
    st.markdown("<div class='section-title'>End-to-End RAG Architecture Pipeline</div>", unsafe_allow_html=True)
    render_architecture_pipeline()

    # Active Configuration Table Section
    st.markdown("<div class='section-title'>Active System Configuration</div>", unsafe_allow_html=True)
    config_items = [
        ("Embedding Model", EMBEDDING_MODEL_NAME),
        ("Embedding Backend", "FastEmbed / ONNX Runtime"),
        ("Vector Search Engine", "FAISS IndexFlatIP (Inner Product)"),
        ("Keyword Search Engine", "BM25 (rank_bm25 / BM25Okapi)"),
        ("Retrieval Fusion", "Reciprocal Rank Fusion (RRF, k=60)"),
        ("Language Model", f"{GEMINI_MODEL} (via google-genai 2.22.0)"),
        ("Frontend Framework", "Streamlit 1.40+ (CLAUSE Enterprise System)"),
        ("Security Policy", "API keys masked; credentials loaded via environment"),
    ]
    render_configuration_table(config_items)

    render_footer()
