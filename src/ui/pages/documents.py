"""
Page 2: Documents - Knowledge Base Ingestion & Management with Document Removal Controls.
"""

import html
from pathlib import Path
import streamlit as st

from src.rag_pipeline import build_and_index_documents
from src.config import DOCUMENTS_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL_NAME
from src.ui.shell import render_page_header, render_footer
from src.ui.components import (
    render_metric_card,
    render_empty_state,
    status_badge_html,
)


def render_documents_page():
    """Render Documents management page with removal and clear controls."""
    render_page_header(
        title="Documents",
        subtitle="Manage and index your enterprise knowledge base.",
    )

    doc_dir = Path(DOCUMENTS_DIR)
    doc_dir.mkdir(parents=True, exist_ok=True)
    vstore_dir = Path(VECTORSTORE_DIR)

    # Clean Upload Card
    st.markdown("<div class='clause-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 15px; font-weight: 700; color: #1C1E1C; margin-bottom: 4px;'>Upload documents</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 13px; color: #6E716D; margin-bottom: 12px;'>Upload PDF or DOCX enterprise policy files to expand knowledge.</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        if st.button("Process & index", type="primary", use_container_width=True):
            if uploaded_files:
                for uf in uploaded_files:
                    if not uf.name.startswith("~$"):
                        target_path = doc_dir / uf.name
                        target_path.write_bytes(uf.getbuffer())
                st.success(f"Saved {len(uploaded_files)} file(s).")

            with st.spinner("Processing & indexing documents..."):
                retriever, stats = build_and_index_documents(
                    documents_dir=doc_dir,
                    vectorstore_dir=vstore_dir,
                )
                st.session_state["retriever"] = retriever
                st.session_state["indexing_stats"] = stats
                try:
                    from src.ui.pages.ask import get_cached_retrieval_engine
                    get_cached_retrieval_engine.clear()
                except Exception:
                    pass
                st.success("Indexing completed.")
                st.rerun()

    with bcol2:
        if st.button("Re-index all", type="secondary", use_container_width=True):
            with st.spinner("Rebuilding indices..."):
                retriever, stats = build_and_index_documents(
                    documents_dir=doc_dir,
                    vectorstore_dir=vstore_dir,
                )
                st.session_state["retriever"] = retriever
                st.session_state["indexing_stats"] = stats
                st.success("Re-indexing completed.")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Filter out MS Word temporary files (~$)
    doc_files = [
        f for f in sorted(list(doc_dir.glob("*.pdf")) + list(doc_dir.glob("*.docx")))
        if not f.name.startswith("~$")
    ]
    from src.kb_state import get_knowledge_base_status
    kb_status = get_knowledge_base_status()
    stats = st.session_state.get("indexing_stats", {})

    total_chunks = kb_status.chunk_count

    # Compact Summary 4-Column Metric Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Documents", str(kb_status.doc_count))
    with m2:
        render_metric_card("Pages", str(stats.get("pages_processed", kb_status.doc_count * 2 if kb_status.doc_count else 0)))
    with m3:
        render_metric_card("Chunks", str(total_chunks))
    with m4:
        render_metric_card("Index Status", "Indexed" if kb_status.is_ready else ("Not indexed" if kb_status.doc_count > 0 else "Empty"))

    st.markdown(f"<div style='font-size: 12px; color: #6E716D; margin-top: 4px; margin-bottom: 20px;'>Embedding Model: <strong>{EMBEDDING_MODEL_NAME}</strong></div>", unsafe_allow_html=True)

    # Action Bar: Document Inventory Title & Clear Knowledge Base Action
    st.markdown("<div style='border-top: 1px solid #E3E3DE; margin-top: 16px; margin-bottom: 16px;'></div>", unsafe_allow_html=True)
    
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        st.markdown("<div class='section-title' style='margin: 0;'>Document Inventory</div>", unsafe_allow_html=True)
    with tcol2:
        if doc_files and st.button("Clear knowledge base", key="btn_clear_kb", type="secondary", use_container_width=True):
            st.session_state["confirm_clear_kb"] = True

    if st.session_state.get("confirm_clear_kb"):
        st.warning("Clear all documents from the active knowledge base?")
        ccol1, ccol2, _ = st.columns([1, 1, 2])
        with ccol1:
            if st.button("Yes, clear all", key="btn_confirm_clear_yes", type="primary", use_container_width=True):
                for f in doc_dir.glob("*"):
                    if f.is_file() and not f.name.startswith("."):
                        try:
                            f.unlink()
                        except Exception:
                            pass
                retriever, stats = build_and_index_documents(
                    documents_dir=doc_dir,
                    vectorstore_dir=vstore_dir,
                )
                st.session_state["retriever"] = retriever
                st.session_state["indexing_stats"] = stats
                st.session_state["confirm_clear_kb"] = False
                st.success("Knowledge base cleared.")
                st.rerun()
        with ccol2:
            if st.button("Cancel", key="btn_confirm_clear_cancel", type="secondary", use_container_width=True):
                st.session_state["confirm_clear_kb"] = False
                st.rerun()

    # Inventory Table with Per-Document Remove Action
    if not doc_files:
        render_empty_state("No documents indexed yet", "Upload PDF or DOCX files above to create your knowledge base.")
    else:
        st.markdown("""
        <div style="display: grid; grid-template-columns: 3fr 1fr 1fr 1.2fr 1.2fr; gap: 8px; padding: 8px 12px; background-color: #F7F6F2; border-bottom: 1px solid #E3E3DE; font-size: 12px; font-weight: 700; color: #6E716D; text-transform: uppercase;">
            <div>Document Name</div>
            <div>Type</div>
            <div>Size</div>
            <div>Status</div>
            <div>Actions</div>
        </div>
        """, unsafe_allow_html=True)

        for idx, f in enumerate(doc_files):
            ext = f.suffix.upper().replace(".", "")
            size_kb = f.stat().st_size / 1024 if f.exists() else 0.0
            badge = status_badge_html("Indexed", "success")

            cols = st.columns([3, 1, 1, 1.2, 1.2])
            with cols[0]:
                st.markdown(f"<div style='font-size: 13px; font-weight: 600; color: #1C1E1C; padding-top: 6px;'>{html.escape(f.name)}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div style='font-size: 13px; color: #6E716D; padding-top: 6px;'>{html.escape(ext)}</div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div style='font-size: 13px; color: #6E716D; padding-top: 6px;'>{size_kb:.1f} KB</div>", unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"<div style='padding-top: 4px;'>{badge}</div>", unsafe_allow_html=True)
            with cols[4]:
                key_confirm = f"confirm_remove_{f.name}"
                if st.session_state.get(key_confirm):
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Yes", key=f"yes_rem_{idx}", type="primary", use_container_width=True):
                            from src.document_loader import safe_remove_document
                            success, msg, new_retriever, new_stats = safe_remove_document(
                                doc_path=f,
                                documents_dir=doc_dir,
                                vectorstore_dir=vstore_dir,
                            )
                            st.session_state[key_confirm] = False
                            if success:
                                st.session_state["retriever"] = new_retriever
                                st.session_state["indexing_stats"] = new_stats
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    with b2:
                        if st.button("No", key=f"no_rem_{idx}", type="secondary", use_container_width=True):
                            st.session_state[key_confirm] = False
                            st.rerun()
                else:
                    if st.button("Remove", key=f"btn_remove_{idx}", type="secondary", use_container_width=True):
                        st.session_state[key_confirm] = True
                        st.rerun()

            st.markdown("<div style='border-bottom: 1px solid #F0F0EC; margin-top: 2px; margin-bottom: 2px;'></div>", unsafe_allow_html=True)

    render_footer()
