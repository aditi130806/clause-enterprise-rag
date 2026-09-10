"""
Page 1: Ask Clause - Enterprise Grounded QA Workspace.
"""

from pathlib import Path
import streamlit as st

from src.models import RAGResponse
from src.rag_pipeline import answer_question_with_timeout, load_retrieval_engine, build_and_index_documents
from src.llm import GeminiLLMService
from src.config import DOCUMENTS_DIR, VECTORSTORE_DIR
from src.ui.shell import render_page_header, render_footer
from src.ui.components import (
    render_question_answer_card,
    render_evidence_panel,
    render_empty_state,
)


@st.cache_resource
def get_cached_retrieval_engine():
    """Globally cached HybridRetriever instance across all sessions."""
    vstore_path = Path(VECTORSTORE_DIR)
    if (vstore_path / "faiss_index.bin").exists():
        try:
            return load_retrieval_engine(vectorstore_dir=vstore_path)
        except Exception:
            return None
    else:
        doc_path = Path(DOCUMENTS_DIR)
        if doc_path.exists() and any(doc_path.iterdir()):
            try:
                retriever, _ = build_and_index_documents(
                    documents_dir=doc_path,
                    vectorstore_dir=vstore_path,
                )
                return retriever
            except Exception:
                return None
        return None


@st.cache_resource
def get_cached_llm_service() -> GeminiLLMService:
    """Globally cached GeminiLLMService instance across all sessions."""
    return GeminiLLMService()


def get_retriever():
    """Return retriever from session state if updated dynamically, else cached global singleton."""
    if "retriever" in st.session_state and st.session_state["retriever"] is not None:
        return st.session_state["retriever"]
    retriever = get_cached_retrieval_engine()
    st.session_state["retriever"] = retriever
    return retriever


def get_llm_service() -> GeminiLLMService:
    """Return LLM service from session state if updated dynamically, else cached global singleton."""
    if "llm_service" in st.session_state and st.session_state["llm_service"] is not None:
        return st.session_state["llm_service"]
    llm = get_cached_llm_service()
    st.session_state["llm_service"] = llm
    return llm


import html


def render_ask_page():
    """Render Ask Clause page."""
    render_page_header(
        title="Ask Clause",
        subtitle="Answers you can trust. Sources you can trace.",
        eyebrow="From documents to decisions.",
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "request_pending" not in st.session_state:
        st.session_state["request_pending"] = False
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = None
    if "current_response" not in st.session_state:
        st.session_state["current_response"] = None
    if "last_submitted_query" not in st.session_state:
        st.session_state["last_submitted_query"] = None

    from src.kb_state import get_knowledge_base_status
    kb_status = get_knowledge_base_status()
    retriever = kb_status.retriever or get_retriever()
    llm_service = get_llm_service()
    has_documents = kb_status.is_ready

    is_pending = st.session_state["request_pending"]

    # Search / Input Field (disabled during pending request)
    query_input = st.text_input(
        label="Ask a question",
        placeholder="Ask a question about your company documents...",
        label_visibility="collapsed",
        key="main_query_input",
        disabled=is_pending,
    )

    # Sample Question Chips
    st.markdown("<div style='font-size: 12px; color: #6E716D; margin-top: 4px; margin-bottom: 6px;'>Suggested queries:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(3)
    sample_queries = [
        "What percentage of healthcare insurance premiums does the company cover for employees and dependents?",
        "What receipts are required for expenses over $25, and when must I submit the claim?",
        "Can employees work from home during probation, and what conditions apply?",
    ]

    selected_chip = None
    for idx, sq in enumerate(sample_queries):
        with chip_cols[idx]:
            st.markdown("<div class='chip-btn'>", unsafe_allow_html=True)
            if st.button(sq, key=f"chip_query_{idx}", use_container_width=True, disabled=is_pending):
                selected_chip = sq
            st.markdown("</div>", unsafe_allow_html=True)

    submitted_query = selected_chip or query_input

    # Step 1: Detect newly submitted question when not pending
    if submitted_query and not is_pending:
        if submitted_query != st.session_state["last_submitted_query"]:
            # IMMEDIATELY clear previous current answer and evidence!
            st.session_state["current_response"] = None
            st.session_state["request_pending"] = True
            st.session_state["pending_query"] = submitted_query
            st.session_state["last_submitted_query"] = submitted_query
            st.rerun()

    # Step 2: Render loading state OR current answer/evidence
    if st.session_state["request_pending"]:
        active_query = st.session_state.get("pending_query", "your question")

        # Render loading grid immediately
        left_col, right_col = st.columns([2, 1], gap="large")
        with left_col:
            st.markdown(
                f'<div class="clause-card" style="margin-bottom: 12px;"><div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;"><span style="font-size: 12px; font-weight: 600; color: #6E716D; text-transform: uppercase;">Your question</span></div><div style="font-size: 16px; font-weight: 600; color: #1C1E1C;">{html.escape(active_query)}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="clause-card" style="padding: 24px;"><div style="font-size: 14px; font-weight: 600; color: #1C1E1C; margin-bottom: 12px;">Searching documents & generating grounded response...</div></div>',
                unsafe_allow_html=True,
            )

        with right_col:
            st.markdown('<div style="font-size: 15px; font-weight: 700; color: #1C1E1C; margin-bottom: 12px;">Evidence</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="clause-card" style="padding: 16px;"><div style="font-size: 13px; color: #6E716D;">Searching knowledge base for evidence...</div></div>',
                unsafe_allow_html=True,
            )

        # Execute long-running RAG request
        if not has_documents:
            st.warning("Upload documents to create your knowledge base.")
            st.session_state["request_pending"] = False
            st.rerun()
        else:
            with st.spinner("Searching documents & generating grounded response..."):
                try:
                    response = answer_question_with_timeout(
                        question=active_query,
                        retriever=retriever,
                        llm_service=llm_service,
                        timeout_seconds=30.0,
                    )
                except Exception as exc:
                    llm_name = llm_service.model_name if llm_service else "gemini-3.8-flash"
                    response = RAGResponse(
                        question=active_query,
                        answer="Clause could not complete the request in time. Please try again.",
                        sources=[],
                        retrieved_results=[],
                        grounded=False,
                        refused=True,
                        confidence=0.0,
                        confidence_label="Insufficient",
                        evidence_status="insufficient",
                        conflict_detected=False,
                        retrieval_corrected=False,
                        evidence_reasons=[f"Execution exception: {exc}"],
                        generation_model=llm_name,
                        retrieval_mode="hybrid",
                        metadata={"refusal_stage": "uncaught_exception", "error": str(exc)},
                    )

            from src.live_analytics import log_live_query_event
            log_live_query_event(response)

            st.session_state["current_response"] = response
            st.session_state["chat_history"].insert(0, response)
            st.session_state["request_pending"] = False
            st.rerun()

    elif st.session_state.get("current_response"):
        resp: RAGResponse = st.session_state["current_response"]
        left_col, right_col = st.columns([2, 1], gap="large")
        with left_col:
            render_question_answer_card(resp)
        with right_col:
            render_evidence_panel(resp)
    elif st.session_state.get("chat_history"):
        latest_response: RAGResponse = st.session_state["chat_history"][0]
        left_col, right_col = st.columns([2, 1], gap="large")
        with left_col:
            render_question_answer_card(latest_response)
        with right_col:
            render_evidence_panel(latest_response)
    elif not has_documents:
        render_empty_state(
            title="Upload documents to create your knowledge base.",
            subtitle="Go to the Documents page to add PDF or DOCX policy files.",
        )
    else:
        render_empty_state(
            title="Ready to answer questions from your documents.",
            subtitle="Ask a question or choose a suggested query above.",
        )

    render_footer()

