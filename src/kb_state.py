"""
Unified Single Source of Truth for CLAUSE Knowledge Base Status.
Determines document count, chunk count, readiness state, and retriever status.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any
import streamlit as st

from src.config import DOCUMENTS_DIR, VECTORSTORE_DIR
from src.rag_pipeline import load_retrieval_engine, build_and_index_documents


@dataclass
class KnowledgeBaseStatus:
    doc_count: int
    chunk_count: int
    is_ready: bool
    status_text: str  # "Knowledge base ready" or "Knowledge base not indexed" or "No documents uploaded"
    retriever: Optional[Any]


def get_knowledge_base_status() -> KnowledgeBaseStatus:
    """
    Computes active knowledge base status, reading real doc files and vector store index.
    Caches retriever in st.session_state for seamless access.
    """
    doc_dir = Path(DOCUMENTS_DIR)
    doc_files = []
    if doc_dir.exists():
        doc_files = [
            f for f in doc_dir.glob("*")
            if f.is_file() and not f.name.startswith("~$") and f.suffix.lower() in [".pdf", ".docx"]
        ]
    doc_count = len(doc_files)

    # Check session state retriever first
    retriever = st.session_state.get("retriever")

    # If missing, attempt to load cached vector store from disk
    if retriever is None:
        vstore_path = Path(VECTORSTORE_DIR)
        if (vstore_path / "faiss_index.bin").exists():
            try:
                retriever = load_retrieval_engine(vectorstore_dir=vstore_path)
                st.session_state["retriever"] = retriever
            except Exception:
                retriever = None
        elif doc_count > 0:
            try:
                retriever, stats = build_and_index_documents(
                    documents_dir=doc_dir,
                    vectorstore_dir=vstore_path,
                )
                st.session_state["retriever"] = retriever
                st.session_state["indexing_stats"] = stats
            except Exception:
                retriever = None

    chunk_count = 0
    if retriever is not None and hasattr(retriever, "vector_store") and retriever.vector_store is not None:
        chunk_count = getattr(retriever.vector_store, "total_chunks", 0)
    elif "indexing_stats" in st.session_state:
        chunk_count = st.session_state["indexing_stats"].get("chunks_generated", 0)

    is_ready = doc_count > 0 and chunk_count > 0 and retriever is not None

    if is_ready:
        status_text = "Knowledge base ready"
    elif doc_count > 0:
        status_text = "Knowledge base not indexed"
    else:
        status_text = "No documents uploaded"

    return KnowledgeBaseStatus(
        doc_count=doc_count,
        chunk_count=chunk_count,
        is_ready=is_ready,
        status_text=status_text,
        retriever=retriever,
    )


def compute_kb_fingerprint() -> str:
    """
    Computes a deterministic hash fingerprint representing the current active knowledge base state.
    Derived from: sorted document filenames, file sizes, and total chunk count.
    """
    import hashlib
    doc_dir = Path(DOCUMENTS_DIR)
    doc_files = []
    if doc_dir.exists():
        doc_files = sorted([
            f for f in doc_dir.glob("*")
            if f.is_file() and not f.name.startswith("~$") and f.suffix.lower() in [".pdf", ".docx"]
        ], key=lambda x: x.name.lower())

    file_info_strs = []
    for f in doc_files:
        try:
            size = f.stat().st_size
        except Exception:
            size = 0
        file_info_strs.append(f"{f.name}:{size}")

    kb_status = get_knowledge_base_status()
    raw_str = "|".join(file_info_strs) + f"|chunks:{kb_status.chunk_count}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]

