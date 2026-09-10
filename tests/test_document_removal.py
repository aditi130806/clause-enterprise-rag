"""
Tests for Document Removal, Knowledge Base Clearing, Index Rebuilding, and Temp File Filtering.
"""

import shutil
from pathlib import Path
import pytest
import docx

from src.document_loader import load_documents_from_directory, load_document
from src.rag_pipeline import build_and_index_documents


@pytest.fixture
def temp_workspace(tmp_path):
    """Creates a temporary workspace with 3 synthetic test documents."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    vstore_dir = tmp_path / "vectorstore"
    vstore_dir.mkdir()

    # Doc 1: Remote Work
    doc1 = docx.Document()
    doc1.add_heading("Remote Work Policy", level=0)
    doc1.add_paragraph("Employees may work remotely under HR-RW-017 with manager approval.")
    doc1.save(str(docs_dir / "Remote_Work.docx"))

    # Doc 2: Travel Expenses
    doc2 = docx.Document()
    doc2.add_heading("Travel Expense Policy", level=0)
    doc2.add_paragraph("Itemized receipts are required for reimbursable expenses exceeding $25.")
    doc2.save(str(docs_dir / "Travel_Expenses.docx"))

    # Doc 3: Temporary MS Word lock file
    temp_doc = docx.Document()
    temp_doc.add_paragraph("Temporary file data")
    temp_doc.save(str(docs_dir / "~$Remote_Work.docx"))

    return docs_dir, vstore_dir


def test_temp_file_ignored(temp_workspace):
    """Test that MS Word temporary files beginning with ~$ are ignored."""
    docs_dir, _ = temp_workspace

    # 1. Direct directory loading
    pages = load_documents_from_directory(docs_dir)
    doc_names = [p.document_name for p in pages]

    assert "~$Remote_Work.docx" not in doc_names
    assert "Remote_Work.docx" in doc_names
    assert "Travel_Expenses.docx" in doc_names

    # 2. Direct single file load on ~$ file returns empty list
    temp_pages = load_document(docs_dir / "~$Remote_Work.docx")
    assert temp_pages == []


def test_remove_one_document(temp_workspace):
    """Test removing one document updates counts and rebuilds clean FAISS/BM25 indices."""
    docs_dir, vstore_dir = temp_workspace

    # Initial indexing (2 real docs + 1 ignored temp file)
    retriever, stats = build_and_index_documents(documents_dir=docs_dir, vectorstore_dir=vstore_dir)
    initial_chunks = retriever.vector_store.total_chunks
    assert stats["documents_indexed"] == 2
    assert initial_chunks > 0

    # Verify initial retrieval finds travel expense
    res_before = retriever.retrieve("receipts required for expenses")
    assert len(res_before) > 0
    assert any("Travel_Expenses" in r.document_name for r in res_before)

    # Remove Travel_Expenses.docx
    (docs_dir / "Travel_Expenses.docx").unlink()

    # Rebuild index
    new_retriever, new_stats = build_and_index_documents(documents_dir=docs_dir, vectorstore_dir=vstore_dir)

    assert new_stats["documents_indexed"] == 1
    assert new_retriever.vector_store.total_chunks < initial_chunks

    # Verify stale chunks from removed doc are completely gone
    for chunk in new_retriever.vector_store.chunks:
        assert chunk.document_name != "Travel_Expenses.docx"

    # Search for exclusive keyword from removed doc returns zero results
    res_after = new_retriever.retrieve("reimbursable expenses exceeding $25")
    assert not any("Travel_Expenses" in r.document_name for r in res_after)


def test_clear_all_documents(temp_workspace):
    """Test clearing knowledge base resets FAISS/BM25 indices to zero chunks."""
    docs_dir, vstore_dir = temp_workspace

    # Initial indexing
    retriever, stats = build_and_index_documents(documents_dir=docs_dir, vectorstore_dir=vstore_dir)
    assert retriever.vector_store.total_chunks > 0

    # Clear all active documents from directory
    for f in docs_dir.glob("*"):
        if f.is_file():
            f.unlink()

    # Rebuild empty index
    empty_retriever, empty_stats = build_and_index_documents(documents_dir=docs_dir, vectorstore_dir=vstore_dir)

    assert empty_stats["documents_indexed"] == 0
    assert empty_stats["chunks_generated"] == 0
    assert empty_retriever.vector_store.total_chunks == 0
    assert empty_retriever.bm25_index.total_chunks == 0

    # Searching empty index returns empty list without error
    empty_res = empty_retriever.retrieve("work remotely")
    assert empty_res == []


def test_safe_remove_document_success(temp_workspace):
    """Test safe_remove_document removes document, rebuilds index, and updates counts."""
    from src.document_loader import safe_remove_document
    docs_dir, vstore_dir = temp_workspace

    # Pre-index
    build_and_index_documents(documents_dir=docs_dir, vectorstore_dir=vstore_dir)
    target_doc = docs_dir / "Travel_Expenses.docx"

    success, msg, new_retriever, new_stats = safe_remove_document(target_doc, docs_dir, vstore_dir)

    assert success is True
    assert "Successfully removed" in msg
    assert not target_doc.exists()
    assert new_stats["documents_indexed"] == 1
    assert not any(c.document_name == "Travel_Expenses.docx" for c in new_retriever.vector_store.chunks)


def test_safe_remove_document_permission_error_handled(temp_workspace, monkeypatch):
    """Test safe_remove_document handles PermissionError / WinError 32 safely without tracebacks."""
    from src.document_loader import safe_remove_document
    docs_dir, vstore_dir = temp_workspace

    build_and_index_documents(documents_dir=docs_dir, vectorstore_dir=vstore_dir)
    target_doc = docs_dir / "Travel_Expenses.docx"

    def mock_unlink(self):
        raise PermissionError(32, "The process cannot access the file because it is being used by another process")

    monkeypatch.setattr(Path, "unlink", mock_unlink)

    success, msg, new_retriever, new_stats = safe_remove_document(target_doc, docs_dir, vstore_dir)

    assert success is False
    assert "currently open or being used by another application" in msg
    assert "Traceback" not in msg
    assert target_doc.exists()
    assert new_retriever is None

