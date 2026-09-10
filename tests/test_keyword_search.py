"""
Unit tests for src/keyword_search.py.
"""

import tempfile
from pathlib import Path
from src.keyword_search import BM25SearchIndex, default_bm25_tokenizer
from src.models import DocumentChunk


def test_default_bm25_tokenizer():
    tokens = default_bm25_tokenizer("Policy HR-RW-017: Remote Work Rules & MFA!")
    assert "hr" in tokens
    assert "rw" in tokens
    assert "017" in tokens
    assert "mfa" in tokens
    assert "policy" in tokens
    assert "!" not in tokens


def test_bm25_exact_policy_id_retrieval():
    chunks = [
        DocumentChunk(
            chunk_id="chk_remote",
            document_id="doc1",
            document_name="remote.pdf",
            page_number=1,
            text="Remote Work Policy HR-RW-017 states employees may work remotely.",
            section="General",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chk_leave",
            document_id="doc2",
            document_name="leave.pdf",
            page_number=1,
            text="Annual leave entitlement is 20 days per year.",
            section="General",
            chunk_index=0,
        ),
    ]

    index = BM25SearchIndex()
    index.build_index(chunks)

    results = index.search("HR-RW-017", top_k=2)

    assert len(results) >= 1
    top_chunk, score, rank = results[0]
    assert top_chunk.chunk_id == "chk_remote"
    assert rank == 1
    assert score > 0.0


def test_bm25_empty_query_and_corpus():
    index = BM25SearchIndex()
    assert index.search("Remote", top_k=5) == []

    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            page_number=1,
            text="Text chunk",
            section=None,
            chunk_index=0,
        )
    ]
    index.build_index(chunks)
    assert index.search("", top_k=5) == []
    assert index.search("   ", top_k=5) == []


def test_bm25_save_and_load_persistence():
    chunks = [
        DocumentChunk(
            chunk_id="c_mfa",
            document_id="d_sec",
            document_name="security.pdf",
            page_number=1,
            text="Multi-factor authentication identifier SEC-AUTH-004.",
            section="Auth",
            chunk_index=0,
        )
    ]
    index = BM25SearchIndex()
    index.build_index(chunks)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir)
        index.save(save_path)

        loaded_index = BM25SearchIndex.load(save_path)
        assert loaded_index.total_chunks == 1
        results = loaded_index.search("SEC-AUTH-004", top_k=1)
        assert len(results) == 1
        assert results[0][0].chunk_id == "c_mfa"
