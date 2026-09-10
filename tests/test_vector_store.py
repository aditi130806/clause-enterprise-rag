"""
Unit tests for src/vector_store.py.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
from src.vector_store import FAISSVectorStore, VectorStoreError
from src.models import DocumentChunk


@pytest.fixture
def sample_chunks_and_embeddings():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="doc_1",
            document_name="security_policy.pdf",
            page_number=1,
            text="Multi-factor authentication is required.",
            section="Security",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_id="doc_2",
            document_name="leave_policy.docx",
            page_number=None,
            text="Annual leave entitlement is 20 days.",
            section="Leave",
            chunk_index=0,
        ),
    ]

    # Create synthetic 4-dimensional normalized vectors
    raw_vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    return chunks, raw_vecs


def test_faiss_add_and_search(sample_chunks_and_embeddings):
    chunks, vecs = sample_chunks_and_embeddings
    store = FAISSVectorStore(dimension=4)
    store.add_chunks(chunks, vecs)

    assert store.total_chunks == 2

    query_vec = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    # L2 normalize
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    top_chunk, score, rank = results[0]
    assert top_chunk.chunk_id == "chunk_001"
    assert rank == 1
    assert score > 0.8


def test_faiss_top_k_larger_than_index_size(sample_chunks_and_embeddings):
    chunks, vecs = sample_chunks_and_embeddings
    store = FAISSVectorStore(dimension=4)
    store.add_chunks(chunks, vecs)

    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_vec, top_k=10)

    assert len(results) == 2  # Gracefully capped at total chunks


def test_faiss_empty_index_search():
    store = FAISSVectorStore(dimension=4)
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query_vec, top_k=5)
    assert results == []


def test_faiss_dimension_mismatch_error(sample_chunks_and_embeddings):
    chunks, vecs = sample_chunks_and_embeddings
    store = FAISSVectorStore(dimension=8)  # Store dimension is 8

    with pytest.raises(VectorStoreError):
        store.add_chunks(chunks, vecs)  # vecs has dimension 4


def test_faiss_save_and_load_persistence(sample_chunks_and_embeddings):
    chunks, vecs = sample_chunks_and_embeddings
    store = FAISSVectorStore(dimension=4)
    store.add_chunks(chunks, vecs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir)
        store.save(save_path)

        loaded_store = FAISSVectorStore.load(save_path)
        assert loaded_store.dimension == 4
        assert loaded_store.total_chunks == 2
        assert loaded_store.chunks[0].chunk_id == "chunk_001"

        query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = loaded_store.search(query_vec, top_k=1)
        assert results[0][0].chunk_id == "chunk_001"
