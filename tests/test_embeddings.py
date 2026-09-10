"""
Unit tests for src/embeddings.py.
"""

import pytest
import numpy as np
from src.embeddings import EmbeddingService, EmbeddingError
from src.models import DocumentChunk


def test_embedding_service_lazy_loading():
    service = EmbeddingService()
    assert service._model is None
    dim = service.dimension
    assert isinstance(dim, int)
    assert dim > 0
    assert service._model is not None


def test_encode_texts_output_shape_and_dtype():
    service = EmbeddingService()
    texts = ["Remote work policy", "Data encryption standard"]
    vecs = service.encode_texts(texts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.dtype == np.float32
    assert vecs.shape == (2, service.dimension)


def test_vector_l2_normalization():
    service = EmbeddingService()
    texts = ["Security guidelines for multi-factor authentication"]
    vecs = service.encode_texts(texts)

    norms = np.linalg.norm(vecs, axis=1)
    assert np.isclose(norms[0], 1.0, atol=1e-4)


def test_encode_query():
    service = EmbeddingService()
    query = "What is the probation period?"
    q_vec = service.encode_query(query)

    assert isinstance(q_vec, np.ndarray)
    assert q_vec.dtype == np.float32
    assert q_vec.shape == (service.dimension,)
    assert np.isclose(np.linalg.norm(q_vec), 1.0, atol=1e-4)


def test_empty_query_raises_error():
    service = EmbeddingService()
    with pytest.raises(EmbeddingError):
        service.encode_query("")

    with pytest.raises(EmbeddingError):
        service.encode_query("   ")


def test_encode_empty_text_list():
    service = EmbeddingService()
    vecs = service.encode_texts([])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (0, service.dimension)


def test_encode_chunks():
    service = EmbeddingService()
    chunks = [
        DocumentChunk(
            chunk_id="chk1",
            document_id="doc1",
            document_name="policy.pdf",
            page_number=1,
            text="Remote work guidelines for full time employees.",
            section="Overview",
            chunk_index=0,
        )
    ]
    vecs = service.encode_chunks(chunks)
    assert vecs.shape == (1, service.dimension)
    assert vecs.dtype == np.float32
