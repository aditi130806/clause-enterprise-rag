"""
Unit tests for src/retriever.py.
"""

import pytest
import numpy as np
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.keyword_search import BM25SearchIndex
from src.retriever import HybridRetriever
from src.models import DocumentChunk


@pytest.fixture
def sample_retrieval_engine():
    c1 = DocumentChunk(
        chunk_id="chunk_remote",
        document_id="doc_remote",
        document_name="remote_work_policy.pdf",
        page_number=1,
        text="Remote work up to two days each week. Probationary employees require approval. Policy HR-RW-017.",
        section="Overview",
        chunk_index=0,
    )
    c2 = DocumentChunk(
        chunk_id="chunk_sec",
        document_id="doc_sec",
        document_name="security_policy.pdf",
        page_number=1,
        text="Multi-factor authentication (MFA) requirements SEC-AUTH-004.",
        section="Auth",
        chunk_index=0,
    )
    chunks = [c1, c2]

    embedding_service = EmbeddingService()
    dim = embedding_service.dimension

    vstore = FAISSVectorStore(dimension=dim)
    vecs = embedding_service.encode_chunks(chunks)
    vstore.add_chunks(chunks, vecs)

    bm25 = BM25SearchIndex()
    bm25.build_index(chunks)

    retriever = HybridRetriever(
        vector_store=vstore,
        bm25_index=bm25,
        embedding_service=embedding_service,
        rrf_k=60,
    )

    return retriever, c1, c2


def test_hybrid_retrieval_modes(sample_retrieval_engine):
    retriever, c1, c2 = sample_retrieval_engine

    # 1. Hybrid Mode
    hybrid_res = retriever.retrieve("work from home while on probation", mode="hybrid", top_k=2)
    assert len(hybrid_res) > 0
    assert hybrid_res[0].chunk.chunk_id == "chunk_remote"
    assert "dense" in hybrid_res[0].retrieval_methods or "keyword" in hybrid_res[0].retrieval_methods
    assert hybrid_res[0].fusion_score > 0.0

    # 2. Dense Mode
    dense_res = retriever.retrieve("work from home", mode="dense", top_k=2)
    assert len(dense_res) > 0
    assert dense_res[0].retrieval_methods == ["dense"]

    # 3. Keyword Mode
    kw_res = retriever.retrieve("HR-RW-017", mode="keyword", top_k=2)
    assert len(kw_res) > 0
    assert kw_res[0].chunk.chunk_id == "chunk_remote"
    assert kw_res[0].retrieval_methods == ["keyword"]


def test_rrf_deduplication_and_score_preservation(sample_retrieval_engine):
    retriever, c1, c2 = sample_retrieval_engine
    results = retriever.retrieve("HR-RW-017 remote work", mode="hybrid", top_k=2)

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    top_r = results[0]
    assert top_r.final_rank == 1
    assert top_r.document_name == "remote_work_policy.pdf"
    assert top_r.page_number == 1
    assert top_r.section == "Overview"


def test_metadata_filtering(sample_retrieval_engine):
    retriever, c1, c2 = sample_retrieval_engine

    filtered_res = retriever.retrieve(
        "MFA authentication",
        mode="hybrid",
        top_k=2,
        filters={"document_name": "security_policy.pdf"},
    )

    assert len(filtered_res) == 1
    assert filtered_res[0].chunk.chunk_id == "chunk_sec"
    assert filtered_res[0].document_name == "security_policy.pdf"
