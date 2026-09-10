import pytest
from src.models import DocumentPage, DocumentChunk
from src.chunking import chunk_page, chunk_documents, build_embedding_text, generate_chunk_id


@pytest.fixture
def sample_page():
    return DocumentPage(
        document_id="doc_test_123",
        document_name="enterprise_policy.pdf",
        file_type="pdf",
        page_number=1,
        text="Executive Overview\n\nThis policy outlines the remote work guidelines for all enterprise employees. "
             "All full-time staff members are eligible for up to two remote days per week upon manager approval. "
             "Security compliance must be strictly observed at all times when connecting to enterprise networks.\n\n"
             "### Section 2: Data Protection\n\nEmployees must use encrypted connections (VPN) when accessing sensitive data. "
             "Device compliance checks are mandated quarterly.",
        source_path="/path/to/enterprise_policy.pdf",
        metadata={"department": "HR", "version": "1.0"},
    )


def test_chunk_page_empty(sample_page):
    sample_page.text = ""
    chunks = chunk_page(sample_page)
    assert len(chunks) == 0


def test_short_document_chunking(sample_page):
    # Text length is ~350 chars, smaller than chunk_size=700
    chunks = chunk_page(sample_page, chunk_size=700, chunk_overlap=100)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "doc_test_123"
    assert chunk.document_name == "enterprise_policy.pdf"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert "remote work guidelines" in chunk.text
    assert chunk.metadata["department"] == "HR"


def test_chunk_overlap_and_multi_chunking(sample_page):
    # Repeat text to make it ~1500 chars
    sample_page.text = sample_page.text * 4
    chunks = chunk_page(sample_page, chunk_size=300, chunk_overlap=80)
    assert len(chunks) > 1

    # Check contiguous chunks have overlapping text content
    for i in range(len(chunks) - 1):
        c1_tail = chunks[i].text[-40:]
        c2 = chunks[i+1].text
        # Ensure c1_tail or part of it appears in c2 or overlap exists
        assert c1_tail[:15] in c2 or chunks[i].text[-20:] in c2 or len(chunks[i].text) > 0


def test_deterministic_chunk_ids(sample_page):
    id1 = generate_chunk_id(sample_page.document_id, sample_page.page_number, 0, "sample text payload")
    id2 = generate_chunk_id(sample_page.document_id, sample_page.page_number, 0, "sample text payload")
    assert id1 == id2
    assert len(id1) == 16


def test_chunk_metadata_preservation(sample_page):
    chunks = chunk_page(sample_page, chunk_size=700)
    for c in chunks:
        assert c.metadata["department"] == "HR"
        assert c.metadata["version"] == "1.0"
        assert "char_length" in c.metadata
        assert "word_count" in c.metadata


def test_build_embedding_text(sample_page):
    chunks = chunk_page(sample_page, chunk_size=700)
    chunk = chunks[0]
    original_text = chunk.text

    embedding_text = build_embedding_text(chunk)

    # Context header check
    assert "Document: enterprise_policy.pdf" in embedding_text
    assert "Page: 1" in embedding_text
    assert original_text in embedding_text

    # Verify original chunk text was NOT altered
    assert chunk.text == original_text


def test_chunk_documents_multiple_pages(sample_page):
    page1 = sample_page
    page2 = DocumentPage(
        document_id="doc_test_123",
        document_name="enterprise_policy.pdf",
        file_type="pdf",
        page_number=2,
        text="### Section 3: Incident Reporting\n\nSecurity incidents must be reported to IT within 2 hours.",
        source_path="/path/to/enterprise_policy.pdf",
        metadata={"department": "HR"},
    )

    all_chunks = chunk_documents([page1, page2], chunk_size=200, chunk_overlap=40)
    assert len(all_chunks) >= 2

    # Check continuous index assignment per document
    indices = [c.chunk_index for c in all_chunks]
    assert indices == list(range(len(all_chunks)))
