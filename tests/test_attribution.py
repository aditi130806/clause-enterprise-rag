"""
Unit tests for src/attribution.py.
"""

from src.attribution import parse_and_validate_citations
from src.models import SourceReference


def test_parse_and_validate_valid_citations():
    source_map = {
        "S1": SourceReference(
            source_id="S1",
            document_id="doc1",
            document_name="remote.pdf",
            page_number=1,
            section="General",
            chunk_id="c1",
            excerpt="Remote work policy",
            retrieval_rank=1,
            retrieval_score=0.032,
        ),
        "S2": SourceReference(
            source_id="S2",
            document_id="doc2",
            document_name="security.pdf",
            page_number=2,
            section="Auth",
            chunk_id="c2",
            excerpt="MFA rules",
            retrieval_rank=2,
            retrieval_score=0.030,
        ),
    }

    text = "Employees can work remotely with manager approval [S1]. MFA is mandatory [S2]."
    sources, is_grounded, diag = parse_and_validate_citations(text, source_map)

    assert len(sources) == 2
    assert sources[0].source_id == "S1"
    assert sources[1].source_id == "S2"
    assert is_grounded is True
    assert diag["invalid_citations_count"] == 0


def test_parse_and_validate_invalid_citations_handled_gracefully():
    source_map = {
        "S1": SourceReference(
            source_id="S1",
            document_id="doc1",
            document_name="remote.pdf",
            page_number=1,
            section="General",
            chunk_id="c1",
            excerpt="Remote work policy",
            retrieval_rank=1,
            retrieval_score=0.032,
        )
    }

    # Model hallucinates [S99] tag
    text = "According to rules [S1] and additional requirements [S99]."
    sources, is_grounded, diag = parse_and_validate_citations(text, source_map)

    assert len(sources) == 1
    assert sources[0].source_id == "S1"
    assert is_grounded is True
    assert diag["invalid_citations_count"] == 1
    assert "S99" in diag["invalid_citations_list"]


def test_parse_and_validate_ungrounded_claims_without_citations():
    source_map = {
        "S1": SourceReference(
            source_id="S1",
            document_id="doc1",
            document_name="remote.pdf",
            page_number=1,
            section="General",
            chunk_id="c1",
            excerpt="Remote work policy",
            retrieval_rank=1,
            retrieval_score=0.032,
        )
    }

    # Claim without any inline citations
    text = "Employees get free company cars every two years."
    sources, is_grounded, diag = parse_and_validate_citations(text, source_map)

    assert len(sources) == 0
    assert is_grounded is False
