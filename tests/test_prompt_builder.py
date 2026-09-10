"""
Unit tests for src/prompt_builder.py.
"""

from src.prompt_builder import build_context_block, build_grounded_prompt, REFUSAL_SENTENCE
from src.models import RetrievalResult, DocumentChunk


def test_build_context_block_formatting_and_mapping():
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        document_name="remote_policy.pdf",
        page_number=1,
        text="Remote work guidelines section 1.",
        section="Overview",
        chunk_index=0,
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc2",
        document_name="leave_policy.docx",
        page_number=None,
        text="Annual leave rules section 2.",
        section="Leave",
        chunk_index=0,
    )
    r1 = RetrievalResult(c1, "remote_policy.pdf", 1, "Overview", fusion_score=0.032, final_rank=1)
    r2 = RetrievalResult(c2, "leave_policy.docx", None, "Leave", fusion_score=0.030, final_rank=2)

    context_str, source_map = build_context_block([r1, r2])

    assert "[S1]" in context_str
    assert "[S2]" in context_str
    assert "remote_policy.pdf" in context_str
    assert "leave_policy.docx" in context_str
    assert "Page N/A" in context_str

    assert "S1" in source_map
    assert "S2" in source_map
    assert source_map["S1"].chunk_id == "c1"
    assert source_map["S1"].document_name == "remote_policy.pdf"
    assert source_map["S2"].chunk_id == "c2"


def test_build_grounded_prompt_instructions():
    question = "Can I work remotely on probation?"
    context_str = "[S1]\nDocument: remote.pdf\nPage 2 | Section: Probation\nContent:\nApproval required."

    prompt = build_grounded_prompt(question, context_str)

    assert "You are Clause" in prompt
    assert REFUSAL_SENTENCE in prompt
    assert "[S1]" in prompt
    assert question in prompt
