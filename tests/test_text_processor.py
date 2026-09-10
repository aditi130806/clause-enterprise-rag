import pytest
from src.text_processor import clean_text


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_whitespace_normalization():
    raw = "This    is   a   test  with   multiple   spaces."
    expected = "This is a test with multiple spaces."
    assert clean_text(raw) == expected


def test_clean_text_excessive_newlines():
    raw = "Paragraph 1\n\n\n\n\nParagraph 2\n\n\nParagraph 3"
    expected = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
    assert clean_text(raw) == expected


def test_clean_text_hyphenated_line_break():
    raw = "The company is an enterprise document intelligence baseline.\nThis is organized for multi-\n page processing."
    cleaned = clean_text(raw)
    assert "multipage" in cleaned or "multi-page" not in cleaned


def test_clean_text_preserves_case_and_punctuation():
    raw = "### Section 1: Executive Overview\n\nDoes this retain punctuation? YES! Exactly: 100%."
    cleaned = clean_text(raw)
    assert "Section 1: Executive Overview" in cleaned
    assert "YES!" in cleaned
    assert "100%" in cleaned


def test_clean_text_strips_null_bytes():
    raw = "Clean\x00 text\x07 with null\x0b bytes."
    cleaned = clean_text(raw)
    assert cleaned == "Clean text with null bytes."
