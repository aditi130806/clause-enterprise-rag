import re


def clean_text(raw_text: str) -> str:
    """
    Cleans raw extracted text while preserving document structure, formatting, and semantic meaning.

    Operations performed:
    - Removes null bytes and non-printable control characters (except newline, carriage return, tab).
    - Normalizes inline repeated spaces and tabs to single spaces without collapsing line breaks.
    - Fixes hyphenated line breaks (e.g. "com-\n pany" -> "company").
    - Normalizes excessive blank lines (more than 2 consecutive newlines reduced to 2).
    - Strips leading and trailing whitespace from each line.

    Preserves:
    - Case sensitivity (no lowercasing)
    - All punctuation
    - Paragraph structure and section headings
    """
    if not raw_text:
        return ""

    # Remove null characters and non-printable control chars except \n, \r, \t
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_text)

    # Rejoin words broken across line breaks with a hyphen (e.g. "organiza-\ntion" -> "organization")
    cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", cleaned)

    # Normalize horizontal whitespace (spaces and tabs) on each line
    lines = cleaned.splitlines()
    cleaned_lines = []
    for line in lines:
        # Collapse multiple horizontal spaces/tabs into a single space
        line_clean = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line_clean)

    cleaned = "\n".join(cleaned_lines)

    # Collapse 3 or more consecutive newlines into 2 (preserving paragraph spacing)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
