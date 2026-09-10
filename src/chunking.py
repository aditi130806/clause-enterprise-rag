import hashlib
import re
from typing import List, Optional, Dict, Any, Tuple

from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.models import DocumentPage, DocumentChunk


def find_smart_split_point(text: str, target_size: int) -> int:
    """
    Finds a natural boundary (paragraph break, newline, sentence end, or space)
    near or before target_size. Falls back to target_size if no boundary is found.
    """
    if len(text) <= target_size:
        return len(text)

    # Search window: target_size - 100 to target_size
    min_search = max(50, target_size - 120)
    substring = text[min_search:target_size]

    # 1. Paragraph break (\n\n)
    para_break = substring.rfind("\n\n")
    if para_break != -1:
        return min_search + para_break + 2

    # 2. Line break (\n)
    line_break = substring.rfind("\n")
    if line_break != -1:
        return min_search + line_break + 1

    # 3. Sentence end (.!?) followed by space
    sentence_matches = [m.end() for m in re.finditer(r"[.!?]\s", substring)]
    if sentence_matches:
        return min_search + sentence_matches[-1]

    # 4. Word boundary (space)
    space_break = substring.rfind(" ")
    if space_break != -1:
        return min_search + space_break + 1

    # Hard cut if no natural boundary found
    return target_size


def generate_chunk_id(doc_id: str, page_number: Optional[int], chunk_index: int, text: str) -> str:
    """Generates a deterministic unique chunk ID based on document ID, page, index, and text payload."""
    raw_str = f"{doc_id}_p{page_number}_{chunk_index}_{text[:50]}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def find_all_page_headings(text: str) -> List[Tuple[int, str]]:
    """Scan page text for all section headings and return a sorted list of (start_char_offset, heading_title)."""
    headings = []
    # 1. Markdown headings (#, ##, ###, ####)
    for m in re.finditer(r"^(#{1,4})\s*(.+)$", text, re.MULTILINE):
        title = m.group(2).strip()
        headings.append((m.start(), title))

    # 2. Numbered headings (e.g. "4. Unexcused Absences & Tardiness" or "Section 3: ...")
    for m in re.finditer(r"^(?:Section\s+\d+:?|\d+\.\s+)([A-Z][A-Za-z0-9\s&,()/:-]{3,80})$", text, re.MULTILINE):
        title = m.group(0).strip()
        if not any(abs(h[0] - m.start()) < 5 for h in headings):
            headings.append((m.start(), title))

    # 3. Bold line headings (e.g. "**Unexcused Absences**")
    for m in re.finditer(r"^\*\*\s*([A-Za-z0-9\s&,()/:-]{3,80})\s*\*\*", text, re.MULTILINE):
        title = m.group(1).strip()
        if not any(abs(h[0] - m.start()) < 5 for h in headings):
            headings.append((m.start(), title))

    headings.sort(key=lambda x: x[0])
    return headings


def get_section_for_offset(headings: List[Tuple[int, str]], offset: int, default_section: Optional[str] = None) -> Optional[str]:
    """Return the section heading immediately preceding or containing the given character offset."""
    active = default_section
    for h_offset, h_title in headings:
        if h_offset <= offset + 50:
            active = h_title
        else:
            break
    return active


def chunk_page(
    page: DocumentPage,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_chunk_index: int = 0,
) -> List[DocumentChunk]:
    """
    Splits a DocumentPage into context-aware text chunks.

    Guarantees:
    - Breaks at natural sentence/paragraph/word boundaries.
    - Preserves metadata, page number, document name, and document ID.
    - Accurately tracks dynamic section headers per chunk offset.
    - Generates deterministic chunk IDs.
    - Never returns empty chunks.
    """
    text = page.text.strip()
    if not text:
        return []

    chunks: List[DocumentChunk] = []
    start = 0
    text_len = len(text)
    current_index = start_chunk_index
    page_default_section = page.metadata.get("headings", [None])[0] if page.metadata.get("headings") else None
    page_headings = find_all_page_headings(text)

    while start < text_len:
        # Determine current window end
        end = start + chunk_size

        # Find if a new section heading starts inside this window (after start)
        next_heading_offset = None
        for h_off, _ in page_headings:
            if h_off > start + 30:
                next_heading_offset = h_off
                break

        min_split_threshold = max(120, chunk_size // 2)

        if end >= text_len:
            actual_end = text_len
            if next_heading_offset and (next_heading_offset - start) >= min_split_threshold and next_heading_offset < actual_end:
                actual_end = next_heading_offset
                is_last_chunk = False
            else:
                is_last_chunk = True
            chunk_text = text[start:actual_end].strip()
            split_at = actual_end - start
        else:
            split_at = find_smart_split_point(text[start:end+50], chunk_size)
            actual_end = start + split_at
            if next_heading_offset and (next_heading_offset - start) >= min_split_threshold and next_heading_offset < actual_end:
                actual_end = next_heading_offset
                split_at = actual_end - start
            chunk_text = text[start:actual_end].strip()
            is_last_chunk = (actual_end >= text_len)

        if chunk_text:
            active_section = get_section_for_offset(page_headings, start, default_section=page_default_section)
            chunk_id = generate_chunk_id(page.document_id, page.page_number, current_index, chunk_text)

            chunk_metadata = dict(page.metadata)
            chunk_metadata.update({
                "char_length": len(chunk_text),
                "word_count": len(chunk_text.split()),
            })

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    text=chunk_text,
                    section=active_section,
                    chunk_index=current_index,
                    metadata=chunk_metadata,
                )
            )
            current_index += 1

        if is_last_chunk:
            break

        # Advance window by step size
        if next_heading_offset and next_heading_offset == actual_end:
            start = next_heading_offset
        else:
            step = split_at - chunk_overlap
            if step <= 0:
                step = max(1, chunk_size // 2)
            start += step

        if start >= text_len:
            break

    return chunks


def chunk_documents(
    pages: List[DocumentPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """Processes multiple DocumentPages into chunks with document-scoped chunk indexing."""
    all_chunks: List[DocumentChunk] = []
    
    # Group by document_id to track chunk indices continuously per document
    doc_chunks_count: Dict[str, int] = {}

    for page in pages:
        doc_id = page.document_id
        start_idx = doc_chunks_count.get(doc_id, 0)
        
        page_chunks = chunk_page(
            page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_chunk_index=start_idx,
        )
        
        doc_chunks_count[doc_id] = start_idx + len(page_chunks)
        all_chunks.extend(page_chunks)

    return all_chunks


def build_embedding_text(chunk: DocumentChunk) -> str:
    """
    Constructs contextualized text for embedding generation.

    The original `chunk.text` remains unaltered for source citations.
    This contextualized representation injects document title, section, and page number
    to optimize dense vector retrieval precision.
    """
    section_str = chunk.section if chunk.section else "N/A"
    page_str = str(chunk.page_number) if chunk.page_number is not None else "N/A"

    return (
        f"Document: {chunk.document_name}\n"
        f"Section: {section_str}\n"
        f"Page: {page_str}\n\n"
        f"{chunk.text}"
    )
