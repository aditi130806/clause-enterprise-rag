from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.models import DocumentPage, DocumentChunk


def calculate_document_stats(
    pages: List[DocumentPage],
    chunks: Optional[List[DocumentChunk]] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive document ingestion and chunking statistics.

    Reported statistics:
    - total_documents: Number of unique documents processed
    - total_pages: Number of extracted document pages
    - total_chunks: Total number of generated chunks
    - chars_per_document: Total extracted characters per document name
    - empty_pages_count: Number of pages with zero extracted text
    - empty_pages_details: Specific list of empty pages (document_name, page_number)
    """
    unique_doc_ids = set()
    doc_name_map: Dict[str, str] = {}
    chars_per_doc: Dict[str, int] = defaultdict(int)
    empty_pages: List[Dict[str, Any]] = []

    for page in pages:
        unique_doc_ids.add(page.document_id)
        doc_name_map[page.document_id] = page.document_name
        chars_per_doc[page.document_name] += len(page.text)

        if len(page.text.strip()) == 0:
            empty_pages.append({
                "document_name": page.document_name,
                "document_id": page.document_id,
                "page_number": page.page_number
            })

    total_chunks = len(chunks) if chunks is not None else 0

    return {
        "total_documents": len(unique_doc_ids),
        "total_pages": len(pages),
        "total_chunks": total_chunks,
        "chars_per_document": dict(chars_per_doc),
        "empty_pages_count": len(empty_pages),
        "empty_pages_details": empty_pages,
    }
