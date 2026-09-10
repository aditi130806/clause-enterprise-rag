import hashlib
from pathlib import Path
from typing import List, Union, Dict, Any, Optional, Tuple

import pypdf
from docx import Document as DocxDocument

from src.models import DocumentPage
from src.text_processor import clean_text


class DocumentLoadingError(Exception):
    """Base exception for document loading failures."""
    pass


class UnsupportedFormatError(DocumentLoadingError):
    """Raised when an unsupported file format is encountered."""
    pass


class CorruptDocumentError(DocumentLoadingError):
    """Raised when a document cannot be parsed due to corruption or invalid syntax."""
    pass


def generate_document_id(file_path: Path) -> str:
    """
    Generates a stable, deterministic SHA-256 hash digest based on file content.
    Returns the first 16 characters of the hex digest.
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    except Exception as e:
        # Fallback to path + size if file reading fails
        fallback_str = f"{file_path.name}_{file_path.stat().st_size if file_path.exists() else 0}"
        return hashlib.sha256(fallback_str.encode("utf-8")).hexdigest()[:16]


def load_pdf(file_path: Union[str, Path]) -> List[DocumentPage]:
    """
    Extracts text page by page from a PDF document using pypdf.

    Limitations:
    - Does NOT perform OCR. Scanned or image-only PDFs will return empty text strings.
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoadingError(f"File not found: {path}")

    doc_id = generate_document_id(path)
    pages: List[DocumentPage] = []

    try:
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                page_number = i + 1
                try:
                    extracted_raw = page.extract_text() or ""
                except Exception as pe:
                    extracted_raw = ""
                
                cleaned = clean_text(extracted_raw)

                metadata: Dict[str, Any] = {
                    "total_pages": num_pages,
                    "is_empty": len(cleaned) == 0,
                    "has_ocr": False,
                }

                pages.append(
                    DocumentPage(
                        document_id=doc_id,
                        document_name=path.name,
                        file_type="pdf",
                        page_number=page_number,
                        text=cleaned,
                        source_path=str(path.resolve()),
                        metadata=metadata,
                    )
                )

    except Exception as e:
        if isinstance(e, DocumentLoadingError):
            raise
        raise CorruptDocumentError(f"Failed to read PDF file '{path.name}': {str(e)}") from e

    return pages


def load_docx(file_path: Union[str, Path]) -> List[DocumentPage]:
    """
    Extracts paragraphs and headings from a Microsoft Word (.docx) document using python-docx.

    Note on page numbers:
    - Word documents do not have physical page numbers stored in the XML structure without layout rendering.
    - page_number is explicitly set to None to represent this honestly.
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoadingError(f"File not found: {path}")

    doc_id = generate_document_id(path)

    try:
        with open(path, "rb") as f:
            doc = DocxDocument(f)
            extracted_paragraphs: List[str] = []
            heading_structure: List[str] = []

            for p in doc.paragraphs:
                text = p.text.strip()
                if not text:
                    continue

                # Capture heading styles where available
                if p.style and p.style.name and p.style.name.startswith("Heading"):
                    heading_structure.append(text)
                    extracted_paragraphs.append(f"### {text}")
                else:
                    extracted_paragraphs.append(text)

        full_text = clean_text("\n\n".join(extracted_paragraphs))

        metadata: Dict[str, Any] = {
            "total_pages": None,
            "is_empty": len(full_text) == 0,
            "headings": heading_structure,
        }

        # DOCX is treated as a unified document page with page_number=None
        return [
            DocumentPage(
                document_id=doc_id,
                document_name=path.name,
                file_type="docx",
                page_number=None,
                text=full_text,
                source_path=str(path.resolve()),
                metadata=metadata,
            )
        ]

    except Exception as e:
        if isinstance(e, DocumentLoadingError):
            raise
        raise CorruptDocumentError(f"Failed to read DOCX file '{path.name}': {str(e)}") from e


def safe_remove_document(doc_path: Path, documents_dir: Path, vectorstore_dir: Path) -> Tuple[bool, str, Optional[Any], Optional[Dict[str, Any]]]:
    """
    Safely removes a document from the documents directory and rebuilds the knowledge base index.
    Handles PermissionError / WinError 32 safely without crashing or exposing Python tracebacks.

    Returns:
        Tuple of (success_boolean, user_message_string, new_retriever, new_stats)
    """
    if not doc_path.exists():
        return False, "Document file does not exist.", None, None

    try:
        doc_path.unlink()
    except (PermissionError, OSError) as pe:
        import logging
        logging.error(f"PermissionError removing file {doc_path.name}: {pe}", exc_info=True)
        return (
            False,
            "Clause could not remove this document because it is currently open or being used by another application. Close the file and try again.",
            None,
            None,
        )
    except Exception as ex:
        import logging
        logging.error(f"Unexpected error removing file {doc_path.name}: {ex}", exc_info=True)
        return False, "An error occurred while removing the document. Please try again.", None, None

    try:
        from src.rag_pipeline import build_and_index_documents
        new_retriever, new_stats = build_and_index_documents(
            documents_dir=documents_dir,
            vectorstore_dir=vectorstore_dir,
        )
        return True, f"Successfully removed '{doc_path.name}'.", new_retriever, new_stats
    except Exception as ex:
        import logging
        logging.error(f"Error rebuilding index after removing {doc_path.name}: {ex}", exc_info=True)
        return False, "Document was removed but index rebuild failed.", None, None



def load_document(file_path: Union[str, Path]) -> List[DocumentPage]:
    """Loads a document (PDF or DOCX) based on file extension, ignoring MS Word temporary files."""
    path = Path(file_path)
    if path.name.startswith("~$"):
        return []

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".docx":
        return load_docx(path)
    else:
        raise UnsupportedFormatError(f"Unsupported file format '{suffix}' for file '{path.name}'. Only .pdf and .docx are supported.")


def load_documents_from_directory(directory_path: Union[str, Path]) -> List[DocumentPage]:
    """Loads all supported PDF and DOCX files from a target directory, skipping temporary MS Word files (~$)."""
    path = Path(directory_path)
    if not path.is_dir():
        raise DocumentLoadingError(f"Directory not found: {path}")

    all_pages: List[DocumentPage] = []
    supported_files = [
        f for f in sorted(list(path.glob("*.pdf")) + list(path.glob("*.docx")))
        if not f.name.startswith("~$")
    ]

    for file_file in supported_files:
        try:
            pages = load_document(file_file)
            all_pages.extend(pages)
        except DocumentLoadingError as e:
            # Continue loading other documents even if one fails
            print(f"[Warning] Error loading {file_file.name}: {e}")

    return all_pages

