from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class DocumentPage:
    """Represents a single extracted page or document section from an ingested file."""
    document_id: str
    document_name: str
    file_type: str  # "pdf" or "docx"
    page_number: Optional[int]  # 1-indexed for PDF; None for DOCX if unpaginated
    text: str
    source_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page object to dictionary representation."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "file_type": self.file_type,
            "page_number": self.page_number,
            "text": self.text,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }


@dataclass
class DocumentChunk:
    """Represents a discrete text chunk extracted from a document page for embedding and retrieval."""
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    text: str
    section: Optional[str]
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "text": self.text,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Reconstruct DocumentChunk from dictionary representation."""
        return cls(
            chunk_id=data["chunk_id"],
            document_id=data["document_id"],
            document_name=data["document_name"],
            page_number=data.get("page_number"),
            text=data["text"],
            section=data.get("section"),
            chunk_index=data["chunk_index"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """
    Represents a retrieved document chunk with source attribution, dense/BM25 scores,
    individual rank information, and final hybrid fusion rank.
    """
    chunk: DocumentChunk
    document_name: str
    page_number: Optional[int]
    section: Optional[str]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    fusion_score: float = 0.0
    final_rank: int = 1
    retrieval_methods: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval result to dictionary representation."""
        return {
            "chunk": self.chunk.to_dict(),
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section": self.section,
            "dense_score": float(self.dense_score),
            "bm25_score": float(self.bm25_score),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "fusion_score": float(self.fusion_score),
            "final_rank": self.final_rank,
            "retrieval_methods": self.retrieval_methods,
        }


@dataclass
class SourceReference:
    """
    Structured attribution details for a cited source chunk.
    """
    source_id: str  # e.g., "S1"
    document_id: str
    document_name: str
    page_number: Optional[int]
    section: Optional[str]
    chunk_id: str
    excerpt: str
    retrieval_rank: int
    retrieval_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert SourceReference to dictionary representation."""
        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": float(self.retrieval_score),
        }


@dataclass
class RAGResponse:
    """
    Complete structured RAG pipeline response object.
    Contains user question, generated answer, validated source attributions,
    refusal state, groundedness flag, confidence score, and execution metadata.
    """
    question: str
    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    retrieved_results: List[RetrievalResult] = field(default_factory=list)
    grounded: bool = True
    refused: bool = False
    confidence: float = 1.0
    generation_model: str = ""
    retrieval_mode: str = "hybrid"
    confidence_label: str = "High"
    evidence_status: str = "sufficient"
    conflict_detected: bool = False
    retrieval_corrected: bool = False
    evidence_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert RAGResponse to dictionary representation."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "retrieved_results": [r.to_dict() for r in self.retrieved_results],
            "grounded": self.grounded,
            "refused": self.refused,
            "confidence": float(self.confidence),
            "generation_model": self.generation_model,
            "retrieval_mode": self.retrieval_mode,
            "confidence_label": self.confidence_label,
            "evidence_status": self.evidence_status,
            "conflict_detected": self.conflict_detected,
            "retrieval_corrected": self.retrieval_corrected,
            "evidence_reasons": self.evidence_reasons,
            "metadata": self.metadata,
        }
