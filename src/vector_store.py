"""
FAISS Vector Store module for CLAUSE Enterprise RAG.

Provides dense vector indexing and cosine similarity search using FAISS CPU (faiss.IndexFlatIP).
Preserves deterministic mapping between FAISS row indices and DocumentChunk objects.
Supports JSON-based persistence and reload capabilities without reprocessing documents.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import faiss
import numpy as np

from src.config import VECTORSTORE_DIR
from src.models import DocumentChunk


class VectorStoreError(Exception):
    """Custom exception raised for vector store operations."""
    pass


class FAISSVectorStore:
    """
    FAISS-backed vector store for dense semantic retrieval.
    Uses IndexFlatIP with L2-normalized embeddings for exact cosine similarity search.
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise VectorStoreError(f"Vector store dimension must be positive, got {dimension}.")
        self.dimension: int = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.chunks: List[DocumentChunk] = []

    @property
    def total_chunks(self) -> int:
        """Return total number of chunks currently indexed in FAISS."""
        return len(self.chunks)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """
        Add DocumentChunk objects and their normalized dense embeddings to the FAISS index.

        Args:
            chunks: List of DocumentChunk instances.
            embeddings: np.ndarray of shape (len(chunks), dimension) with float32 dtype.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
            )

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding shape {embeddings.shape} does not match vector store dimension ({self.dimension})."
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Perform top-K cosine similarity search using a query vector.

        Args:
            query_embedding: 1D or 2D float32 query vector of shape (dimension,) or (1, dimension).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of tuples: (DocumentChunk, dense_score, rank_1_indexed) sorted by descending similarity.
        """
        if self.total_chunks == 0:
            return []

        if top_k <= 0:
            return []

        # Reshape to (1, dimension) if 1D array
        if query_embedding.ndim == 1:
            query_vec = query_embedding.reshape(1, -1)
        elif query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
            query_vec = query_embedding
        else:
            raise VectorStoreError(f"Query embedding must be 1D or 2D single vector, got shape {query_embedding.shape}.")

        if query_vec.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension ({query_vec.shape[1]}) does not match index dimension ({self.dimension})."
            )

        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        actual_k = min(top_k, self.total_chunks)
        scores, indices = self.index.search(query_vec, actual_k)

        results: List[Tuple[DocumentChunk, float, int]] = []
        for rank_idx, (score, row_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if row_idx < 0 or row_idx >= len(self.chunks):
                continue
            chunk = self.chunks[row_idx]
            results.append((chunk, float(score), rank_idx))

        return results

    def save(self, dir_path: Path | str = VECTORSTORE_DIR) -> None:
        """
        Persist FAISS index file and chunk mapping metadata to disk.

        Args:
            dir_path: Directory path where vectorstore artifacts should be saved.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / "faiss_index.bin"
        metadata_file = dir_path / "chunks_metadata.json"

        try:
            faiss.write_index(self.index, str(index_file))

            serialized_chunks = [chunk.to_dict() for chunk in self.chunks]
            metadata_payload = {
                "dimension": self.dimension,
                "total_chunks": self.total_chunks,
                "chunks": serialized_chunks,
            }

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_payload, f, indent=2, ensure_ascii=False)
        except Exception as err:
            raise VectorStoreError(f"Failed to save FAISS vector store to '{dir_path}': {err}") from err

    @classmethod
    def load(cls, dir_path: Path | str = VECTORSTORE_DIR) -> "FAISSVectorStore":
        """
        Load FAISS index and chunk mapping metadata from disk without reprocessing documents.

        Args:
            dir_path: Directory path containing faiss_index.bin and chunks_metadata.json.

        Returns:
            FAISSVectorStore instance with populated index and chunk metadata.
        """
        dir_path = Path(dir_path)
        index_file = dir_path / "faiss_index.bin"
        metadata_file = dir_path / "chunks_metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise VectorStoreError(f"Vector store files not found in '{dir_path}'. Run indexing first.")

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_payload = json.load(f)

            dimension = metadata_payload["dimension"]
            store = cls(dimension=dimension)

            loaded_index = faiss.read_index(str(index_file))
            store.index = loaded_index

            raw_chunks = metadata_payload.get("chunks", [])
            store.chunks = [DocumentChunk.from_dict(c_dict) for c_dict in raw_chunks]

            return store
        except Exception as err:
            raise VectorStoreError(f"Failed to load FAISS vector store from '{dir_path}': {err}") from err
