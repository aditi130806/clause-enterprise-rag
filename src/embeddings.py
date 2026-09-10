"""
Embedding generation module for CLAUSE Enterprise RAG.

Provides lazy-loaded dense vector embeddings for document chunks and user queries.
L2-normalizes vectors so inner-product search (FAISS IndexFlatIP) corresponds directly
to cosine similarity.
"""

from typing import List, Optional, Union
import numpy as np

from src.config import EMBEDDING_MODEL_NAME
from src.models import DocumentChunk
from src.chunking import build_embedding_text


class EmbeddingError(Exception):
    """Custom exception raised for errors during embedding initialization or encoding."""
    pass


class EmbeddingService:
    """
    Manages embedding model lifecycle and vector generation.
    Supports lazy loading and graceful fallback between sentence-transformers and fastembed ONNX runtime.
    Caches model globally to prevent repeated disk/memory initializations.
    """

    _global_model = None
    _global_backend: Optional[str] = None
    _global_dimension: Optional[int] = None
    _global_model_name: Optional[str] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._backend: Optional[str] = None
        self._dimension: Optional[int] = None

    def _load_model(self) -> None:
        """Lazily load the embedding model if not already initialized."""
        if self._model is not None:
            return

        # Reuse class-level singleton if already loaded for the same model_name
        if (
            EmbeddingService._global_model is not None
            and EmbeddingService._global_model_name == self.model_name
        ):
            self._model = EmbeddingService._global_model
            self._backend = EmbeddingService._global_backend
            self._dimension = EmbeddingService._global_dimension
            return

        # Attempt 1: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            loaded_model = SentenceTransformer(self.model_name)
            dummy_vec = loaded_model.encode(["test"], convert_to_numpy=True)
            dimension = int(dummy_vec.shape[1])
            backend = "sentence_transformers"

            self._model = loaded_model
            self._backend = backend
            self._dimension = dimension

            EmbeddingService._global_model = loaded_model
            EmbeddingService._global_backend = backend
            EmbeddingService._global_dimension = dimension
            EmbeddingService._global_model_name = self.model_name
            return
        except Exception as st_err:
            pass

        # Attempt 2: fastembed ONNX runtime fallback
        try:
            from fastembed import TextEmbedding
            # Normalize model identifier for fastembed
            fe_name = "sentence-transformers/all-MiniLM-L6-v2" if "MiniLM" in self.model_name else self.model_name
            loaded_model = TextEmbedding(model_name=fe_name)
            dummy_gen = list(loaded_model.embed(["test"]))
            dummy_arr = np.array(dummy_gen[0])
            dimension = int(dummy_arr.shape[0])
            backend = "fastembed"

            self._model = loaded_model
            self._backend = backend
            self._dimension = dimension

            EmbeddingService._global_model = loaded_model
            EmbeddingService._global_backend = backend
            EmbeddingService._global_dimension = dimension
            EmbeddingService._global_model_name = self.model_name
            return
        except Exception as fe_err:
            raise EmbeddingError(
                f"Failed to initialize embedding model '{self.model_name}' with sentence-transformers "
                f"or fastembed backend: {fe_err}"
            ) from fe_err

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        self._load_model()
        if self._dimension is None:
            raise EmbeddingError("Embedding dimension could not be determined.")
        return self._dimension

    @property
    def backend(self) -> str:
        """Return the active backend name ('sentence_transformers' or 'fastembed')."""
        self._load_model()
        return self._backend or "unknown"

    def _l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize float32 vectors along axis=1.
        Ensures dot-product (IP) equals cosine similarity.
        """
        if vectors.size == 0:
            return vectors.astype(np.float32)
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return (vectors / norms).astype(np.float32)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of raw text strings into L2-normalized float32 NumPy vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray of shape (len(texts), dimension) with dtype float32.
        """
        self._load_model()
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            if self._backend == "sentence_transformers":
                raw_vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                vecs = np.asarray(raw_vecs, dtype=np.float32)
            else:
                raw_gen = self._model.embed(texts)
                vecs = np.array(list(raw_gen), dtype=np.float32)

            return self._l2_normalize(vecs)
        except Exception as err:
            raise EmbeddingError(f"Error generating text embeddings: {err}") from err

    def encode_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Encode a list of DocumentChunk objects into dense embeddings using build_embedding_text(chunk).
        Original chunk text remains untouched for exact source citations.

        Args:
            chunks: List of DocumentChunk instances.

        Returns:
            np.ndarray of shape (len(chunks), dimension) with dtype float32.
        """
        self._load_model()
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        contextual_texts = [build_embedding_text(chunk) for chunk in chunks]
        return self.encode_texts(contextual_texts)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single user search query string into a normalized 1D float32 NumPy array.

        Args:
            query: User input query string.

        Returns:
            np.ndarray of shape (dimension,) with dtype float32.
        """
        if not query or not query.strip():
            raise EmbeddingError("Cannot encode an empty or whitespace query.")

        matrix = self.encode_texts([query.strip()])
        return matrix[0]
