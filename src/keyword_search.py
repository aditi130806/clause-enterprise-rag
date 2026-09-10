"""
BM25 Sparse Keyword Search module for CLAUSE Enterprise RAG.

Provides exact lexical, phrase, identifier (e.g. policy ID 'HR-RW-017'), acronym,
and technical term matching using BM25Plus from rank-bm25.
Uses a transparent, lightweight regex tokenizer while keeping original source text untouched.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from rank_bm25 import BM25Plus

from src.config import VECTORSTORE_DIR
from src.models import DocumentChunk


class BM25SearchError(Exception):
    """Custom exception raised for BM25 keyword search operations."""
    pass


def default_bm25_tokenizer(text: str) -> List[str]:
    """
    Lightweight, transparent tokenizer for BM25 search index representation.
    Converts text to lowercase and extracts both hyphenated policy identifiers (e.g. 'hr-rw-017')
    and single alphanumeric words ('hr', 'rw', '017', 'remote').
    Original source chunk.text remains untouched.

    Args:
        text: Input string.

    Returns:
        List of lowercase token strings.
    """
    if not text or not text.strip():
        return []

    text_lower = text.lower()
    # Match full hyphenated policy identifiers (e.g. "hr-rw-017", "sec-auth-004")
    hyphenated_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text_lower)
    # Match individual alphanumeric words
    word_tokens = re.findall(r"[a-z0-9]+", text_lower)

    # Combine and preserve order while avoiding duplicate token noise
    seen = set()
    combined_tokens = []
    for token in hyphenated_tokens + word_tokens:
        if token not in seen:
            seen.add(token)
            combined_tokens.append(token)

    return combined_tokens


class BM25SearchIndex:
    """
    BM25 sparse keyword search index manager using BM25Plus.
    BM25Plus prevents negative IDF scores on small corpora while retrieving exact lexical evidence,
    acronyms, code identifiers, and specific policy IDs.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Plus] = None

    @property
    def total_chunks(self) -> int:
        """Return total number of chunks indexed in BM25."""
        return len(self.chunks)

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """
        Build BM25Plus index from a list of DocumentChunk objects.

        Args:
            chunks: List of DocumentChunk instances.
        """
        if not chunks:
            self.chunks = []
            self.corpus_tokens = []
            self.bm25 = None
            return

        self.chunks = list(chunks)
        # Tokenize chunk text for search representation without mutating original text
        self.corpus_tokens = [default_bm25_tokenizer(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Plus(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float, int]]:
        """
        Search Top-K relevant chunks using BM25 query token scoring.

        Args:
            query: User input search string.
            top_k: Maximum number of results to return.

        Returns:
            List of tuples: (DocumentChunk, bm25_score, rank_1_indexed) sorted by descending score.
        """
        if not self.bm25 or self.total_chunks == 0 or not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query_tokens = default_bm25_tokenizer(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)

        # Pair each chunk index with its BM25 score
        scored_pairs = [(idx, float(score)) for idx, score in enumerate(doc_scores) if score > 0.0]

        # Sort by score descending; break ties by original chunk index for determinism
        scored_pairs.sort(key=lambda item: (-item[1], item[0]))

        actual_k = min(top_k, len(scored_pairs))
        results: List[Tuple[DocumentChunk, float, int]] = []

        for rank_idx, (idx, score) in enumerate(scored_pairs[:actual_k], start=1):
            chunk = self.chunks[idx]
            results.append((chunk, score, rank_idx))

        return results

    def save(self, dir_path: Path | str = VECTORSTORE_DIR) -> None:
        """
        Persist BM25 index state and chunk metadata to JSON for rapid reloading.

        Args:
            dir_path: Directory path where BM25 state will be saved.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        bm25_file = dir_path / "bm25_index.json"

        try:
            serialized_chunks = [chunk.to_dict() for chunk in self.chunks]
            payload = {
                "total_chunks": self.total_chunks,
                "chunks": serialized_chunks,
                "corpus_tokens": self.corpus_tokens,
            }
            with open(bm25_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as err:
            raise BM25SearchError(f"Failed to save BM25 index to '{dir_path}': {err}") from err

    @classmethod
    def load(cls, dir_path: Path | str = VECTORSTORE_DIR) -> "BM25SearchIndex":
        """
        Load BM25 index from disk without reprocessing documents.

        Args:
            dir_path: Directory containing bm25_index.json.

        Returns:
            Populated BM25SearchIndex instance.
        """
        dir_path = Path(dir_path)
        bm25_file = dir_path / "bm25_index.json"

        if not bm25_file.exists():
            raise BM25SearchError(f"BM25 index file not found at '{bm25_file}'. Run indexing first.")

        try:
            with open(bm25_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            instance = cls()
            raw_chunks = payload.get("chunks", [])
            instance.chunks = [DocumentChunk.from_dict(c) for c in raw_chunks]
            instance.corpus_tokens = payload.get("corpus_tokens", [])

            if instance.corpus_tokens:
                instance.bm25 = BM25Plus(instance.corpus_tokens)
            else:
                instance.bm25 = None

            return instance
        except Exception as err:
            raise BM25SearchError(f"Failed to load BM25 index from '{dir_path}': {err}") from err
