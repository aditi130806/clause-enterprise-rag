"""
Script to index the official CLAUSE demonstration policy corpus into FAISS and BM25 stores.
Uses existing build_and_index_documents pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import build_and_index_documents
from src.config import DOCUMENTS_DIR, VECTORSTORE_DIR

def index_corpus():
    print(f"Indexing documents from: {DOCUMENTS_DIR}")
    print(f"Saving vectorstore to: {VECTORSTORE_DIR}")
    retriever, stats = build_and_index_documents(
        documents_dir=DOCUMENTS_DIR,
        vectorstore_dir=VECTORSTORE_DIR,
    )
    print("\n--- Indexing Statistics ---")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print("\nIndexing complete!")

if __name__ == "__main__":
    index_corpus()
