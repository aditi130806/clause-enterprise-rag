"""
RAG Pipeline Orchestration & End-to-End QA Service module for CLAUSE Enterprise RAG.

Provides complete orchestration for document indexing, index reloading, and
evidence-grounded question answering with evidence sufficiency gating, grounded prompt building,
Gemini LLM generation, and source attribution validation.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL,
    SUFFICIENCY_MIN_SCORE,
    SUFFICIENCY_MIN_CHUNKS,
)
from src.document_loader import load_documents_from_directory
from src.chunking import chunk_documents
from src.stats import calculate_document_stats
from src.embeddings import EmbeddingService
from src.vector_store import FAISSVectorStore
from src.keyword_search import BM25SearchIndex
from src.retriever import HybridRetriever
from src.llm import GeminiLLMService, LLMError
from src.evidence_gate import evaluate_evidence_sufficiency, INSUFFICIENT_INFORMATION_REFUSAL
from src.prompt_builder import build_context_block, build_grounded_prompt, REFUSAL_SENTENCE
from src.attribution import parse_and_validate_citations
from src.models import RAGResponse, SourceReference, RetrievalResult, DocumentChunk
from src.query_rewriter import normalize_query, expand_query_for_corrective_retrieval
from src.reranker import rerank_retrieval_results
from src.evidence_guard import EvidenceGuard, EvidenceGuardResult

SERVICE_UNAVAILABLE_MESSAGE = "Clause could not reach the language model service. Please try again."


def build_and_index_documents(
    documents_dir: Path | str = DOCUMENTS_DIR,
    vectorstore_dir: Path | str = VECTORSTORE_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> Tuple[HybridRetriever, Dict[str, Any]]:
    """
    Full end-to-end document processing and indexing pipeline:
    Documents -> Extract -> Clean -> Chunk -> Embed -> FAISS Vector Store -> BM25 Index -> Save.

    Args:
        documents_dir: Path to directory containing source PDF / DOCX documents.
        vectorstore_dir: Path to directory for persisting FAISS and BM25 indices.
        chunk_size: Target character length per chunk.
        chunk_overlap: Overlap character length between consecutive chunks.
        embedding_model_name: SentenceTransformer model name/path.

    Returns:
        Tuple of (HybridRetriever instance, indexing_stats dictionary).
    """
    doc_path = Path(documents_dir)
    vstore_path = Path(vectorstore_dir)
    vstore_path.mkdir(parents=True, exist_ok=True)

    # 1. Load document pages
    pages = load_documents_from_directory(doc_path)
    page_stats = calculate_document_stats(pages)

    # 2. Chunk pages with context preservation
    chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 3. Initialize Embedding Service & Generate Embeddings
    embedding_service = EmbeddingService(model_name=embedding_model_name)
    dimension = embedding_service.dimension

    if chunks:
        embeddings = embedding_service.encode_chunks(chunks)
    else:
        import numpy as np
        embeddings = np.empty((0, dimension), dtype=np.float32)

    # 4. Build & Save FAISS Vector Store
    vector_store = FAISSVectorStore(dimension=dimension)
    if chunks:
        vector_store.add_chunks(chunks, embeddings)
    vector_store.save(vstore_path)

    # 5. Build & Save BM25 Keyword Search Index
    bm25_index = BM25SearchIndex()
    bm25_index.build_index(chunks)
    bm25_index.save(vstore_path)

    # 6. Instantiate HybridRetriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_service=embedding_service,
    )

    stats = {
        "documents_indexed": page_stats["total_documents"],
        "pages_processed": page_stats["total_pages"],
        "chunks_generated": len(chunks),
        "embedding_model": embedding_model_name,
        "embedding_backend": embedding_service.backend,
        "embedding_dimension": dimension,
        "faiss_index_size": vector_store.total_chunks,
        "bm25_chunk_count": bm25_index.total_chunks,
        "vectorstore_path": str(vstore_path),
    }

    return retriever, stats


def load_retrieval_engine(
    vectorstore_dir: Path | str = VECTORSTORE_DIR,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
) -> HybridRetriever:
    """
    Reload FAISS vector store and BM25 index from disk without reprocessing source documents.

    Args:
        vectorstore_dir: Directory containing faiss_index.bin, chunks_metadata.json, bm25_index.json.
        embedding_model_name: Embedding model identifier for query encoding.

    Returns:
        Fully initialized HybridRetriever instance.
    """
    vstore_path = Path(vectorstore_dir)
    vector_store = FAISSVectorStore.load(vstore_path)
    bm25_index = BM25SearchIndex.load(vstore_path)
    embedding_service = EmbeddingService(model_name=embedding_model_name)

    return HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_service=embedding_service,
    )


def answer_question(
    question: str,
    retriever: HybridRetriever,
    llm_service: Optional[GeminiLLMService] = None,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    min_sufficiency_score: float = SUFFICIENCY_MIN_SCORE,
    min_sufficiency_chunks: int = SUFFICIENCY_MIN_CHUNKS,
) -> RAGResponse:
    """
    Full End-to-End RAG QA Orchestration Pipeline:
    Question -> Hybrid Retrieval -> Evidence Sufficiency Gate -> (Refusal if insufficient) ->
    Context Construction -> Grounded Prompt Builder -> Gemini Generation -> Citation Validation -> RAGResponse.

    Args:
        question: User query string.
        retriever: Initialized HybridRetriever instance.
        llm_service: GeminiLLMService instance (or None to auto-initialize).
        retrieval_mode: Retrieval strategy ('hybrid', 'dense', 'keyword').
        top_k: Number of top candidate chunks to retrieve.
        filters: Optional metadata filters.
        min_sufficiency_score: Sufficiency threshold for top chunk score.
        min_sufficiency_chunks: Minimum candidate chunk count threshold.

    Returns:
        Structured RAGResponse object.
    """
    t_start = time.perf_counter()
    cleaned_question = question.strip() if question else ""
    llm = llm_service or GeminiLLMService()
    guard = EvidenceGuard(min_sufficiency_score=min_sufficiency_score)

    # Step 1: Validate Question Input
    if not cleaned_question:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        print(f"[Clause Timing] retrieval_ms: 0.0ms | reranking_ms: 0.0ms | evidence_ms: 0.0ms | gemini_ms: 0.0ms | total_ms: {total_ms:.1f}ms")
        return RAGResponse(
            question=question,
            answer=INSUFFICIENT_INFORMATION_REFUSAL,
            sources=[],
            retrieved_results=[],
            grounded=True,
            refused=True,
            confidence=0.0,
            confidence_label="Insufficient",
            evidence_status="insufficient",
            conflict_detected=False,
            retrieval_corrected=False,
            evidence_reasons=["Empty question input."],
            generation_model=llm.model_name,
            retrieval_mode=retrieval_mode,
            metadata={"refusal_stage": "empty_question_validation"},
        )

    # Step 2: Query Rewriting & Normalization
    normalized_query = normalize_query(cleaned_question)

    # Step 3: Initial Hybrid Retrieval
    t_ret_start = time.perf_counter()
    initial_results = retriever.retrieve(
        query=normalized_query,
        mode=retrieval_mode,
        top_k=top_k,
        filters=filters,
    )
    t_ret_end = time.perf_counter()
    retrieval_ms = (t_ret_end - t_ret_start) * 1000.0

    # Step 4: Lightweight Reranking
    t_rerank_start = time.perf_counter()
    reranked_results = rerank_retrieval_results(
        query=cleaned_question,
        results=initial_results,
        top_k=top_k,
    )
    t_rerank_end = time.perf_counter()
    reranking_ms = (t_rerank_end - t_rerank_start) * 1000.0

    # Step 5: Initial Evidence Evaluation via EvidenceGuard
    t_ev_start = time.perf_counter()
    guard_result = guard.evaluate_retrieval_evidence(
        query=cleaned_question,
        retrieved_results=reranked_results,
    )
    retrieval_corrected = False

    # Step 6: Bounded Corrective Retrieval (Exactly ONE Attempt)
    if guard_result.evidence_status in ("weak", "insufficient") and cleaned_question:
        expanded_query = expand_query_for_corrective_retrieval(cleaned_question)
        corrected_initial = retriever.retrieve(
            query=expanded_query,
            mode=retrieval_mode,
            top_k=top_k,
            filters=filters,
        )
        reranked_corrected = rerank_retrieval_results(
            query=cleaned_question,
            results=corrected_initial,
            top_k=top_k,
        )
        corrected_guard_result = guard.evaluate_retrieval_evidence(
            query=cleaned_question,
            retrieved_results=reranked_corrected,
        )

        # Select stronger evidence set if it actually yields valid sufficient evidence
        if corrected_guard_result.strongest_source_score > guard_result.strongest_source_score:
            if not corrected_guard_result.refusal_recommended:
                reranked_results = reranked_corrected
                guard_result = corrected_guard_result
                retrieval_corrected = True

    # Step 7: Evidence Sufficiency Check & Strict Refusal Gate
    is_sufficient, gate_confidence, refusal_reason, gate_diagnostics = evaluate_evidence_sufficiency(
        retrieved_results=reranked_results,
        min_score=min_sufficiency_score,
        min_chunks=min_sufficiency_chunks,
    )
    t_ev_end = time.perf_counter()
    evidence_ms = (t_ev_end - t_ev_start) * 1000.0

    if guard_result.refusal_recommended or not is_sufficient:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        print(f"[Clause Timing] retrieval_ms: {retrieval_ms:.1f}ms | reranking_ms: {reranking_ms:.1f}ms | evidence_ms: {evidence_ms:.1f}ms | gemini_ms: 0.0ms | total_ms: {total_ms:.1f}ms")
        return RAGResponse(
            question=cleaned_question,
            answer=INSUFFICIENT_INFORMATION_REFUSAL,
            sources=[],
            retrieved_results=reranked_results,
            grounded=True,
            refused=True,
            confidence=gate_confidence,
            confidence_label="Insufficient",
            evidence_status="insufficient",
            conflict_detected=guard_result.conflict_detected,
            retrieval_corrected=retrieval_corrected,
            evidence_reasons=guard_result.reasons,
            generation_model=llm.model_name,
            retrieval_mode=retrieval_mode,
            metadata={
                "refusal_stage": "evidence_sufficiency_gate",
                "refusal_reason": refusal_reason,
                "gate_diagnostics": gate_diagnostics,
                "guard_diagnostics": guard_result.to_dict(),
                "retrieval_ms": retrieval_ms,
                "reranking_ms": reranking_ms,
                "evidence_ms": evidence_ms,
                "gemini_ms": 0.0,
                "total_ms": total_ms,
            },
        )

    # Step 8: Context Construction & Grounded Prompt Building
    context_str, source_map = build_context_block(reranked_results)
    prompt = build_grounded_prompt(cleaned_question, context_str)

    # Step 9: Gemini LLM Generation (with Grounded Extractive Fallback Mode)
    from src.extractive_answer import generate_extractive_answer

    t_gem_start = time.perf_counter()
    gemini_ms = 0.0
    raw_answer = None
    validated_sources = []
    fallback_used = False

    try:
        raw_answer = llm.generate(prompt)
        t_gem_end = time.perf_counter()
        gemini_ms = (t_gem_end - t_gem_start) * 1000.0
    except Exception as llm_err:
        t_gem_end = time.perf_counter()
        gemini_ms = (t_gem_end - t_gem_start) * 1000.0

        # Gemini failed/timed out/rate limited. Attempt Grounded Extractive Fallback
        fallback_res = generate_extractive_answer(
            query=cleaned_question,
            retrieved_results=reranked_results,
        )
        if fallback_res is not None:
            raw_answer, validated_sources = fallback_res
            fallback_used = True
        else:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            print(f"[Clause Timing] retrieval_ms: {retrieval_ms:.1f}ms | reranking_ms: {reranking_ms:.1f}ms | evidence_ms: {evidence_ms:.1f}ms | gemini_ms: {gemini_ms:.1f}ms | total_ms: {total_ms:.1f}ms")
            return RAGResponse(
                question=cleaned_question,
                answer=SERVICE_UNAVAILABLE_MESSAGE,
                sources=[],
                retrieved_results=reranked_results,
                grounded=False,
                refused=True,
                confidence=0.0,
                confidence_label="Insufficient",
                evidence_status="insufficient",
                conflict_detected=guard_result.conflict_detected,
                retrieval_corrected=retrieval_corrected,
                evidence_reasons=["LLM generation error occurred and extractive fallback could not find explicit answer sentences."],
                generation_model=llm.model_name,
                retrieval_mode=retrieval_mode,
                metadata={
                    "error": str(llm_err),
                    "refusal_stage": "llm_generation_error",
                    "retrieval_ms": retrieval_ms,
                    "reranking_ms": reranking_ms,
                    "evidence_ms": evidence_ms,
                    "gemini_ms": gemini_ms,
                    "total_ms": total_ms,
                },
            )

    # Step 10: Citation Validation & Final Evidence Guard Validation
    if not fallback_used:
        validated_sources, is_grounded, attribution_diagnostics = parse_and_validate_citations(
            raw_answer, source_map
        )
    else:
        is_grounded = True
        attribution_diagnostics = {"fallback_mode": True}

    final_guard_result = guard.evaluate_retrieval_evidence(
        query=cleaned_question,
        retrieved_results=reranked_results,
        cited_sources=validated_sources,
    )

    is_refused = (
        REFUSAL_SENTENCE.lower() in raw_answer.lower()
        or INSUFFICIENT_INFORMATION_REFUSAL.lower() in raw_answer.lower()
        or SERVICE_UNAVAILABLE_MESSAGE.lower() in raw_answer.lower()
    )

    total_ms = (time.perf_counter() - t_start) * 1000.0
    print(f"[Clause Timing] retrieval_ms: {retrieval_ms:.1f}ms | reranking_ms: {reranking_ms:.1f}ms | evidence_ms: {evidence_ms:.1f}ms | gemini_ms: {gemini_ms:.1f}ms | total_ms: {total_ms:.1f}ms")

    return RAGResponse(
        question=cleaned_question,
        answer=raw_answer,
        sources=validated_sources,
        retrieved_results=reranked_results,
        grounded=is_grounded,
        refused=is_refused,
        confidence=gate_confidence if not fallback_used else min(gate_confidence, 0.85),
        confidence_label=final_guard_result.confidence_label if not fallback_used else "Moderate",
        evidence_status=final_guard_result.evidence_status if not fallback_used else "Evidence-backed fallback",
        conflict_detected=final_guard_result.conflict_detected,
        retrieval_corrected=retrieval_corrected,
        evidence_reasons=final_guard_result.reasons,
        generation_model=llm.model_name if not fallback_used else "extractive-fallback",
        retrieval_mode=retrieval_mode,
        metadata={
            "gate_diagnostics": gate_diagnostics,
            "attribution_diagnostics": attribution_diagnostics,
            "guard_diagnostics": final_guard_result.to_dict(),
            "fallback_used": fallback_used,
            "fallback_note": "Generated from retrieved document evidence because the language model service was unavailable." if fallback_used else None,
            "retrieval_ms": retrieval_ms,
            "reranking_ms": reranking_ms,
            "evidence_ms": evidence_ms,
            "gemini_ms": gemini_ms,
            "total_ms": total_ms,
        },
    )


import concurrent.futures

ASK_TIMEOUT_SECONDS = 30.0


def answer_question_with_timeout(
    question: str,
    retriever: HybridRetriever,
    llm_service: Optional[GeminiLLMService] = None,
    timeout_seconds: float = ASK_TIMEOUT_SECONDS,
    **kwargs,
) -> RAGResponse:
    """
    Executes answer_question within a hard outer thread execution timeout budget (default 30s).
    If execution exceeds timeout_seconds, attempts extractive fallback or returns standard refusal.
    """
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            answer_question,
            question=question,
            retriever=retriever,
            llm_service=llm_service,
            **kwargs,
        )
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"[Clause Timing] HARD TIMEOUT EXCEEDED ({timeout_seconds}s) | total_ms: {elapsed_ms:.1f}ms")

            # Try extractive fallback on timeout
            from src.extractive_answer import generate_extractive_answer
            try:
                initial_res = retriever.retrieve(query=question, top_k=5)
                fb_res = generate_extractive_answer(query=question, retrieved_results=initial_res)
                if fb_res is not None:
                    answer_str, sources = fb_res
                    llm_name = llm_service.model_name if llm_service else "gemini-3.8-flash"
                    return RAGResponse(
                        question=question,
                        answer=answer_str,
                        sources=sources,
                        retrieved_results=initial_res,
                        grounded=True,
                        refused=False,
                        confidence=0.75,
                        confidence_label="Moderate",
                        evidence_status="Evidence-backed fallback",
                        conflict_detected=False,
                        retrieval_corrected=False,
                        evidence_reasons=["Answered via extractive fallback mode after timeout."],
                        generation_model="extractive-fallback",
                        retrieval_mode="hybrid",
                        metadata={
                            "refusal_stage": "ask_timeout_extractive_fallback",
                            "fallback_used": True,
                            "fallback_note": "Generated from retrieved document evidence because the language model service was unavailable.",
                        },
                    )
            except Exception:
                pass

            llm_name = llm_service.model_name if llm_service else "gemini-3.8-flash"
            return RAGResponse(
                question=question,
                answer="Clause could not complete the request in time. Please try again.",
                sources=[],
                retrieved_results=[],
                grounded=False,
                refused=True,
                confidence=0.0,
                confidence_label="Insufficient",
                evidence_status="insufficient",
                conflict_detected=False,
                retrieval_corrected=False,
                evidence_reasons=["Execution timed out after 30 seconds."],
                generation_model=llm_name,
                retrieval_mode="hybrid",
                metadata={"refusal_stage": "ask_timeout_exceeded", "timeout_seconds": timeout_seconds},
            )


