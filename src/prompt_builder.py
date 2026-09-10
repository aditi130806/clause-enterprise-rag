"""
Grounded Prompt Builder and Context Construction module for CLAUSE Enterprise RAG.

Constructs bounded context blocks with deterministic source IDs ([S1], [S2], etc.)
and generates strict system-instructed prompts for Gemini.
"""

from typing import List, Tuple, Dict, Any
from src.models import RetrievalResult, SourceReference, DocumentChunk

REFUSAL_SENTENCE = "I could not find sufficient information in the provided documents to answer this question."


def build_context_block(
    results: List[RetrievalResult],
    max_context_chars: int = 4000,
) -> Tuple[str, Dict[str, SourceReference]]:
    """
    Format top retrieved chunks into a clean, bounded context block with deterministic source IDs ([S1], [S2], ...).

    Args:
        results: List of RetrievalResult objects sorted by rank.
        max_context_chars: Maximum character length for context block.

    Returns:
        Tuple of:
        - context_str (str): Bounded text block formatted with source markers.
        - source_map (Dict[str, SourceReference]): Map from source_id (e.g. "S1") -> SourceReference object.
    """
    context_lines: List[str] = []
    source_map: Dict[str, SourceReference] = {}
    current_chars = 0

    seen_chunk_ids = set()

    for idx, result in enumerate(results, start=1):
        chunk = result.chunk
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)

        source_id = f"S{idx}"
        page_str = f"Page {chunk.page_number}" if chunk.page_number is not None else "Page N/A"
        sec_str = f"Section: {chunk.section}" if chunk.section else "Section: N/A"

        block = (
            f"[{source_id}]\n"
            f"Document: {chunk.document_name}\n"
            f"{page_str} | {sec_str}\n"
            f"Content:\n{chunk.text.strip()}\n"
        )

        if current_chars + len(block) > max_context_chars and idx > 1:
            break

        context_lines.append(block)
        current_chars += len(block)

        # Build structured SourceReference mapping
        best_score = max(result.fusion_score, result.dense_score, result.bm25_score)
        excerpt = chunk.text[:150].strip() + ("..." if len(chunk.text) > 150 else "")

        source_map[source_id] = SourceReference(
            source_id=source_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            section=chunk.section,
            chunk_id=chunk.chunk_id,
            excerpt=excerpt,
            retrieval_rank=result.final_rank,
            retrieval_score=best_score,
        )

    context_str = "\n".join(context_lines)
    return context_str, source_map


def build_grounded_prompt(
    question: str,
    context_str: str,
) -> str:
    """
    Construct system-guided grounded prompt for Gemini LLM generation.

    Args:
        question: User query string.
        context_str: Formatted context block containing source-tagged document excerpts.

    Returns:
        Full prompt string to be passed to LLM.
    """
    prompt = f"""You are Clause, an enterprise document intelligence assistant.

ROLE & OBJECTIVE:
Your sole task is to answer the user question accurately using ONLY the provided evidence blocks below.

SYSTEM RULES:
1. Answer ONLY from the supplied evidence blocks.
2. Do not use general world knowledge to fill in missing information.
3. Do not invent or assume policies, dates, amounts, names, rules, or exceptions.
4. If the supplied evidence does not support an answer, respond with EXACTLY:
   "{REFUSAL_SENTENCE}"
5. When multiple evidence chunks agree, synthesize them cleanly.
6. If evidence appears inconsistent, do not silently choose one. State that the retrieved documents contain potentially conflicting information and cite both sources.
7. Preserve important qualifiers such as approvals, exceptions, eligibility requirements, deadlines, monetary amounts, and policy conditions.
8. Use concise, professional enterprise language.
9. Cite evidence inline using ONLY the exact supplied source IDs (e.g., [S1], [S2]). Do NOT create source IDs that were not supplied.

EVIDENCE BLOCKS:
{context_str}

USER QUESTION:
{question}

ANSWER & SOURCE CITATIONS:"""

    return prompt
