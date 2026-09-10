"""
Evaluation Engine module for CLAUSE RAG.
Computes deterministic offline metrics: Retrieval Hit@1, Hit@3, MRR,
Expected Source Accuracy, Correct Refusal Rate, and Citation Grounding Validation.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.rag_pipeline import answer_question, HybridRetriever
from src.llm import GeminiLLMService
from src.config import SUFFICIENCY_MIN_SCORE

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"


class OfflineEvaluationLLM:
    """Lightweight dummy LLM service for fast offline retrieval & refusal evaluation."""
    def __init__(self):
        self.model_name = "offline-evaluator"

    def generate(self, prompt: str) -> str:
        return "Offline evaluation answer [S1]."


from datetime import datetime
from src.kb_state import get_knowledge_base_status, compute_kb_fingerprint


def load_benchmark_questions(dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the ground-truth 24 benchmark questions from dataset JSON."""
    d_path = dataset_path or (EVAL_DIR / "rag_evaluation_dataset.json")
    if not d_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {d_path}")
    with open(d_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval_and_rag(
    retriever: HybridRetriever,
    dataset_path: Optional[Path] = None,
    llm_service: Optional[GeminiLLMService] = None,
    eval_llm_limit: int = 8,
    kb_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation across dataset questions.

    Args:
        retriever: Loaded HybridRetriever instance.
        dataset_path: Path to rag_evaluation_dataset.json (defaults to evaluation/rag_evaluation_dataset.json).
        llm_service: Optional GeminiLLMService for live generation evaluation.
        eval_llm_limit: Maximum number of questions to evaluate through LLM generation.
        kb_fingerprint: Optional knowledge base fingerprint hash.

    Returns:
        Summary metrics dictionary and saves evaluation_results.json & evaluation_results.csv.
    """
    kb_status = get_knowledge_base_status()
    final_fingerprint = kb_fingerprint or compute_kb_fingerprint()

    d_path = dataset_path or (EVAL_DIR / "rag_evaluation_dataset.json")
    if not d_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {d_path}")

    with open(d_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)

    total_questions = len(dataset)
    hit1_count = 0
    hit3_count = 0
    total_rr = 0.0
    source_acc_count = 0
    
    unsupported_total = 0
    unsupported_refused_correctly = 0

    grounded_checked_total = 0
    grounded_passed_count = 0

    per_question_results = []
    llm_eval_counter = 0

    for item in dataset:
        q_id = item["question_id"]
        category = item["category"]
        question_text = item["question"]
        expected_docs_str = item["expected_source_document"]
        expected_supported = item["expected_supported"]

        # Parse expected docs list
        expected_docs = [d.strip() for d in expected_docs_str.split(",") if d.strip() and d.strip().lower() != "none"]

        # Decide whether to execute full RAG (with Gemini) or pipeline without Gemini
        use_llm = False
        if llm_service is not None and llm_eval_counter < eval_llm_limit:
            use_llm = True
            llm_eval_counter += 1

        # Run pipeline answer_question with active_llm
        active_llm = llm_service if use_llm else OfflineEvaluationLLM()
        response = answer_question(
            question=question_text,
            retriever=retriever,
            llm_service=active_llm,
        )

        retrieved_docs = [r.document_name for r in response.retrieved_results]
        top_1_doc = retrieved_docs[0] if retrieved_docs else None
        top_3_docs = retrieved_docs[:3]

        # 1. Hit@1
        hit1 = any(e_doc in (top_1_doc or "") for e_doc in expected_docs) if expected_docs else False
        if hit1 or not expected_supported:
            if expected_supported:
                hit1_count += 1

        # 2. Hit@3
        hit3 = any(any(e_doc in doc for doc in top_3_docs) for e_doc in expected_docs) if expected_docs else False
        if hit3 or not expected_supported:
            if expected_supported:
                hit3_count += 1

        # 3. Reciprocal Rank (MRR)
        rr = 0.0
        if expected_docs and retrieved_docs:
            for idx, doc in enumerate(retrieved_docs, start=1):
                if any(e_doc in doc for e_doc in expected_docs):
                    rr = 1.0 / idx
                    break
        total_rr += rr

        # 4. Expected Source Accuracy
        source_acc = hit3  # source in top 3
        if source_acc:
            source_acc_count += 1

        # 5. Unsupported Refusal Rate
        refused_correctly = False
        if not expected_supported:
            unsupported_total += 1
            # Refused if pipeline output refused or answer matches refusal string
            if response.refused or "sufficient information" in response.answer.lower():
                unsupported_refused_correctly += 1
                refused_correctly = True

        # 6. Groundedness check (if generated answer)
        grounded_passed = False
        if use_llm and not response.refused:
            grounded_checked_total += 1
            if response.grounded and len(response.sources) > 0:
                grounded_passed = True
                grounded_passed_count += 1

        row_result = {
            "question_id": q_id,
            "category": category,
            "question": question_text,
            "expected_supported": expected_supported,
            "expected_docs": expected_docs,
            "top_retrieved_docs": retrieved_docs[:3],
            "hit_at_1": hit1 if expected_supported else True,
            "hit_at_3": hit3 if expected_supported else True,
            "reciprocal_rank": rr,
            "refused": response.refused,
            "correct_refusal": refused_correctly if not expected_supported else None,
            "confidence_label": response.confidence_label,
            "evidence_status": response.evidence_status,
            "used_llm": use_llm,
        }
        per_question_results.append(row_result)

    supported_count = total_questions - unsupported_total

    hit_at_1_rate = (hit1_count / supported_count) if supported_count > 0 else 1.0
    hit_at_3_rate = (hit3_count / supported_count) if supported_count > 0 else 1.0
    mrr = (total_rr / supported_count) if supported_count > 0 else 1.0
    expected_source_accuracy = (source_acc_count / supported_count) if supported_count > 0 else 1.0
    correct_refusal_rate = (unsupported_refused_correctly / unsupported_total) if unsupported_total > 0 else 1.0
    grounding_validation_rate = (grounded_passed_count / grounded_checked_total) if grounded_checked_total > 0 else 1.0

    summary_results = {
        "knowledge_base_fingerprint": final_fingerprint,
        "evaluated_at": datetime.now().isoformat(),
        "document_count": kb_status.doc_count,
        "chunk_count": kb_status.chunk_count,
        "total_questions": total_questions,
        "supported_questions": supported_count,
        "unsupported_questions": unsupported_total,
        "retrieval_hit_at_1": float(hit_at_1_rate),
        "retrieval_hit_at_3": float(hit_at_3_rate),
        "mrr": float(mrr),
        "expected_source_accuracy": float(expected_source_accuracy),
        "correct_refusal_rate": float(correct_refusal_rate),
        "grounding_validation_rate": float(grounding_validation_rate),
        "llm_eval_samples_run": llm_eval_counter,
        "details": per_question_results,
    }

    # Save to JSON
    json_out = EVAL_DIR / "evaluation_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Saved evaluation JSON results: {json_out}")

    # Save to CSV
    csv_out = EVAL_DIR / "evaluation_results.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "category", "question", "expected_supported",
            "hit_at_1", "hit_at_3", "reciprocal_rank", "refused", "correct_refusal", "confidence_label"
        ])
        writer.writeheader()
        for d in per_question_results:
            writer.writerow({
                "question_id": d["question_id"],
                "category": d["category"],
                "question": d["question"],
                "expected_supported": d["expected_supported"],
                "hit_at_1": d["hit_at_1"],
                "hit_at_3": d["hit_at_3"],
                "reciprocal_rank": f"{d['reciprocal_rank']:.2f}",
                "refused": d["refused"],
                "correct_refusal": d["correct_refusal"],
                "confidence_label": d["confidence_label"],
            })
    print(f"Saved evaluation CSV results: {csv_out}")

    return summary_results
