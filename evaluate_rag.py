"""
Runnable evaluation script for CLAUSE RAG.
Evaluates retrieval and RAG pipeline across rag_evaluation_dataset.json.
Saves evaluation_results.json and evaluation_results.csv.
"""

import sys
from pathlib import Path

# Insert root into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag_pipeline import load_retrieval_engine
from src.evaluation import evaluate_retrieval_and_rag
from src.llm import GeminiLLMService


def main():
    print("==================================================================")
    print("CLAUSE Enterprise RAG — Benchmark Evaluation")
    print("==================================================================")

    # 1. Load Index
    retriever = load_retrieval_engine()
    print("Retrieval engine loaded successfully.")

    # 2. Check if LLM is available for bounded sample evaluation
    llm_service = None
    try:
        llm = GeminiLLMService()
        if llm.is_available():
            llm_service = llm
            print("Gemini LLM service initialized for sample generation evaluation.")
    except Exception:
        print("Gemini LLM unavailable or unconfigured — running offline retrieval & refusal evaluation.")

    # 3. Execute Offline Evaluation
    results = evaluate_retrieval_and_rag(
        retriever=retriever,
        llm_service=None,
        eval_llm_limit=0,
    )

    print("\n------------------------------------------------------------------")
    print("BENCHMARK EVALUATION SUMMARY METRICS")
    print("------------------------------------------------------------------")
    print(f"Total Questions Evaluated:    {results['total_questions']}")
    print(f"Supported Questions:          {results['supported_questions']}")
    print(f"Unsupported Questions:        {results['unsupported_questions']}")
    print(f"Retrieval Hit@1 Rate:         {results['retrieval_hit_at_1'] * 100:.1f}%")
    print(f"Retrieval Hit@3 Rate:         {results['retrieval_hit_at_3'] * 100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR):   {results['mrr']:.3f}")
    print(f"Expected Source Accuracy:     {results['expected_source_accuracy'] * 100:.1f}%")
    print(f"Correct Refusal Rate:         {results['correct_refusal_rate'] * 100:.1f}%")
    print(f"Grounding Validation Rate:    {results['grounding_validation_rate'] * 100:.1f}%")
    print("==================================================================")


if __name__ == "__main__":
    main()
