# Clause Enterprise RAG — Benchmark Evaluation Methodology & Results

## 1. Evaluation Overview

To ensure production readiness for enterprise deployment, **Clause** was benchmarked using a 24-question offline evaluation dataset (`evaluation/rag_evaluation_dataset.csv`).

The evaluation methodology measures both **Retrieval Effectiveness** (Hit@1, Hit@3, MRR) and **Safety / Guardrail Integrity** (Source Accuracy, Correct Refusal Rate, Grounding Validation Rate).

---

## 2. Benchmark Dataset Composition

The 24-question benchmark dataset spans 5 standard evaluation categories:

| Category ID | Category Name | Question Count | Supported / Unsupported | Objective Tested |
| :--- | :--- | :---: | :---: | :--- |
| **CAT-01** | Semantic Retrieval | 6 | Supported | Natural language semantic matching of enterprise policy rules. |
| **CAT-02** | Exact Identifier / Keyword | 4 | Supported | Pinpoint retrieval of exact policy codes (e.g. `HR-RW-017`, `FIN-EXP-011`). |
| **CAT-03** | Complex Paraphrased | 4 | Supported | Robustness against indirect phrasing and conversational queries. |
| **CAT-04** | Multi-Document Evidence | 4 | Supported | Synthesis across multi-document sources (e.g. Remote Work + Security). |
| **CAT-05** | Unsupported Questions | 6 | Unsupported | Strict refusal of unmentioned or non-existent policy topics. |
| **Total** | | **24** | **18 / 6** | Comprehensive Enterprise Benchmark |

---

## 3. Metric Definitions & Formulas

### 3.1 Retrieval Metrics (Supported Questions, N=18)
- **Hit@1 Rate**: Percentage of supported queries where the ground-truth target document is ranked #1.
  $$\text{Hit@1} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Rank}_i = 1)}{N}$$
- **Hit@3 Rate**: Percentage of supported queries where the target document is within top 3 retrieved results.
  $$\text{Hit@3} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Rank}_i \le 3)}{N}$$
- **Mean Reciprocal Rank (MRR)**: Average inverse rank of the first correct document.
  $$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{Rank}_i}$$

### 3.2 Safety & Guardrail Metrics (N=24)
- **Expected Source Accuracy**: Percentage of answered questions where the top source document matches expected policy doc.
- **Correct Refusal Rate**: Percentage of unsupported queries correctly flagged and refused by Evidence Guard (`refused = True`).
- **Grounding Validation Rate**: Percentage of generated answers containing valid, traceable inline source citations (`[S1]`, `[S2]`).

---

## 4. Final Verified Evaluation Results

Evaluation executed via `evaluate_rag.py` against the official synthetic policy demonstration corpus:

| Metric Name | Target Benchmark | Achieved Result | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **Total Benchmark Queries** | 24 | **24** | **PASS** |
| **Supported Questions** | 18 | **18** | **PASS** |
| **Unsupported Questions** | 6 | **6** | **PASS** |
| **Retrieval Hit@1 Rate** | $\ge 85.0\%$ | **94.4%** | **PASS** |
| **Retrieval Hit@3 Rate** | $\ge 95.0\%$ | **100.0%** | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.900$ | **0.963** | **PASS** |
| **Expected Source Accuracy** | $\ge 95.0\%$ | **100.0%** | **PASS** |
| **Correct Refusal Rate** | $100.0\%$ | **100.0%** | **PASS** |
| **Grounding Validation Rate** | $100.0\%$ | **100.0%** | **PASS** |
| **Unit Test Suite Pass Rate** | $100.0\%$ | **63 / 63 PASS** | **PASS** |

---

## 5. Category-Level Performance Breakdown

| Question Category | Count | Hit@1 | Hit@3 | MRR | Source Accuracy | Refusal Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Semantic Retrieval** | 6 | 100.0% | 100.0% | 1.000 | 100.0% | N/A |
| **Exact Identifier (`HR-RW-017`)** | 4 | 100.0% | 100.0% | 1.000 | 100.0% | N/A |
| **Paraphrased Queries** | 4 | 75.0% | 100.0% | 0.875 | 100.0% | N/A |
| **Multi-Document Evidence** | 4 | 100.0% | 100.0% | 1.000 | 100.0% | N/A |
| **Unsupported Refusal** | 6 | N/A | N/A | N/A | N/A | **100.0%** |

---

## 6. Reproducing Benchmark Results

To execute the offline deterministic evaluation runner locally:

```powershell
.\.venv\Scripts\python evaluate_rag.py
```

The script generates:
- `evaluation/evaluation_results.json`: Full diagnostic breakdown per question.
- `evaluation/evaluation_results.csv`: Tabular metric report.

---

## 7. Known Evaluation Limitations
1. **Deterministic Offline Evaluation**: The primary benchmark utilizes `OfflineEvaluationLLM` to execute sub-second evaluations without consuming API quota. Live Gemini generation is validated separately via unit tests (`test_rag_pipeline.py`).
2. **Synthetic Demonstration Data**: Results reflect performance on the 6-document synthetic enterprise policy corpus. Deployment onto third-party corporate corpora may require threshold calibration.
