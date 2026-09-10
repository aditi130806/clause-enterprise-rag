# Demonstration Policy Corpus Disclosure

> **[!IMPORTANT]**
> **Synthetic Demonstration Data Disclosure**: All documents in this directory (`data/documents/`) are original, synthetic demonstration policy documents created solely for the CLAUSE Enterprise RAG capstone project and benchmark evaluation.

## Disclosure Statement

1. **Originality & Synthetic Nature**: These policies were authored specifically to demonstrate multi-document RAG capabilities, citation validation, policy identifier indexing, and strict evidence sufficiency gating. They do **NOT** represent actual rules, policies, or operational standards of any real-world corporation or entity.
2. **Reproducibility**: All `.docx` files in this directory are reproducibly generated via [`scripts/create_demo_corpus.py`](../../scripts/create_demo_corpus.py).
3. **No Confidential Data**: No confidential, proprietary, or copyrighted enterprise data is included or referenced in this corpus.
4. **Purpose**: The primary purpose of this corpus is to provide a standardized, deterministic ground-truth benchmark for evaluating hybrid retrieval (FAISS + BM25), lightweight reranking, Evidence Guard accuracy, and refusal behavior on unsupported questions.

## Included Policy Documents

| File Name | Policy Title | Policy Identifier | Version |
| :--- | :--- | :--- | :--- |
| `Remote Work Policy.docx` | Remote Work & Hybrid Workplace Policy | `HR-RW-017` | 3.2 |
| `Annual Leave & PTO Policy.docx` | Annual Leave, Paid Time Off (PTO) & Holiday Policy | `HR-LV-004` | 2.1 |
| `Expense Reimbursement Policy.docx` | Business Travel & Expense Reimbursement Policy | `FIN-EXP-011` | 4.0 |
| `Information Security Policy.docx` | Enterprise Information Security & Acceptable Use Policy | `SEC-IS-002` | 5.1 |
| `Attendance & Flexible Work Hours Policy.docx` | Working Hours, Attendance & Core Hours Policy | `HR-AT-009` | 1.4 |
| `Employee Benefits & Wellness Policy.docx` | Health Benefits, Wellness Stipend & Employee Care Policy | `HR-BEN-006` | 2.0 |
