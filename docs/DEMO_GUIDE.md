# Clause Enterprise RAG — 3-Minute Live Demonstration Guide

This guide provides a structured 3-to-5 minute demonstration script for presenting **Clause Enterprise RAG** to stakeholders or technical evaluators.

---

## Pre-Demo Checklist

1. Activate virtual environment and launch application:
   ```powershell
   python -m streamlit run app.py
   ```
2. Open browser at `http://localhost:8501`.
3. Confirm status indicator shows **Active / Indexed (6 Documents / 16 Chunks)**.

---

## Live Demonstration Script (Step-by-Step)

### Step 1: Open the "Ask" Interface (Landed State)
- **Action**: Direct attention to the sleek enterprise interface, search input field, and preset question chips.
- **Narrative**: *"Clause is an enterprise document assistant designed to answer policy questions with traceable evidence and strict refusal when data is missing."*

---

### Step 2: Semantic Policy Question (Supported)
- **Action**: Click or type:
  > *"Can an employee work from home while on probation?"*
- **Highlight**:
  - **Grounded Answer**: Note the clear answer citing probationary rules (first 90 days requires manager approval and HR endorsement).
  - **Inline Citation**: Point to the `[S1]` tag.
  - **Confidence Badge**: Highlight **High Confidence** status.
  - **Evidence Panel**: Expand to show the exact chunk from `Remote Work Policy.docx` (Section 2).

---

### Step 3: Exact Policy Code Search (Identifier Retrieval)
- **Action**: Click or type:
  > *"What does HR-RW-017 say?"*
- **Highlight**:
  - **Identifier Boosting**: Explain that Clause detects exact enterprise codes like `HR-RW-017` and boosts relevancy in sparse/dense fusion.
  - **Retrieved Document**: Confirms `Remote Work Policy.docx` (DOC-RW-001).

---

### Step 4: Security & Compliance Policy Inquiry
- **Action**: Click or type:
  > *"What are the MFA requirements?"*
- **Highlight**:
  - **Accuracy**: Retrieves `Information Security Policy.docx` (SEC-IS-002, Section 2).
  - **Rule Precision**: Cites mandatory 16-character password rules and mobile/hardware authenticators.

---

### Step 5: Finance & Receipts Inquiry
- **Action**: Click or type:
  > *"What receipts are required for expenses?"*
- **Highlight**:
  - **Policy Rule**: Answers that itemized receipts are strictly required for expenses exceeding $25 under `Expense Reimbursement Policy.docx`.

---

### Step 6: Unsupported Policy Refusal (Hallucination Prevention)
- **Action**: Click or type:
  > *"Does the company provide employees with free company cars?"*
- **Highlight**:
  - **Refusal Badge**: Point to **Insufficient Evidence** / **Refused** indicator.
  - **Standard Notice**: Highlight exact response:
    > *"I could not find sufficient information in the provided documents to answer this question."*
  - **Safety Guarantee**: Explain that Evidence Guard intercepted the query before calling Gemini, saving cost and guaranteeing zero hallucination.

---

### Step 7: Documents, Evaluation & System Tabs (Brief Overview)
- **Action 1**: Switch to **Documents** page — show 6 active policy files and manual document upload capability.
- **Action 2**: Switch to **Evaluation** page — show live benchmark metrics (**Hit@1: 94.4%**, **Hit@3: 100.0%**, **MRR: 0.963**, **Correct Refusal: 100.0%**).
- **Action 3**: Switch to **System** page — show active embedding model, FAISS index statistics, and memory health.
