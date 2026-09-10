# Clause Enterprise RAG — Visual Screenshot Catalog Guide

This directory is reserved for capturing and embedding production screenshots of the Clause Streamlit web application interface.

## Recommended Screenshots to Capture Manually

When launching the Clause interface (`streamlit run app.py`), capture the following high-resolution PNG screenshots into this directory:

1. **`01-ask-home.png`**:
   - Initial landed state of the **Ask** page showing search bar, quick example query chips, and active enterprise policy statistics.

2. **`02-grounded-answer.png`**:
   - Answer state for query *"Can an employee work from home while on probation?"* showing the Gemini answer, confidence badge ("High Confidence"), and inline `[S1]` source citation tag.

3. **`03-evidence-panel.png`**:
   - Expanded Evidence Panel showing source chunk metadata, exact policy identifier (`HR-RW-017`), page number, and fusion relevance score.

4. **`04-documents.png`**:
   - **Documents** page showing the indexed policy corpus inventory, total vector store chunk count, and manual upload/re-indexing interface.

5. **`05-evaluation.png`**:
   - **Evaluation** page displaying summary benchmark metrics (Hit@1: 94.4%, Hit@3: 100.0%, MRR: 0.963, Correct Refusal: 100.0%) and the interactive 24-question test result table.

6. **`06-system.png`**:
   - **System** page displaying runtime system status, model configurations, memory usage, and active retrieval pipeline components.
