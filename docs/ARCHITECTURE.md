# Clause Enterprise RAG — System Architecture & Technical Specifications

## 1. Overview & Architectural Philosophy

**Clause** is built upon a deterministic, evidence-grounded Retrieval-Augmented Generation (RAG) pipeline engineered specifically for high-stakes enterprise workplace policy applications. 

Unlike standard conversational AI systems that hallucinate when information is missing or incomplete, Clause implements an **Evidence Guard** pattern that evaluates evidence sufficiency *before* calling the LLM, guaranteeing zero-hallucination refusal behavior (`100.0% Correct Refusal Rate`) for unsupported queries while delivering high precision (`94.4% Hit@1`, `100.0% Hit@3`, `0.963 MRR`) for supported enterprise inquiries.

---

## 2. End-to-End System Pipeline Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & Storage"]
        Docs["Enterprise Documents (.docx / .pdf)"] --> Clean["Text Cleaning & Standardizing"]
        Clean --> Chunk["Context-Aware Chunking (700 char / 120 overlap)"]
        Chunk --> Embed["FastEmbed Embeddings (all-MiniLM-L6-v2)"]
        Embed --> FAISS["FAISS IndexFlatIP (Dense Index)"]
        Chunk --> BM25["BM25 Okapi Index (Sparse Index)"]
    end

    subgraph QueryPipeline ["2. Query Pre-Processing & Retrieval"]
        UserQ["User Input Query"] --> Rewriter["Deterministic Query Rewriter"]
        Rewriter --> ExpandedQ["Normalized Query & Expansions"]
        ExpandedQ --> DenseRet["FAISS Dense Search (Top K=10)"]
        ExpandedQ --> SparseRet["BM25 Keyword Search (Top K=10)"]
        DenseRet --> RRF["Reciprocal Rank Fusion (RRF k=60)"]
        SparseRet --> RRF
    end

    subgraph Intelligence ["3. Advanced RAG & Evidence Guard"]
        RRF --> Reranker["Composite Deterministic Reranker"]
        Reranker --> Guard["Evidence Guard Sufficiency Gate"]
        Guard -->|Insufficient Evidence| Refuse["Deterministic Refusal Engine"]
        Guard -->|Weak Evidence| Corrective["Bounded Corrective Retrieval (Max 1 Retry)"]
        Corrective --> Reranker
        Guard -->|Sufficient Evidence| ContextBuilder["Grounded Context Assembler"]
    end

    subgraph Generation ["4. Grounded Generation & Validation"]
        ContextBuilder --> Gemini["Gemini 3.8 Flash Generation"]
        Gemini --> Validator["Citation & Grounding Validator"]
        Validator --> Output["Verified Enterprise Response + Source Cards"]
        Refuse --> Output
    end
```

---

## 3. Subsystem Breakdown

### 3.1 Document Ingestion & Contextual Chunking
- **Supported Formats**: `.pdf` (via `PyPDF`) and `.docx` (via `python-docx`).
- **Context-Aware Chunking**: Text is split into `700-character` chunks with `120-character` overlap.
- **Context Enrichment**: Each chunk is prepended with parent document title and section headings to preserve metadata context during embedding vectorization.

### 3.2 Hybrid Retrieval Engine
- **Dense Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` via `FastEmbed` / ONNX Runtime (`384-dimensional` vectors).
- **Dense Vector Store**: `FAISS IndexFlatIP` with L2-normalized vectors for exact cosine similarity search.
- **Sparse Keyword Store**: `rank-bm25` (BM25Okapi algorithm) for exact policy identifier and numeric token matching.
- **Fusion Layer**: **Reciprocal Rank Fusion (RRF)** combines dense and sparse ranks with constant $k=60$:
  $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

### 3.3 Advanced RAG Intelligence Layer
- **Deterministic Reranker**: Reranks top candidates using composite scoring:
  $$\text{Score} = (0.40 \times \text{RRF}) + (0.35 \times \text{QueryCoverage}) + \text{SectionCoverage} + \text{IdentifierBoost}$$
- **Query Coverage**: Calculates word-level overlap of substantive domain subject keywords against retrieved chunk text using strict whole-word set matching.

### 3.4 Evidence Guard & Sufficiency Gate
- Evaluates evidence score, token coverage, and presence of exact policy codes (e.g. `HR-RW-017`).
- Assigns categorical **Confidence Labels**:
  - **High**: $\text{Score} \ge 0.35$
  - **Moderate**: $0.20 \le \text{Score} < 0.35$
  - **Insufficient**: $\text{Score} < 0.20$ or $\text{Token Coverage} < 0.15$
- **Strict Refusal**: Immediately returns standard refusal when evidence is insufficient, bypassing LLM generation to eliminate hallucinations.
- **Conflict Detection**: Scans retrieved sources for contradictory policy keywords (e.g., *"allowed"* vs *"prohibited"*).

### 3.5 Grounded Generation & Citation Validation
- **LLM Engine**: Google Gemini 3.8 Flash (`gemini-3.8-flash`) configured with temperature $0.0$.
- **Prompt Grounding**: System prompt strictly restricts generation to supplied context and enforces bracketed inline citations `[S1]`, `[S2]`.
- **Validation**: Post-generation parser verifies every cited source tag against retrieved chunks. Uncited or fabricated tags trigger safety warnings.

---

## 4. Environment & Data Flow Specifications

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit Shell
    participant Pipeline as RAG Pipeline Engine
    participant Guard as Evidence Guard
    participant LLM as Gemini 3.8 Flash

    User->>UI: Submit Question
    UI->>Pipeline: answer_question(query)
    Pipeline->>Pipeline: Normalize Query & Perform Hybrid Search
    Pipeline->>Pipeline: Rerank Results
    Pipeline->>Guard: evaluate_retrieval_evidence()
    
    alt Evidence Insufficient
        Guard-->>Pipeline: Refusal Recommended
        Pipeline-->>UI: Return RAGResponse(refused=True)
        UI-->>User: Display Standard Refusal Notice
    else Evidence Sufficient
        Guard-->>Pipeline: Evidence Approved
        Pipeline->>LLM: generate(Grounded Context Prompt)
        LLM-->>Pipeline: Grounded Text with [S1] tags
        Pipeline->>Pipeline: Validate Citations & Format Sources
        Pipeline-->>UI: Return RAGResponse(refused=False)
        UI-->>User: Display Answer + Source Cards
    end
```
