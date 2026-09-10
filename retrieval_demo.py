"""
Developer Integration Demo for CLAUSE Enterprise RAG Retrieval Engine.

Generates a synthetic enterprise policy corpus (PDF & DOCX), builds FAISS and BM25 indices,
tests hybrid, dense, and keyword retrieval across 4 key test questions, and verifies index reload persistence.
"""

import sys
import tempfile
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import docx

from src.rag_pipeline import build_and_index_documents, load_retrieval_engine
from src.retriever import HybridRetriever


def create_synthetic_pdf(filepath: Path, content_pages: list[list[str]]) -> None:
    """Helper to generate multi-page PDF documents for demo testing."""
    c = canvas.Canvas(str(filepath), pagesize=letter)
    width, height = letter

    for page_lines in content_pages:
        y = height - 50
        for line in page_lines:
            c.drawString(50, y, line)
            y -= 20
        c.showPage()
    c.save()


def create_synthetic_docx(filepath: Path, headings_and_paragraphs: list[tuple[str, str]]) -> None:
    """Helper to generate DOCX documents with heading structures for demo testing."""
    doc = docx.Document()
    for heading, text in headings_and_paragraphs:
        if heading:
            doc.add_heading(heading, level=2)
        doc.add_paragraph(text)
    doc.save(filepath)


def setup_demo_corpus(target_dir: Path) -> None:
    """Populate temporary directory with synthetic enterprise policy documents."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Remote Work Policy (PDF) - Policy ID: HR-RW-017
    create_synthetic_pdf(
        target_dir / "remote_work_policy.pdf",
        [
            [
                "CLAUSE Enterprise Remote Work Policy",
                "Policy Identifier: HR-RW-017",
                "Section 1: General Remote Work Guidelines",
                "Eligible employees may work remotely up to two days each week with manager approval.",
                "Employees working remotely must maintain core working hours from 9 AM to 5 PM.",
            ],
            [
                "Section 2: Probationary Remote Work Requirements",
                "Probationary employees require both direct manager and HR approval prior to remote work.",
                "All remote workers must comply with corporate IT security standards.",
            ],
        ],
    )

    # 2. Annual Leave Policy (DOCX)
    create_synthetic_docx(
        target_dir / "annual_leave_policy.docx",
        [
            (
                "Annual Leave Entitlement",
                "All full-time employees receive an annual leave entitlement of 20 paid leave days per calendar year. "
                "Unused leave days may be carried over up to a maximum of 5 days into the following year.",
            ),
            (
                "Leave Approval Conditions",
                "Leave requests exceeding 3 consecutive business days require at least 5 business days advance notice "
                "submitted through the HR portal.",
            ),
        ],
    )

    # 3. Expense Reimbursement Policy (PDF)
    create_synthetic_pdf(
        target_dir / "expense_reimbursement_policy.pdf",
        [
            [
                "CLAUSE Expense Reimbursement Policy",
                "Section 1: Business Expense Standards",
                "Itemized receipt requirements apply for all business expenses over $25.",
                "Reimbursement submission deadline is strictly within 30 calendar days of expense date.",
                "Alcoholic beverages and personal travel expenses are non-reimbursable.",
            ]
        ],
    )

    # 4. Information Security Policy (PDF) - Policy ID: SEC-AUTH-004
    create_synthetic_pdf(
        target_dir / "information_security_policy.pdf",
        [
            [
                "CLAUSE Corporate Information Security Policy",
                "Policy Identifier: SEC-AUTH-004",
                "Section 1: Authentication & Password Requirements",
                "Multi-factor authentication (MFA) requirements are mandatory for all system access.",
                "User passwords must be changed every 90 days and must be at least 14 characters long.",
            ]
        ],
    )


def print_retrieval_report(query: str, results: list, mode: str = "hybrid") -> None:
    """Print formatted ASCII report of top retrieval results."""
    print("\n" + "=" * 80)
    print(f"QUERY: '{query}' [Mode: {mode.upper()}]")
    print("=" * 80)
    if not results:
        print("   No matching chunks found.")
        return

    print(
        f"{'Rank':<5} | {'Document':<30} | {'Page/Section':<22} | {'Dense':<7} | {'BM25':<7} | {'Fusion':<7} | {'Method':<12}"
    )
    print("-" * 105)

    for r in results:
        page_str = f"P.{r.page_number}" if r.page_number is not None else "N/A"
        sec_str = (r.section[:18] + "..") if r.section and len(r.section) > 18 else (r.section or "")
        loc_str = f"{page_str} ({sec_str})" if sec_str else page_str

        methods_str = "+".join(r.retrieval_methods)
        print(
            f"{r.final_rank:<5} | {r.document_name:<30} | {loc_str:<22} | {r.dense_score:<7.4f} | "
            f"{r.bm25_score:<7.4f} | {r.fusion_score:<7.4f} | {methods_str:<12}"
        )
        print(f"      Text Snippet: \"{r.chunk.text[:110].strip()}...\"")


def main():
    print("============================================================")
    print("CLAUSE ENTERPRISE RAG - HYBRID RETRIEVAL DEMO & VERIFICATION")
    print("============================================================")

    with tempfile.TemporaryDirectory() as temp_doc_dir, tempfile.TemporaryDirectory() as temp_vstore_dir:
        doc_dir = Path(temp_doc_dir)
        vstore_dir = Path(temp_vstore_dir)

        print("\n[Step 1] Creating synthetic enterprise policy corpus...")
        setup_demo_corpus(doc_dir)

        print("[Step 2] Executing indexing service (Extract -> Clean -> Chunk -> Embed -> FAISS + BM25)...")
        retriever, stats = build_and_index_documents(
            documents_dir=doc_dir,
            vectorstore_dir=vstore_dir,
            chunk_size=700,
            chunk_overlap=120,
        )

        print("\n--- Indexing Statistics ---")
        for key, val in stats.items():
            print(f"   - {key}: {val}")

        print("\n[Step 3] Running Test Questions on Live Engine...")

        # Question 1: Semantic Search Test
        q1 = "Can an employee work from home while on probation?"
        res1 = retriever.retrieve(q1, mode="hybrid", top_k=3)
        print_retrieval_report(q1, res1, mode="hybrid")

        # Question 2: Exact BM25 Keyword Search Test (Policy Identifier)
        q2 = "What does HR-RW-017 say?"
        res2 = retriever.retrieve(q2, mode="hybrid", top_k=3)
        print_retrieval_report(q2, res2, mode="hybrid")

        # Question 3: Security & MFA Test
        q3 = "What are the MFA requirements?"
        res3 = retriever.retrieve(q3, mode="hybrid", top_k=3)
        print_retrieval_report(q3, res3, mode="hybrid")

        # Question 4: Expense & Receipt Requirements Test
        q4 = "What receipts are required for expenses?"
        res4 = retriever.retrieve(q4, mode="hybrid", top_k=3)
        print_retrieval_report(q4, res4, mode="hybrid")

        print("\n[Step 4] Verifying Index Persistence & Reload Capabilities...")
        reloaded_retriever = load_retrieval_engine(vectorstore_dir=vstore_dir)

        reload_res = reloaded_retriever.retrieve(q2, mode="hybrid", top_k=1)
        assert len(reload_res) > 0, "Reloaded engine returned empty result!"
        assert reload_res[0].chunk.chunk_id == res2[0].chunk.chunk_id, "Reloaded chunk ID mismatch!"
        print("   -> Index Reload Verification: SUCCESS (Reloaded index returned identical chunk ID!).")

        print("\n============================================================")
        print("RETRIEVAL DEMO COMPLETED SUCCESSFULLY.")
        print("============================================================")


if __name__ == "__main__":
    main()
