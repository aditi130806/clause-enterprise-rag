"""
Script to generate the benchmark RAG evaluation dataset (24 questions).
Saves dataset to evaluation/rag_evaluation_dataset.csv and evaluation/rag_evaluation_dataset.json.
"""

import csv
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"

DATASET_ROWS = [
    # Single-document Semantic (10)
    {
        "question_id": "Q01",
        "category": "single_doc_semantic",
        "question": "Can an employee work from home while on probation?",
        "expected_answer_or_key_fact": "Probationary employees may work remotely only upon receiving explicit written manager approval and HR endorsement under HR-RW-017.",
        "expected_source_document": "Remote Work Policy.docx",
        "expected_policy_id": "HR-RW-017",
        "expected_supported": True,
        "notes": "Paraphrased query testing probationary remote work rule.",
    },
    {
        "question_id": "Q02",
        "category": "single_doc_semantic",
        "question": "What is the maximum number of PTO days that can be carried over into the next year?",
        "expected_answer_or_key_fact": "A maximum of 5 accrued unused PTO days may be carried over into the following calendar year.",
        "expected_source_document": "Annual Leave & PTO Policy.docx",
        "expected_policy_id": "HR-LV-004",
        "expected_supported": True,
        "notes": "Query regarding PTO carry-over limits.",
    },
    {
        "question_id": "Q03",
        "category": "single_doc_semantic",
        "question": "What itemized receipts are required for business expense reimbursement?",
        "expected_answer_or_key_fact": "Itemized receipts are strictly required for all reimbursable expenses exceeding $25.",
        "expected_source_document": "Expense Reimbursement Policy.docx",
        "expected_policy_id": "FIN-EXP-011",
        "expected_supported": True,
        "notes": "Query regarding receipt requirement threshold.",
    },
    {
        "question_id": "Q04",
        "category": "single_doc_semantic",
        "question": "What length and complexity is required for employee account passwords?",
        "expected_answer_or_key_fact": "Passwords must be at least 16 characters long incorporating uppercase, lowercase, numbers, and special symbols.",
        "expected_source_document": "Information Security Policy.docx",
        "expected_policy_id": "SEC-IS-002",
        "expected_supported": True,
        "notes": "Query regarding password security policy.",
    },
    {
        "question_id": "Q05",
        "category": "single_doc_semantic",
        "question": "What are the mandatory core collaboration hours when employees must be reachable?",
        "expected_answer_or_key_fact": "Core collaboration hours are from 10:00 AM to 3:00 PM local time.",
        "expected_source_document": "Attendance & Flexible Work Hours Policy.docx",
        "expected_policy_id": "HR-AT-009",
        "expected_supported": True,
        "notes": "Query regarding core work hours.",
    },
    {
        "question_id": "Q06",
        "category": "single_doc_semantic",
        "question": "What percentage of monthly health insurance premiums does the company pay?",
        "expected_answer_or_key_fact": "The company pays 85% of monthly healthcare insurance premiums for full-time employees.",
        "expected_source_document": "Employee Benefits & Wellness Policy.docx",
        "expected_policy_id": "HR-BEN-006",
        "expected_supported": True,
        "notes": "Query regarding health insurance contribution.",
    },
    {
        "question_id": "Q07",
        "category": "single_doc_semantic",
        "question": "What internet speed is required for telecommuting employees?",
        "expected_answer_or_key_fact": "Employees must maintain high-speed internet with a minimum 50 Mbps download speed.",
        "expected_source_document": "Remote Work Policy.docx",
        "expected_policy_id": "HR-RW-017",
        "expected_supported": True,
        "notes": "Query regarding remote workspace requirements.",
    },
    {
        "question_id": "Q08",
        "category": "single_doc_semantic",
        "question": "How many paid sick days do full-time employees receive annually?",
        "expected_answer_or_key_fact": "Full-time employees receive 10 paid sick days per year.",
        "expected_source_document": "Annual Leave & PTO Policy.docx",
        "expected_policy_id": "HR-LV-004",
        "expected_supported": True,
        "notes": "Query regarding sick leave entitlement.",
    },
    {
        "question_id": "Q09",
        "category": "single_doc_semantic",
        "question": "Within how many days must an expense reimbursement claim be submitted?",
        "expected_answer_or_key_fact": "All reimbursement claims must be submitted within 30 calendar days of incurring the expense.",
        "expected_source_document": "Expense Reimbursement Policy.docx",
        "expected_policy_id": "FIN-EXP-011",
        "expected_supported": True,
        "notes": "Query regarding expense submission timeframe.",
    },
    {
        "question_id": "Q10",
        "category": "single_doc_semantic",
        "question": "What is the dollar limit for the annual employee wellness stipend?",
        "expected_answer_or_key_fact": "Full-time employees receive a $500 annual wellness stipend.",
        "expected_source_document": "Employee Benefits & Wellness Policy.docx",
        "expected_policy_id": "HR-BEN-006",
        "expected_supported": True,
        "notes": "Query regarding wellness stipend amount.",
    },

    # Exact Identifier / Keyword (4)
    {
        "question_id": "Q11",
        "category": "exact_identifier",
        "question": "What does policy HR-RW-017 state about remote work?",
        "expected_answer_or_key_fact": "HR-RW-017 governs telecommuting, hybrid workplace arrangements, equipment standards, and probationary employee eligibility.",
        "expected_source_document": "Remote Work Policy.docx",
        "expected_policy_id": "HR-RW-017",
        "expected_supported": True,
        "notes": "Exact policy ID match query.",
    },
    {
        "question_id": "Q12",
        "category": "exact_identifier",
        "question": "What are the MFA authentication rules in SEC-IS-002?",
        "expected_answer_or_key_fact": "Multi-Factor Authentication (MFA) is strictly mandatory for all employees accessing corporate systems, email, VPN, and cloud repositories.",
        "expected_source_document": "Information Security Policy.docx",
        "expected_policy_id": "SEC-IS-002",
        "expected_supported": True,
        "notes": "Exact policy ID and security term query.",
    },
    {
        "question_id": "Q13",
        "category": "exact_identifier",
        "question": "What are the non-reimbursable expense rules under FIN-EXP-011?",
        "expected_answer_or_key_fact": "Personal entertainment, parking fines, traffic violations, personal grooming, and unauthorized software subscriptions are non-reimbursable.",
        "expected_source_document": "Expense Reimbursement Policy.docx",
        "expected_policy_id": "FIN-EXP-011",
        "expected_supported": True,
        "notes": "Exact policy ID query for expense exceptions.",
    },
    {
        "question_id": "Q14",
        "category": "exact_identifier",
        "question": "What is the scope of policy HR-LV-004?",
        "expected_answer_or_key_fact": "HR-LV-004 outlines time-off entitlements, paid time off accrual schedules, sick leave, statutory holidays, and leave request procedures.",
        "expected_source_document": "Annual Leave & PTO Policy.docx",
        "expected_policy_id": "HR-LV-004",
        "expected_supported": True,
        "notes": "Exact policy ID query for PTO scope.",
    },

    # Multi-document (4)
    {
        "question_id": "Q15",
        "category": "multi_document",
        "question": "What security measures and approval steps are required when working remotely?",
        "expected_answer_or_key_fact": "Requires manager/HR approval for probation (HR-RW-017) and mandatory VPN with Multi-Factor Authentication (SEC-IS-002).",
        "expected_source_document": "Remote Work Policy.docx, Information Security Policy.docx",
        "expected_policy_id": "HR-RW-017, SEC-IS-002",
        "expected_supported": True,
        "notes": "Multi-document query spanning remote work rules and security requirements.",
    },
    {
        "question_id": "Q16",
        "category": "multi_document",
        "question": "What advance notice is required for taking annual leave versus booking business travel?",
        "expected_answer_or_key_fact": "Leave >3 days requires 14 days notice (HR-LV-004); domestic economy travel requires booking 14 days in advance (FIN-EXP-011).",
        "expected_source_document": "Annual Leave & PTO Policy.docx, Expense Reimbursement Policy.docx",
        "expected_policy_id": "HR-LV-004, FIN-EXP-011",
        "expected_supported": True,
        "notes": "Multi-document comparison query across leave and expense policies.",
    },
    {
        "question_id": "Q17",
        "category": "multi_document",
        "question": "How do flexible working hours interact with MFA security during remote work?",
        "expected_answer_or_key_fact": "Employees can adjust working hours between 7 AM and 6 PM but must remain reachable during core hours (HR-AT-009) and use MFA (SEC-IS-002).",
        "expected_source_document": "Attendance & Flexible Work Hours Policy.docx, Information Security Policy.docx",
        "expected_policy_id": "HR-AT-009, SEC-IS-002",
        "expected_supported": True,
        "notes": "Multi-document query across attendance and security policies.",
    },
    {
        "question_id": "Q18",
        "category": "multi_document",
        "question": "What benefits and remote work equipment options are available for new employees?",
        "expected_answer_or_key_fact": "Standard laptop and peripherals are provided (HR-RW-017); health coverage begins 1st of following month and $500 wellness stipend is available (HR-BEN-006).",
        "expected_source_document": "Remote Work Policy.docx, Employee Benefits & Wellness Policy.docx",
        "expected_policy_id": "HR-RW-017, HR-BEN-006",
        "expected_supported": True,
        "notes": "Multi-document query across equipment and benefits policies.",
    },

    # Unsupported / Insufficient-Information (6)
    {
        "question_id": "Q19",
        "category": "unsupported",
        "question": "Does the company provide employees with free company cars?",
        "expected_answer_or_key_fact": "Clause must refuse because company car policy is not in corpus (explicitly stated company does not provide company cars).",
        "expected_source_document": "None",
        "expected_policy_id": "None",
        "expected_supported": False,
        "notes": "Unsupported question testing car refusal.",
    },
    {
        "question_id": "Q20",
        "category": "unsupported",
        "question": "What is the corporate pet policy for bring-your-dog to work days?",
        "expected_answer_or_key_fact": "Clause must refuse because pet policy is not in corpus.",
        "expected_source_document": "None",
        "expected_policy_id": "None",
        "expected_supported": False,
        "notes": "Unsupported question regarding pet policy.",
    },
    {
        "question_id": "Q21",
        "category": "unsupported",
        "question": "What is the company policy for tuition reimbursement for university master's degrees?",
        "expected_answer_or_key_fact": "Clause must refuse because tuition reimbursement is not in corpus.",
        "expected_source_document": "None",
        "expected_policy_id": "None",
        "expected_supported": False,
        "notes": "Unsupported question regarding tuition reimbursement.",
    },
    {
        "question_id": "Q22",
        "category": "unsupported",
        "question": "Does the company reimburse employee stock option trading losses?",
        "expected_answer_or_key_fact": "Clause must refuse because stock trading loss reimbursement is not in corpus.",
        "expected_source_document": "None",
        "expected_policy_id": "None",
        "expected_supported": False,
        "notes": "Unsupported question regarding stock loss reimbursement.",
    },
    {
        "question_id": "Q23",
        "category": "unsupported",
        "question": "What are the company guidelines for sabbatical leaves exceeding 2 years?",
        "expected_answer_or_key_fact": "Clause must refuse because 2-year sabbatical policy is not in corpus.",
        "expected_source_document": "None",
        "expected_policy_id": "None",
        "expected_supported": False,
        "notes": "Unsupported question regarding long sabbaticals.",
    },
    {
        "question_id": "Q24",
        "category": "unsupported",
        "question": "What is the company policy regarding enterprise relocation bonuses to Tokyo?",
        "expected_answer_or_key_fact": "Clause must refuse because relocation bonus policy is not in corpus.",
        "expected_source_document": "None",
        "expected_policy_id": "None",
        "expected_supported": False,
        "notes": "Unsupported question regarding international relocation bonus.",
    },
]


def build_evaluation_dataset():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV
    csv_path = EVAL_DIR / "rag_evaluation_dataset.csv"
    fieldnames = [
        "question_id",
        "category",
        "question",
        "expected_answer_or_key_fact",
        "expected_source_document",
        "expected_policy_id",
        "expected_supported",
        "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(DATASET_ROWS)
    print(f"Created CSV dataset: {csv_path} ({len(DATASET_ROWS)} questions)")

    # 2. Save JSON
    json_path = EVAL_DIR / "rag_evaluation_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(DATASET_ROWS, f, indent=2)
    print(f"Created JSON dataset: {json_path}")


if __name__ == "__main__":
    build_evaluation_dataset()
