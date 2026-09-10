"""
Deterministic Query Rewriting and Expansion Utility for CLAUSE RAG.
Normalizes policy abbreviations, preserves exact document/policy identifiers,
and expands domain terms without external LLM calls.
"""

import re
from typing import List, Optional


# Exact policy/code pattern (e.g., HR-RW-017, SEC-IS-002, FIN-EXP-011)
POLICY_ID_PATTERN = re.compile(r"\b[A-Z0-9]{2,8}(?:-[A-Z0-9]{2,8})+\b")

# Common enterprise policy domain expansions
DOMAIN_EXPANSIONS = {
    "wfh": "work from home remote work",
    "mfa": "multi-factor authentication MFA",
    "probation": "probation probationary employee eligibility",
    "receipt": "receipt expense reimbursement invoice proof",
    "receipts": "receipts expense reimbursements invoices proof",
    "car": "company car vehicle automobile transport",
    "cars": "company cars vehicles automobiles transport",
    "pto": "paid time off vacation leave PTO",
    "vpn": "virtual private network vpn secure connection",
    "meal": "meal allowance per diem business meal limit",
    "meals": "meals allowance per diem business meal limit",
    "allowance": "allowance per diem daily meal limit rate",
    "per diem": "per diem daily meal allowance limit",
}

# Conversational filler prefixes to trim for keyword focus
FILLER_PREFIXES = [
    r"^can an employee\b",
    r"^can I\b",
    r"^what does\b",
    r"^what are the\b",
    r"^does the company\b",
    r"^is it allowed to\b",
    r"^tell me about\b",
]


def extract_policy_identifiers(query: str) -> List[str]:
    """Extract exact policy or document identifiers like HR-RW-017."""
    return POLICY_ID_PATTERN.findall(query)


def normalize_query(query: str) -> str:
    """
    Clean and normalize user query while preserving exact identifiers.
    Removes conversational filler and expands known policy abbreviations.
    """
    if not query:
        return ""

    # Preserve exact policy IDs
    policy_ids = extract_policy_identifiers(query)

    cleaned = query.strip()
    
    # Strip conversational filler prefixes
    for prefix in FILLER_PREFIXES:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()

    tokens = cleaned.split()
    expanded_tokens = []

    for token in tokens:
        clean_token = re.sub(r"[^\w\-]", "", token).lower()
        if clean_token in DOMAIN_EXPANSIONS:
            expanded_tokens.append(DOMAIN_EXPANSIONS[clean_token])
        else:
            expanded_tokens.append(token)

    # Re-insert exact policy IDs if stripped or modified
    result = " ".join(expanded_tokens)
    for pid in policy_ids:
        if pid not in result:
            result = f"{pid} {result}"

    return result.strip()


def expand_query_for_corrective_retrieval(query: str) -> str:
    """
    Generate an expanded query string for ONE bounded corrective retrieval attempt.
    """
    normalized = normalize_query(query)
    policy_ids = extract_policy_identifiers(query)
    
    # Add domain-specific terms based on query intent
    additional_terms = []
    query_lower = query.lower()

    if any(k in query_lower for k in ["probation", "wfh", "remote", "telecommute"]):
        additional_terms.extend(["remote work policy", "probationary period eligibility", "telecommuting"])
    elif any(k in query_lower for k in ["mfa", "security", "password", "vpn", "authentication"]):
        additional_terms.extend(["information security policy", "authentication requirements", "multi factor"])
    elif any(k in query_lower for k in ["receipt", "expense", "reimbursement", "per diem"]):
        additional_terms.extend(["travel expense policy", "itemized receipt", "reimbursement claim"])
    elif any(k in query_lower for k in ["leave", "pto", "vacation", "holiday"]):
        additional_terms.extend(["annual leave policy", "pto accrual"])
    elif any(k in query_lower for k in ["benefit", "wellness", "stipend", "insurance"]):
        additional_terms.extend(["employee benefits policy", "wellness stipend"])

    combined = f"{normalized} {' '.join(additional_terms)} {' '.join(policy_ids)}"
    
    # Deduplicate words while preserving order
    seen = set()
    deduped = []
    for word in combined.split():
        w_lower = word.lower()
        if w_lower not in seen:
            seen.add(w_lower)
            deduped.append(word)

    return " ".join(deduped)
