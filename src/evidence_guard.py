"""
Evidence Guard & Validation Service for CLAUSE RAG.
Evaluates evidence sufficiency, retrieval strength, source diversity,
conflict detection, and citation validity before final answer acceptance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import re

from src.models import RetrievalResult, SourceReference


@dataclass
class EvidenceGuardResult:
    """Diagnostic evidence analysis object produced by EvidenceGuard."""
    evidence_status: str  # "sufficient" | "moderate" | "weak" | "insufficient"
    confidence_label: str  # "High" | "Moderate" | "Low" | "Insufficient"
    supporting_source_count: int
    source_diversity: int
    citation_validity: bool
    strongest_source_score: float
    conflict_detected: bool
    refusal_recommended: bool
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_status": self.evidence_status,
            "confidence_label": self.confidence_label,
            "supporting_source_count": self.supporting_source_count,
            "source_diversity": self.source_diversity,
            "citation_validity": self.citation_validity,
            "strongest_source_score": float(self.strongest_source_score),
            "conflict_detected": self.conflict_detected,
            "refusal_recommended": self.refusal_recommended,
            "reasons": self.reasons,
        }


# Modal conflict keywords for lightweight conflict detection
OPPOSING_MODAL_PAIRS = [
    ({"allowed", "permitted", "eligible"}, {"prohibited", "forbidden", "ineligible", "not allowed", "not eligible"}),
    ({"required", "mandatory"}, {"optional", "voluntary", "discretionary"}),
    ({"provided", "free", "included"}, {"not provided", "excluded", "self-funded", "employee expense"}),
]


def detect_evidence_conflicts(results: List[RetrievalResult]) -> Tuple[bool, List[str]]:
    """
    Detect potential modal or factual conflicts across retrieved top evidence chunks.
    """
    if len(results) < 2:
        return False, []

    combined_texts = [r.chunk.text.lower() for r in results if r.chunk and r.chunk.text]
    reasons = []
    conflict_found = False

    for pos_set, neg_set in OPPOSING_MODAL_PAIRS:
        has_pos = any(any(p in text for p in pos_set) for text in combined_texts)
        has_neg = any(any(n in text for n in neg_set) for text in combined_texts)

        if has_pos and has_neg:
            conflict_found = True
            pos_match = next(p for p in pos_set if any(p in t for t in combined_texts))
            neg_match = next(n for n in neg_set if any(n in t for t in combined_texts))
            reasons.append(f"Potential modal conflict between terms '{pos_match}' and '{neg_match}' across retrieved sources.")

    return conflict_found, reasons


class EvidenceGuard:
    """
    Structured Evidence Guard evaluating evidence quality, confidence, and conflicts.
    """

    def __init__(
        self,
        min_sufficiency_score: float = 0.015,
        high_confidence_score: float = 0.15,
        moderate_confidence_score: float = 0.06,
    ):
        self.min_sufficiency_score = min_sufficiency_score
        self.high_confidence_score = high_confidence_score
        self.moderate_confidence_score = moderate_confidence_score

    def evaluate_retrieval_evidence(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        cited_sources: Optional[List[SourceReference]] = None,
    ) -> EvidenceGuardResult:
        """
        Evaluate retrieved evidence chunks before answer generation or acceptance.
        """
        reasons = []

        if not retrieved_results:
            return EvidenceGuardResult(
                evidence_status="insufficient",
                confidence_label="Insufficient",
                supporting_source_count=0,
                source_diversity=0,
                citation_validity=True,
                strongest_source_score=0.0,
                conflict_detected=False,
                refusal_recommended=True,
                reasons=["No document chunks retrieved for query."],
            )

        strongest_score = retrieved_results[0].fusion_score
        supporting_count = len(retrieved_results)
        unique_docs = len(set(r.document_name for r in retrieved_results if r.document_name))

        # Conflict Detection
        conflict_detected, conflict_reasons = detect_evidence_conflicts(retrieved_results)
        if conflict_detected:
            reasons.extend(conflict_reasons)

        # Citation Validity (if cited_sources provided)
        citation_validity = True
        if cited_sources is not None:
            # Check for non-empty excerpts and valid source IDs
            citation_validity = all(s.source_id and s.excerpt for s in cited_sources)
            if not citation_validity:
                reasons.append("Invalid or fabricated citation structure detected.")

        from src.reranker import calculate_query_coverage
        from src.query_rewriter import extract_policy_identifiers

        top_chunk_text = retrieved_results[0].chunk.text if retrieved_results[0].chunk else ""
        token_cov = calculate_query_coverage(query, top_chunk_text)
        policy_ids = extract_policy_identifiers(query)

        # Key subject term coverage check across all retrieved candidate chunks
        basic_stops = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did",
            "can", "will", "what", "how", "where", "when", "why", "who", "which",
            "company", "policy", "policies", "employee", "employees", "for", "of", "to", "in",
            "on", "at", "with", "or", "and", "regarding", "guidelines", "rules", "rule",
            "reimbursement", "reimbursements", "leave", "leaves", "days", "work"
        }
        key_terms = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query) if w.lower() not in basic_stops]
        all_retrieved_text = " ".join([r.chunk.text.lower() for r in retrieved_results if r.chunk and r.chunk.text])
        if key_terms:
            key_matches = sum(1 for kt in key_terms if kt in all_retrieved_text)
            key_term_cov = key_matches / len(key_terms)
        else:
            key_term_cov = 1.0

        # Determine Evidence Status & Confidence Label
        if strongest_score < self.min_sufficiency_score or (not policy_ids and (token_cov < 0.15 or key_term_cov < 0.35)):
            evidence_status = "insufficient"
            confidence_label = "Insufficient"
            refusal_recommended = True
            if key_term_cov < 0.35 and not policy_ids:
                reasons.append(f"Key subject terms coverage ({key_term_cov:.2f}) across retrieved documents is insufficient.")
            elif token_cov < 0.15 and not policy_ids:
                reasons.append(f"Query keyword coverage ({token_cov:.2f}) in retrieved evidence is insufficient for refusal safety.")
            else:
                reasons.append(f"Strongest evidence score ({strongest_score:.2f}) below sufficiency threshold ({self.min_sufficiency_score}).")
        elif strongest_score < self.moderate_confidence_score:
            evidence_status = "weak"
            confidence_label = "Low"
            refusal_recommended = True
            reasons.append(f"Evidence score ({strongest_score:.2f}) is weak and insufficient for guaranteed accuracy.")
        elif strongest_score < self.high_confidence_score:
            evidence_status = "moderate"
            confidence_label = "Moderate"
            refusal_recommended = False
            reasons.append(f"Evidence score ({strongest_score:.2f}) is moderate.")
        else:
            evidence_status = "sufficient"
            confidence_label = "High"
            refusal_recommended = False
            reasons.append(f"Strong evidence score ({strongest_score:.2f}) across {unique_docs} document(s).")

        return EvidenceGuardResult(
            evidence_status=evidence_status,
            confidence_label=confidence_label,
            supporting_source_count=supporting_count,
            source_diversity=unique_docs,
            citation_validity=citation_validity,
            strongest_source_score=strongest_score,
            conflict_detected=conflict_detected,
            refusal_recommended=refusal_recommended,
            reasons=reasons,
        )
