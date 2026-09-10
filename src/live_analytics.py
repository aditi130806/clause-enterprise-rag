"""
Live Usage Analytics module for CLAUSE Enterprise RAG.
Persistently records and reads Ask-page user queries and response metrics to evaluation/live_usage.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import BASE_DIR
from src.models import RAGResponse

LIVE_USAGE_FILE = BASE_DIR / "evaluation" / "live_usage.jsonl"


def log_live_query_event(response: RAGResponse) -> None:
    """
    Append a structured analytics event to live_usage.jsonl when an Ask request completes.
    
    Args:
        response: Completed RAGResponse object.
    """
    try:
        LIVE_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine mode / response type
        if response.refused or response.confidence_label == "Insufficient":
            mode = "Refused"
        elif response.metadata.get("fallback") or "Fallback" in str(response.generation_model):
            mode = "Fallback"
        elif response.metadata.get("error") or "Execution exception" in " ".join(response.evidence_reasons):
            mode = "Error"
        else:
            mode = "Gemini"

        cited_count = len(response.sources)
        citation_valid = cited_count > 0 and all(s.excerpt and len(s.excerpt) > 5 for s in response.sources)
        
        total_ms = response.metadata.get("total_ms", 0.0)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": response.question.strip() if response.question else "",
            "mode": mode,
            "confidence": response.confidence_label or "Insufficient",
            "sources": cited_count,
            "citation_valid": citation_valid,
            "corrective_retrieval": response.retrieval_corrected,
            "conflict_detected": response.conflict_detected,
            "refused": response.refused or (mode == "Refused"),
            "latency_ms": round(float(total_ms), 1),
        }

        with open(LIVE_USAGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Analytics recording must be non-blocking and fail-safe
        pass


def load_live_query_events() -> List[Dict[str, Any]]:
    """
    Load all recorded live analytics query events from live_usage.jsonl.
    
    Returns:
        List of query event dictionaries (newest first).
    """
    if not LIVE_USAGE_FILE.exists():
        return []
    
    events = []
    try:
        with open(LIVE_USAGE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        events.append(json.loads(line_str))
                    except Exception:
                        continue
    except Exception:
        return []
    
    # Return newest events first
    events.reverse()
    return events


def compute_live_analytics_summary() -> Dict[str, Any]:
    """
    Compute aggregate live metrics from recorded query events.
    
    Returns:
        Dictionary containing aggregate count and average metrics.
    """
    events = load_live_query_events()
    total_queries = len(events)
    if total_queries == 0:
        return {
            "total_queries": 0,
            "answered_queries": 0,
            "refused_queries": 0,
            "fallback_responses": 0,
            "gemini_responses": 0,
            "citation_valid_responses": 0,
            "avg_latency_ms": 0.0,
            "events": [],
        }

    answered_count = sum(1 for e in events if not e.get("refused") and e.get("mode") in ["Gemini", "Fallback"])
    refused_count = sum(1 for e in events if e.get("refused") or e.get("mode") == "Refused")
    fallback_count = sum(1 for e in events if e.get("mode") == "Fallback")
    gemini_count = sum(1 for e in events if e.get("mode") == "Gemini")
    valid_citation_count = sum(1 for e in events if e.get("citation_valid"))
    
    latencies = [e.get("latency_ms", 0.0) for e in events if e.get("latency_ms", 0.0) > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "total_queries": total_queries,
        "answered_queries": answered_count,
        "refused_queries": refused_count,
        "fallback_responses": fallback_count,
        "gemini_responses": gemini_count,
        "citation_valid_responses": valid_citation_count,
        "avg_latency_ms": round(avg_latency, 1),
        "events": events,
    }


def clear_live_query_events() -> None:
    """
    Clear recorded live query events without removing benchmark or KB files.
    """
    try:
        if LIVE_USAGE_FILE.exists():
            LIVE_USAGE_FILE.unlink()
    except Exception:
        pass
