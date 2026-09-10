"""
Page 3: Evaluation & System Intelligence.
Renders fixed 24-question benchmark metrics and dynamic runtime live usage analytics.
"""

import json
from pathlib import Path
import streamlit as st

from src.config import BASE_DIR
from src.evaluation import evaluate_retrieval_and_rag
from src.kb_state import get_knowledge_base_status, compute_kb_fingerprint
from src.live_analytics import (
    compute_live_analytics_summary,
    clear_live_query_events,
)
from src.ui.shell import render_page_header, render_footer
from src.ui.components import render_metric_card, render_empty_state


def render_evaluation_page():
    """Render professional two-layer Evaluation page (Benchmark Evaluation + Live Usage Analytics)."""
    render_page_header(
        title="Evaluation",
        subtitle="Benchmark quality and live runtime analytics.",
    )

    eval_file = BASE_DIR / "evaluation" / "evaluation_results.json"
    kb_status = get_knowledge_base_status()
    current_fp = compute_kb_fingerprint()

    eval_data = None
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            eval_data = None

    stored_fp = eval_data.get("knowledge_base_fingerprint") if eval_data else None
    is_stale = eval_data is None or stored_fp != current_fp

    # Automatic Benchmark Re-evaluation when Knowledge Base has changed
    if is_stale and kb_status.is_ready and kb_status.retriever is not None:
        last_attempted = st.session_state.get("auto_eval_attempted_fp")
        if last_attempted != current_fp:
            st.session_state["auto_eval_attempted_fp"] = current_fp
            st.info("Knowledge base changed. Refreshing benchmark evaluation metrics...")
            try:
                with st.spinner("Running benchmark evaluation..."):
                    eval_data = evaluate_retrieval_and_rag(
                        retriever=kb_status.retriever,
                        llm_service=None,
                        eval_llm_limit=0,
                        kb_fingerprint=current_fp,
                    )
                is_stale = False
            except Exception:
                st.warning("Benchmark evaluation could not be refreshed. Previous results may be outdated.")

    # ============================================================
    # SECTION A: BENCHMARK EVALUATION
    # ============================================================
    st.markdown(
        '<div style="margin-top: 12px; margin-bottom: 4px;">'
        '<div style="font-size: 18px; font-weight: 700; color: #1C1E1C;">Benchmark Evaluation</div>'
        '<div style="font-size: 13px; color: #6E716D;">Performance on the fixed 24-question ground-truth evaluation set.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 4 Benchmark Primary Metric Cards
    e1, e2, e3, e4 = st.columns(4)

    if eval_data:
        hit1 = eval_data.get("retrieval_hit_at_1", 0.0) * 100
        src_acc = eval_data.get("expected_source_accuracy", 0.0) * 100
        grounding = eval_data.get("grounding_validation_rate", 0.0) * 100
        refusal = eval_data.get("correct_refusal_rate", 0.0) * 100

        with e1:
            render_metric_card("Retrieval Precision", f"{hit1:.1f}%", "Hit@1 Benchmark")
        with e2:
            render_metric_card("Source Accuracy", f"{src_acc:.1f}%", "Expected Source Match")
        with e3:
            render_metric_card("Groundedness", f"{grounding:.1f}%", "Citation Grounding Rate")
        with e4:
            render_metric_card("Correct Refusal Rate", f"{refusal:.1f}%", "Unsupported Question Refusal")
    else:
        with e1:
            render_metric_card("Retrieval Precision", "—", "Not evaluated yet")
        with e2:
            render_metric_card("Source Accuracy", "—", "Not evaluated yet")
        with e3:
            render_metric_card("Groundedness", "—", "Not evaluated yet")
        with e4:
            render_metric_card("Correct Refusal Rate", "—", "Not evaluated yet")

    # Benchmark Results Sub-Header & Optional Stale Badge
    if is_stale and eval_data is not None:
        st.markdown(
            '<div style="display: flex; align-items: center; justify-content: space-between; margin-top: 20px; margin-bottom: 10px;">'
            '<div style="font-size: 14px; font-weight: 600; color: #1C1E1C;">Benchmark Results</div>'
            '<span style="background-color: #FEF3C7; color: #92400E; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 12px;">● Outdated (KB Modified)</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div style='font-size: 14px; font-weight: 600; color: #1C1E1C; margin-top: 20px; margin-bottom: 10px;'>Benchmark Results</div>", unsafe_allow_html=True)

    if eval_data:
        hit1_str = f"{eval_data.get('retrieval_hit_at_1', 0.0) * 100:.1f}%"
        hit3_str = f"{eval_data.get('retrieval_hit_at_3', 0.0) * 100:.1f}%"
        mrr_str = f"{eval_data.get('mrr', 0.0):.3f}"
        src_acc_str = f"{eval_data.get('expected_source_accuracy', 0.0) * 100:.1f}%"
        refusal_str = f"{eval_data.get('correct_refusal_rate', 0.0) * 100:.1f}%"
        grounding_str = f"{eval_data.get('grounding_validation_rate', 0.0) * 100:.1f}%"
        total_q_str = str(eval_data.get("total_questions", 24))
        supp_q_str = str(eval_data.get("supported_questions", 18))
        unsupp_q_str = str(eval_data.get("unsupported_questions", 6))

        table_rows = [
            {"Metric": "Retrieval Hit@1", "Result": hit1_str},
            {"Metric": "Retrieval Hit@3", "Result": hit3_str},
            {"Metric": "MRR", "Result": mrr_str},
            {"Metric": "Expected Source Accuracy", "Result": src_acc_str},
            {"Metric": "Correct Refusal Rate", "Result": refusal_str},
            {"Metric": "Grounding Validation Rate", "Result": grounding_str},
            {"Metric": "Total Questions", "Result": total_q_str},
            {"Metric": "Supported Questions", "Result": supp_q_str},
            {"Metric": "Unsupported Questions", "Result": unsupp_q_str},
        ]
        st.table(table_rows)
    else:
        render_empty_state(
            title="No evaluation results available.",
            subtitle="Please add and index documents on the Documents page first.",
        )

    bcol1, _ = st.columns([1, 2])
    with bcol1:
        if st.button("Run benchmark", type="primary", use_container_width=True):
            if not kb_status.is_ready or kb_status.retriever is None:
                st.error("Cannot run benchmark: Knowledge base is not indexed.")
            else:
                try:
                    with st.spinner("Running benchmark evaluation..."):
                        evaluate_retrieval_and_rag(
                            retriever=kb_status.retriever,
                            llm_service=None,
                            eval_llm_limit=0,
                            kb_fingerprint=current_fp,
                        )
                    st.session_state["auto_eval_attempted_fp"] = current_fp
                    st.success("Benchmark evaluation completed successfully.")
                    st.rerun()
                except Exception:
                    st.error("Evaluation could not be refreshed. Previous results may be outdated.")

    st.markdown("<hr style='margin: 32px 0; border: none; border-top: 1px solid #E3E3DE;' />", unsafe_allow_html=True)

    # ============================================================
    # SECTION B: LIVE USAGE ANALYTICS
    # ============================================================
    st.markdown(
        '<div style="margin-bottom: 16px;">'
        '<div style="font-size: 18px; font-weight: 700; color: #1C1E1C;">Live Usage Analytics</div>'
        '<div style="font-size: 13px; color: #6E716D;">Runtime behavior from questions asked through Clause.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    summary = compute_live_analytics_summary()
    events = summary.get("events", [])

    # Dynamic Live Metric Cards
    l1, l2, l3, l4, l5, l6 = st.columns(6)
    with l1:
        render_metric_card("Total Queries", str(summary["total_queries"]), "Ask-page count")
    with l2:
        render_metric_card("Answered", str(summary["answered_queries"]), "Grounded responses")
    with l3:
        render_metric_card("Refused", str(summary["refused_queries"]), "Refusal triggers")
    with l4:
        render_metric_card("Fallbacks", str(summary["fallback_responses"]), "Extractive fallback")
    with l5:
        render_metric_card("Citation Valid", str(summary["citation_valid_responses"]), "Valid citations")
    with l6:
        lat_str = f"{summary['avg_latency_ms']:.0f} ms" if summary['avg_latency_ms'] > 0 else "—"
        render_metric_card("Avg Latency", lat_str, "Mean request time")

    st.markdown("<div style='font-size: 14px; font-weight: 600; color: #1C1E1C; margin-top: 24px; margin-bottom: 10px;'>Recent Queries</div>", unsafe_allow_html=True)

    if events:
        table_records = []
        for e in events:
            mode_val = e.get("mode", "Gemini")
            res_str = "Refused" if e.get("refused") else "Answered"
            sources_val = str(e.get("sources", 0))
            lat_val = f"{e.get('latency_ms', 0):.0f} ms" if e.get("latency_ms", 0) > 0 else "N/A"
            q_text = e.get("question", "")
            if len(q_text) > 65:
                q_text = q_text[:65] + "..."

            table_records.append({
                "Question": q_text,
                "Result": res_str,
                "Confidence": e.get("confidence", "Insufficient"),
                "Sources": sources_val,
                "Mode": mode_val,
                "Latency": lat_val,
            })

        st.table(table_records)

        lcol1, _ = st.columns([1, 2])
        with lcol1:
            if st.button("Clear live analytics", use_container_width=True):
                clear_live_query_events()
                st.success("Live usage analytics cleared.")
                st.rerun()
    else:
        render_empty_state(
            title="No live queries recorded yet.",
            subtitle="Ask questions on the Ask Clause page to see runtime analytics here.",
        )

    render_footer()

