"""
CLAUSE — Enterprise Document Intelligence & Evidence-Grounded RAG Assistant
Main Streamlit Application Entrypoint.
"""

import streamlit as st

# Configure Streamlit page parameters
st.set_page_config(
    page_title="Clause — Enterprise Document Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui import (
    inject_custom_css,
    render_sidebar,
    render_ask_page,
    render_documents_page,
    render_evaluation_page,
    render_system_page,
)


def main():
    """Main application routing and layout rendering."""
    # Inject unified CSS design system
    inject_custom_css()

    # Render left navigation sidebar shell with unified KB status
    current_page = render_sidebar()

    # Route to active page module
    if current_page == "Ask":
        render_ask_page()
    elif current_page == "Documents":
        render_documents_page()
    elif current_page == "Evaluation":
        render_evaluation_page()
    elif current_page == "System":
        render_system_page()
    else:
        render_ask_page()


if __name__ == "__main__":
    main()
