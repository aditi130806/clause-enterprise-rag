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

    # Track session state for page navigation scroll reset
    if "previous_page" not in st.session_state:
        st.session_state["previous_page"] = None

    # Render left navigation sidebar shell with unified KB status & mobile header
    current_page = render_sidebar()
    previous_page = st.session_state.get("previous_page")

    # One-time scroll reset against parent Streamlit document ONLY when page changes
    if previous_page is not None and previous_page != current_page:
        st.components.v1.html(
            """
            <script>
            (function() {
                function forceScrollTop() {
                    try {
                        var doc = window.parent.document;
                        if (!doc) return;
                        var selectors = [
                            '[data-testid="stAppViewContainer"]',
                            'section.main',
                            '[data-testid="stMain"]',
                            '.main',
                            '#root'
                        ];
                        selectors.forEach(function(sel) {
                            var el = doc.querySelector(sel);
                            if (el) {
                                el.scrollTop = 0;
                                if (el.scrollTo) {
                                    try { el.scrollTo({ top: 0, left: 0, behavior: 'instant' }); }
                                    catch(e) { el.scrollTo(0, 0); }
                                }
                            }
                        });
                        if (doc.scrollingElement) doc.scrollingElement.scrollTop = 0;
                        if (doc.documentElement) doc.documentElement.scrollTop = 0;
                        if (doc.body) doc.body.scrollTop = 0;
                        if (window.parent && window.parent.scrollTo) {
                            try { window.parent.scrollTo({ top: 0, left: 0, behavior: 'instant' }); }
                            catch(e) { window.parent.scrollTo(0, 0); }
                        }
                    } catch(err) {}
                }
                forceScrollTop();
                if (window.parent && window.parent.requestAnimationFrame) {
                    window.parent.requestAnimationFrame(forceScrollTop);
                }
                setTimeout(forceScrollTop, 10);
                setTimeout(forceScrollTop, 50);
                setTimeout(forceScrollTop, 150);
            })();
            </script>
            """,
            height=0,
        )

    # Always update previous_page after checking
    st.session_state["previous_page"] = current_page

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
