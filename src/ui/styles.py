"""
Strict Brand & Layout Styling for CLAUSE Enterprise RAG Shell.
Enforces responsive, cohesive design tokens, typography, and container layout.
"""

import streamlit as st


def inject_custom_css():
    """Inject unified CSS rules for the entire CLAUSE application."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Design Tokens */
    :root {
        --bg-main: #F7F6F2;
        --bg-sidebar: #F2F1ED;
        --bg-surface: #FFFFFF;
        --text-primary: #1C1E1C;
        --text-secondary: #6E716D;
        --color-forest: #24533F;
        --color-forest-hover: #1E4636;
        --color-active-tint: #E5EEE8;
        --border-color: #E3E3DE;
        --badge-success-bg: #EAF2EC;
        --badge-success-text: #24533F;
        --badge-warning-bg: #F8F2E6;
        --badge-warning-text: #8A561E;
        --badge-error-bg: #F9EBEA;
        --badge-error-text: #8C322B;
        --font-serif: 'Playfair Display', Georgia, serif;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Global Overflow & Box Model Protection */
    *, *::before, *::after {
        box-sizing: border-box !important;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        overflow-x: hidden !important;
        background-color: var(--bg-main) !important;
        font-family: var(--font-sans) !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Streamlit Chrome Hiding */
    #MainMenu, header, footer, .stDeployButton, div[data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Desktop Centered Container */
    .block-container {
        max-width: 1240px !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-top: 24px !important;
        padding-left: 32px !important;
        padding-right: 32px !important;
        padding-bottom: 48px !important;
    }

    /* Sidebar Constraints & Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
        min-width: 230px !important;
        max-width: 236px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        padding-bottom: 1.25rem !important;
    }

    /* Navigation Buttons */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        background-color: transparent !important;
        color: var(--text-primary) !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        text-align: left !important;
        padding: 0 12px !important;
        height: 42px !important;
        font-family: var(--font-sans) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        margin-bottom: 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-shadow: none !important;
        transition: all 0.15s ease-in-out !important;
    }

    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: var(--color-active-tint) !important;
        color: var(--color-forest) !important;
    }

    div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: var(--color-active-tint) !important;
        color: var(--color-forest) !important;
        border-left: 3px solid var(--color-forest) !important;
        font-weight: 600 !important;
    }

    /* Typography Scale */
    .page-title {
        font-family: var(--font-serif) !important;
        font-size: clamp(30px, 3vw, 42px) !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0 0 4px 0 !important;
        line-height: 1.25 !important;
        letter-spacing: -0.01em !important;
    }

    .page-subtitle {
        font-family: var(--font-sans) !important;
        font-size: 15px !important;
        color: var(--text-secondary) !important;
        margin: 0 0 24px 0 !important;
        line-height: 1.5 !important;
    }

    .eyebrow-text {
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--color-forest) !important;
        margin-bottom: 4px !important;
    }

    .section-title {
        font-family: var(--font-sans) !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 24px 0 12px 0 !important;
    }

    /* Card System */
    .clause-card {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
        box-shadow: none !important;
    }

    .clause-compact-card {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 14px 16px !important;
        margin-bottom: 12px !important;
    }

    /* Empty States */
    .empty-state-box {
        background-color: var(--bg-surface) !important;
        border: 1px dashed var(--border-color) !important;
        border-radius: 10px !important;
        padding: 28px 20px !important;
        text-align: center !important;
        min-height: 130px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 16px 0 !important;
    }

    /* Badges */
    .badge-pill {
        font-size: 12px !important;
        font-weight: 600 !important;
        padding: 3px 10px !important;
        border-radius: 12px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        line-height: 1 !important;
    }

    .badge-success {
        background-color: var(--badge-success-bg) !important;
        color: var(--badge-success-text) !important;
    }

    .badge-warning {
        background-color: var(--badge-warning-bg) !important;
        color: var(--badge-warning-text) !important;
    }

    .badge-error {
        background-color: var(--badge-error-bg) !important;
        color: var(--badge-error-text) !important;
    }

    /* Inline Citation Tags */
    .citation-tag {
        color: var(--color-forest) !important;
        font-weight: 600 !important;
        background-color: var(--color-active-tint) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 13px !important;
        border: 1px solid #D0E3D7 !important;
        display: inline-block !important;
        margin: 0 2px !important;
    }

    /* Input & Search Customization */
    .stTextInput > div > div > input {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        font-size: 15px !important;
        height: 48px !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--color-forest) !important;
        box-shadow: 0 0 0 2px rgba(36, 83, 63, 0.12) !important;
    }

    /* Button Styling Overrides */
    .stButton > button[kind="primary"] {
        background-color: var(--color-forest) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        height: 42px !important;
        transition: background-color 0.15s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: var(--color-forest-hover) !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="secondary"] {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 10px 18px !important;
        height: 42px !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: var(--color-active-tint) !important;
        border-color: var(--color-forest) !important;
        color: var(--color-forest) !important;
    }

    /* Sample Chip Buttons */
    .chip-btn button {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
        height: auto !important;
        min-height: 38px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }

    .chip-btn button *,
    .chip-btn button div[data-testid="stMarkdownContainer"],
    .chip-btn button div[data-testid="stMarkdownContainer"] p,
    .chip-btn button p,
    .chip-btn button div,
    .chip-btn button span {
        white-space: normal !important;
        word-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        line-height: 1.35 !important;
    }

    .chip-btn button:hover {
        background-color: var(--color-active-tint) !important;
        color: var(--color-forest) !important;
        border-color: var(--color-forest) !important;
    }

    .chip-toggle-btn button {
        background-color: transparent !important;
        color: var(--color-forest) !important;
        border: 1px solid #D0E3D7 !important;
        font-weight: 600 !important;
    }

    .chip-toggle-btn button:hover {
        background-color: var(--color-active-tint) !important;
        border-color: var(--color-forest) !important;
    }

    /* Table System */
    .clause-table, div[data-testid="stTable"] {
        width: 100% !important;
        border-collapse: collapse;
        margin: 12px 0;
        font-size: 14px;
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow-x: auto !important;
    }

    .clause-table th {
        text-align: left;
        padding: 10px 16px;
        background-color: #EFEFEA;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 13px;
        border-bottom: 1px solid var(--border-color);
    }

    .clause-table td {
        padding: 12px 16px;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    .clause-table tr:last-child td {
        border-bottom: none;
    }

    /* Streamlit Alert Overrides (Remove default blue) */
    .stAlert {
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }

    /* Minimal Footer */
    .clause-minimal-footer {
        margin-top: 48px;
        padding-top: 16px;
        border-top: 1px solid var(--border-color);
        font-size: 12px;
        color: var(--text-secondary);
        text-align: center;
    }

    /* Mobile Header & Responsiveness (max-width: 768px vs min-width: 769px) */
    .mobile-header-bar {
        display: none;
    }

    @media (min-width: 769px) {
        .mobile-header-bar,
        .mobile-logo-wrap,
        .mobile-menu-card,
        div[data-testid="stColumn"]:has(button[key="mob_menu_toggle_btn"]),
        div[data-testid="element-container"]:has(.mobile-header-bar),
        div[data-testid="element-container"]:has(.mobile-menu-card),
        div[data-testid="element-container"]:has(.mobile-logo-wrap) {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
    }

    @media (max-width: 768px) {
        .mobile-header-bar {
            display: block !important;
            visibility: visible !important;
            width: 100% !important;
            margin-bottom: 16px !important;
        }

        .mobile-logo-wrap {
            display: flex !important;
            visibility: visible !important;
            align-items: center !important;
            height: 44px !important;
        }

        .mobile-logo-wrap svg {
            max-width: 130px !important;
            height: 36px !important;
        }

        .mobile-menu-card {
            display: block !important;
            visibility: visible !important;
            background-color: var(--bg-surface) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            padding: 12px !important;
            margin-top: 8px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
        }

        .block-container {
            padding-top: 16px !important;
            padding-left: 16px !important;
            padding-right: 16px !important;
            padding-bottom: 32px !important;
        }

        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        .chip-btn button {
            width: 100% !important;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
