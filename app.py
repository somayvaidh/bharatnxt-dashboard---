"""
BharatNXT — Data-Driven SME FinTech Analytics Dashboard
Streamlit multi-page app | github → streamlit.io deployment
"""

import streamlit as st

st.set_page_config(
    page_title="BharatNXT Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:16px 20px;border:1px solid #e9ecef;}
.metric-val{font-size:28px;font-weight:700;color:#1a1a2e;}
.metric-lbl{font-size:13px;color:#6c757d;margin-top:2px;}
.section-tag{display:inline-block;font-size:11px;font-weight:600;padding:3px 10px;
             border-radius:12px;margin-bottom:8px;}
.tag-cls{background:#dbeafe;color:#1d4ed8;}
.tag-clus{background:#d1fae5;color:#065f46;}
.tag-arm{background:#fef3c7;color:#92400e;}
.tag-reg{background:#ede9fe;color:#5b21b6;}
.stTabs [data-baseweb="tab"]{font-weight:500;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.shields.io/badge/BharatNXT-Analytics-blue?style=for-the-badge",
    use_container_width=True,
)
st.sidebar.markdown("### Navigation")

PAGES = {
    "🏠  Overview & Data Health":       "page_overview",
    "📊  Descriptive Analytics":        "page_descriptive",
    "🔍  Diagnostic Analytics":         "page_diagnostic",
    "🎯  Clustering (K-Means)":         "page_clustering",
    "🔗  Association Rule Mining":      "page_arm",
    "🤖  Classification Models":        "page_classification",
    "📈  Regression — GTV Prediction":  "page_regression",
    "🚀  New Customer Scoring Engine":  "page_scoring",
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** 2,000 synthetic SME respondents  \n"
    "**Segments:** Real Estate + IT/Services  \n"
    "**Target:** Adoption intent (0–3)"
)

# ── Page routing ──────────────────────────────────────────────────────────────
page_module = PAGES[selection]

if page_module == "page_overview":
    from pages import page_overview as pg
elif page_module == "page_descriptive":
    from pages import page_descriptive as pg
elif page_module == "page_diagnostic":
    from pages import page_diagnostic as pg
elif page_module == "page_clustering":
    from pages import page_clustering as pg
elif page_module == "page_arm":
    from pages import page_arm as pg
elif page_module == "page_classification":
    from pages import page_classification as pg
elif page_module == "page_regression":
    from pages import page_regression as pg
elif page_module == "page_scoring":
    from pages import page_scoring as pg

pg.show()
