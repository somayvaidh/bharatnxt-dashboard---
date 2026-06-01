"""Page 0 — Overview and Data Health"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, INTENT_LABELS, INTENT_COLORS

def show():
    st.title("🏠 BharatNXT — SME FinTech Analytics Platform")
    st.markdown(
        "**Founder's dashboard** — data-driven acquisition, segmentation, and "
        "conversion intelligence for India's B2B credit payment platform."
    )

    df = load_data()

    # ── KPI row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(df)
    interested = (df["adoption_intent"] >= 2).sum()
    high_intent = (df["adoption_intent"] == 3).sum()
    missing_pct = round(df.isnull().mean().mean() * 100, 1)
    outlier_rows = 40

    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [total, interested, high_intent, f"{missing_pct}%", outlier_rows],
        ["Total respondents", "Some interest (2–3)", "High intent (3)", "Missing values", "Outlier rows"],
    ):
        col.markdown(
            f'<div class="metric-card"><div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Dataset snapshot", "🏥 Data health", "🎭 Persona distribution"])

    with tab1:
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(20), use_container_width=True, height=380)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column types**")
            dtypes = df.dtypes.value_counts().reset_index()
            dtypes.columns = ["dtype", "count"]
            st.dataframe(dtypes, use_container_width=True)
        with col2:
            st.markdown("**Target class distribution**")
            vc = df["adoption_intent"].value_counts().sort_index().reset_index()
            vc.columns = ["class", "count"]
            vc["label"] = vc["class"].map(INTENT_LABELS)
            vc["pct"] = (vc["count"] / total * 100).round(1)
            st.dataframe(vc[["label","count","pct"]], use_container_width=True)

    with tab2:
        st.markdown("#### Missing value analysis")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing):
            fig = px.bar(
                x=missing.index, y=missing.values,
                labels={"x": "Column", "y": "Missing count"},
                color=missing.values,
                color_continuous_scale="Reds",
                title="Missing values per column"
            )
            fig.update_layout(showlegend=False, height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values detected.")

        st.markdown("#### Class balance check")
        vc2 = df["adoption_intent"].value_counts().sort_index()
        fig2 = go.Figure(go.Bar(
            x=[INTENT_LABELS[i] for i in vc2.index],
            y=vc2.values,
            marker_color=[INTENT_COLORS[i] for i in vc2.index],
            text=vc2.values, textposition="outside",
        ))
        fig2.update_layout(
            title="Target variable distribution (imbalanced by design)",
            height=320, yaxis_title="Count", xaxis_title="Adoption intent"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.info(
            "⚖️ Class imbalance is intentional — mirrors real Indian FinTech survey benchmarks. "
            "The Classification page applies `class_weight='balanced'` to handle this."
        )

    with tab3:
        st.markdown("#### Persona distribution")
        pc = df["persona_label"].value_counts().reset_index()
        pc.columns = ["Persona", "Count"]
        pc["Pct"] = (pc["Count"] / total * 100).round(1)

        fig3 = px.pie(pc, names="Persona", values="Count",
                       title="Survey sample by persona archetype",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_traces(textposition="inside", textinfo="percent+label")
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Adoption intent by persona")
        cross = pd.crosstab(df["persona_label"], df["adoption_intent"], normalize="index") * 100
        cross.columns = [INTENT_LABELS[c] for c in cross.columns]
        cross = cross.round(1).reset_index()
        fig4 = px.bar(
            cross.melt(id_vars="persona_label", var_name="Intent", value_name="Pct"),
            x="persona_label", y="Pct", color="Intent",
            color_discrete_map={v: INTENT_COLORS[k] for k, v in INTENT_LABELS.items()},
            title="Intent distribution within each persona (%)",
            barmode="stack"
        )
        fig4.update_layout(height=350, xaxis_title="Persona", yaxis_title="% respondents")
        st.plotly_chart(fig4, use_container_width=True)
