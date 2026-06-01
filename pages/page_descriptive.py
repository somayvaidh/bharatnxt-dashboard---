"""Page 1 — Descriptive Analytics"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, INTENT_LABELS, INTENT_COLORS

def show():
    st.title("📊 Descriptive Analytics")
    st.markdown("**What does our survey sample look like?** Baseline distributions before any modelling.")

    df = load_data()
    st.markdown('<span class="section-tag tag-cls">Informs: All algorithms</span>', unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────
    with st.expander("🔽 Filter data", expanded=False):
        col1, col2, col3 = st.columns(3)
        seg = col1.multiselect("Business type", df["business_type"].dropna().unique(),
                                default=list(df["business_type"].dropna().unique()))
        tier = col2.multiselect("City tier", df["city_tier"].dropna().unique(),
                                 default=list(df["city_tier"].dropna().unique()))
        intent_f = col3.multiselect("Adoption intent", [0,1,2,3],
                                     default=[0,1,2,3],
                                     format_func=lambda x: INTENT_LABELS[x])
        mask = (df["business_type"].isin(seg) & df["city_tier"].isin(tier) &
                df["adoption_intent"].isin(intent_f))
        df = df[mask]
    st.caption(f"Showing {len(df):,} of 2,000 respondents")

    # ── Row 1: Demographics ───────────────────────────────────────────────
    st.markdown("### Demographics")
    c1, c2 = st.columns(2)

    with c1:
        age_vc = df["age_group"].value_counts().reset_index()
        age_vc.columns = ["Age group", "Count"]
        order = ["18-25","26-35","36-45","46-55","56+"]
        age_vc["Age group"] = pd.Categorical(age_vc["Age group"], categories=order, ordered=True)
        age_vc = age_vc.sort_values("Age group")
        fig = px.bar(age_vc, x="Age group", y="Count", title="Age group distribution",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tier_vc = df["city_tier"].value_counts().reset_index()
        tier_vc.columns = ["City Tier","Count"]
        fig2 = px.pie(tier_vc, names="City Tier", values="Count", title="City tier distribution",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        bt_vc = df["business_type"].value_counts().reset_index()
        bt_vc.columns = ["Business type","Count"]
        fig3 = px.bar(bt_vc, x="Business type", y="Count", title="Business type breakdown",
                      color="Count", color_continuous_scale="Teal")
        fig3.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        to_vc = df["annual_turnover"].value_counts().reset_index()
        to_vc.columns = ["Turnover","Count"]
        t_order = ["<25L","25L-1Cr","1Cr-5Cr",">5Cr"]
        to_vc["Turnover"] = pd.Categorical(to_vc["Turnover"], categories=t_order, ordered=True)
        to_vc = to_vc.sort_values("Turnover")
        fig4 = px.funnel(to_vc, x="Count", y="Turnover", title="Annual turnover distribution")
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 2: Payment Behaviour ──────────────────────────────────────────
    st.markdown("### Payment behaviour")
    c5, c6 = st.columns(2)

    with c5:
        vol_vc = df["monthly_payment_vol"].value_counts().reset_index()
        vol_vc.columns = ["Volume","Count"]
        v_order = ["<1L","1L-5L","5L-25L",">25L"]
        vol_vc["Volume"] = pd.Categorical(vol_vc["Volume"], categories=v_order, ordered=True)
        vol_vc = vol_vc.sort_values("Volume")
        fig5 = px.bar(vol_vc, x="Volume", y="Count",
                      title="Monthly B2B payment volume",
                      color="Count", color_continuous_scale="Oranges")
        fig5.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        cc_vc = df["has_credit_card"].value_counts().reset_index()
        cc_vc.columns = ["Credit Card Status","Count"]
        fig6 = px.pie(cc_vc, names="Credit Card Status", values="Count",
                      title="Credit card penetration",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig6.update_layout(height=300)
        st.plotly_chart(fig6, use_container_width=True)

    # ── Row 3: Fee Tolerance & Intent ────────────────────────────────────
    st.markdown("### Fee tolerance and adoption intent")
    c7, c8 = st.columns(2)

    with c7:
        fee_vc = df["max_fee_tolerance"].value_counts().reset_index()
        fee_vc.columns = ["Fee tolerance","Count"]
        f_order = ["No_fee","<1pct","1-1.5pct","1.5-2pct","2-2.5pct",">2.5pct"]
        fee_vc["Fee tolerance"] = pd.Categorical(fee_vc["Fee tolerance"],
                                                   categories=f_order, ordered=True)
        fee_vc = fee_vc.sort_values("Fee tolerance")
        fig7 = px.bar(fee_vc, x="Fee tolerance", y="Count",
                      title="Max convenience fee respondents will pay",
                      color="Count", color_continuous_scale="RdYlGn")
        fig7.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig7, use_container_width=True)

    with c8:
        # Adoption funnel
        surveyed  = len(df)
        aware     = int(surveyed * 0.85)
        interested= (df["adoption_intent"] >= 2).sum()
        high_int  = (df["adoption_intent"] == 3).sum()
        fig8 = go.Figure(go.Funnel(
            y=["Surveyed","Aware of category","Some interest","High intent"],
            x=[surveyed, aware, int(interested), int(high_int)],
            marker_color=["#6366f1","#22d3ee","#22c55e","#16a34a"],
            textposition="inside", textinfo="value+percent initial"
        ))
        fig8.update_layout(title="Adoption intent funnel", height=300)
        st.plotly_chart(fig8, use_container_width=True)

    # ── Payment types heatmap ─────────────────────────────────────────────
    st.markdown("### Payment type penetration by business segment")
    pay_cols = [c for c in df.columns if c.startswith("pays_")]
    pay_by_seg = df.groupby("business_type")[pay_cols].mean().round(3) * 100
    pay_by_seg.columns = [c.replace("pays_","") for c in pay_by_seg.columns]
    fig9 = px.imshow(pay_by_seg, text_auto=".0f",
                      color_continuous_scale="Blues",
                      title="% of segment making each payment type regularly",
                      aspect="auto")
    fig9.update_layout(height=280)
    st.plotly_chart(fig9, use_container_width=True)

    # ── City × Intent heatmap ─────────────────────────────────────────────
    st.markdown("### Adoption intent by city tier")
    ci_cross = pd.crosstab(df["city_tier"], df["adoption_intent"])
    ci_cross.columns = [INTENT_LABELS[c] for c in ci_cross.columns]
    fig10 = px.imshow(ci_cross, text_auto=True, color_continuous_scale="Greens",
                       title="Respondent count: city tier × adoption intent", aspect="auto")
    fig10.update_layout(height=260)
    st.plotly_chart(fig10, use_container_width=True)

    # ── Summary stats ─────────────────────────────────────────────────────
    st.markdown("### Summary statistics — numeric columns")
    num_df = df.select_dtypes(include=np.number).drop(
        columns=["adoption_intent"], errors="ignore")
    st.dataframe(num_df.describe().round(2), use_container_width=True)
