"""Page 7 — New Customer Scoring Engine"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from utils import (load_data, preprocess, train_classifier, train_kmeans,
                   score_new_data, INTENT_LABELS, INTENT_COLORS, CLUSTER_NAMES)

def show():
    st.title("🚀 New Customer Scoring Engine")
    st.markdown(
        "**Upload new survey responses → instant adoption scoring → prioritised action list.** "
        "Use this every time you collect new leads from trade fairs, CA referrals, or campaigns."
    )

    # ── Ensure models are trained ─────────────────────────────────────────
    df_raw = load_data()
    df = preprocess(df_raw)

    with st.spinner("Preparing models..."):
        train_classifier(df)
        train_kmeans(df, k=6)

    st.success("✅ Models ready. Upload a CSV of new survey responses below.")

    # ── Template download ─────────────────────────────────────────────────
    st.markdown("### Step 1 — Download the template")
    template_cols = [
        "respondent_id","age_group","city_tier","business_type","business_age",
        "annual_turnover","decision_authority","adoption_cycle","payment_methods",
        "monthly_payment_vol","cash_flow_crisis_freq","has_credit_card",
        "working_capital_crisis","digital_comfort","onboarding_pref",
        "vendor_pay_usefulness","settlement_pref","max_fee_tolerance",
        "credit_card_limit","saas_budget","gst_importance","accounting_software",
        "peer_network_size","social_influence_score","pain_point",
        "cash_pressure_months","festival_spike","bc1_fee_tradeoff","bc2_disqualifier",
        "biggest_concern","fraud_experience","bad_fintech_exp","rbi_importance",
        "pays_vendor_pay","pays_gst_tax","pays_rent","pays_utility","pays_salary",
        "pays_saas_sub","pays_insurance","pays_loan_emi",
        "num_payment_types",
        "feat_reconciliation","feat_analytics","feat_virtual_cards",
        "feat_gst_filing","feat_invoice_finance","feat_payroll_bridge",
    ]
    template_df = pd.DataFrame(columns=template_cols)
    # Add one example row
    example = {
        "respondent_id":"NEW001","age_group":"26-35","city_tier":"Metro",
        "business_type":"IT/Software","business_age":"3-7yr",
        "annual_turnover":"1Cr-5Cr","decision_authority":"Owner alone",
        "adoption_cycle":"2-4weeks","payment_methods":"UPI",
        "monthly_payment_vol":"5L-25L","cash_flow_crisis_freq":"Sometimes",
        "has_credit_card":"Yes_corporate","working_capital_crisis":"Yes_manageable",
        "digital_comfort":"Very_comfortable","onboarding_pref":"Self_tutorial",
        "vendor_pay_usefulness":4,"settlement_pref":"Same_day",
        "max_fee_tolerance":"2-2.5pct","credit_card_limit":"5L-15L",
        "saas_budget":"500-1500","gst_importance":4,"accounting_software":"Zoho_QB",
        "peer_network_size":"6-20","social_influence_score":4,
        "pain_point":"Card_not_accepted","cash_pressure_months":"Oct-Dec",
        "festival_spike":"No_change","bc1_fee_tradeoff":"Instant_2.5pct",
        "bc2_disqualifier":"Yes_need_float","biggest_concern":"No_concerns",
        "fraud_experience":"No","bad_fintech_exp":"No_bad_exp","rbi_importance":3,
        "pays_vendor_pay":1,"pays_gst_tax":1,"pays_rent":1,"pays_utility":1,
        "pays_salary":1,"pays_saas_sub":1,"pays_insurance":0,"pays_loan_emi":0,
        "num_payment_types":6,
        "feat_reconciliation":4,"feat_analytics":4,"feat_virtual_cards":3,
        "feat_gst_filing":4,"feat_invoice_finance":3,"feat_payroll_bridge":3,
    }
    template_df = pd.concat([template_df, pd.DataFrame([example])], ignore_index=True)

    buf = io.BytesIO()
    template_df.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download CSV template",
        data=buf.getvalue(),
        file_name="bharatnxt_new_customers_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Upload & score ────────────────────────────────────────────────────
    st.markdown("### Step 2 — Upload new customer responses")
    uploaded = st.file_uploader(
        "Upload CSV (same columns as template)", type=["csv"]
    )

    if uploaded is None:
        st.info("👆 Upload a CSV file to score new customers. "
                "Use the sample dataset to see the engine in action.")

        if st.button("🧪 Run demo with 50 rows from training data"):
            sample = load_data().sample(50, random_state=99).copy()
            sample = sample.drop(columns=["adoption_intent","persona_label"], errors="ignore")
            _render_scores(sample)
        return

    try:
        new_df = pd.read_csv(uploaded)
        st.success(f"✅ Uploaded {len(new_df)} rows, {len(new_df.columns)} columns.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    _render_scores(new_df)


def _render_scores(new_df: pd.DataFrame):
    """Score the dataframe and render results."""
    with st.spinner("Scoring customers..."):
        results, error = score_new_data(new_df)

    if error:
        st.error(error)
        return

    st.markdown("### Step 3 — Scored results")

    # ── Summary KPIs ──────────────────────────────────────────────────────
    high   = (results["adoption_probability"] >= 0.75).sum()
    medium = ((results["adoption_probability"] >= 0.50) &
              (results["adoption_probability"] < 0.75)).sum()
    low    = (results["adoption_probability"] < 0.50).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total scored", len(results))
    c2.metric("🟢 Priority (≥75%)", int(high))
    c3.metric("🔵 Nurture (50–74%)", int(medium))
    c4.metric("🔴 Cold (<50%)", int(low))

    # ── Action distribution ───────────────────────────────────────────────
    st.markdown("#### Recommended actions distribution")
    act_vc = results["recommended_action"].value_counts().reset_index()
    act_vc.columns = ["Action","Count"]
    fig1 = px.bar(act_vc, x="Count", y="Action", orientation="h",
                   color="Count", color_continuous_scale="RdYlGn",
                   title="Recommended actions for uploaded leads")
    fig1.update_layout(height=300, coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig1, use_container_width=True)

    # ── Probability distribution ──────────────────────────────────────────
    st.markdown("#### Adoption probability distribution")
    fig2 = px.histogram(results, x="adoption_probability", nbins=20,
                         color_discrete_sequence=["#6366f1"],
                         title="Distribution of adoption probability scores")
    fig2.add_vline(x=0.75, line_dash="dash", line_color="green",
                    annotation_text="Priority threshold")
    fig2.add_vline(x=0.50, line_dash="dash", line_color="orange",
                    annotation_text="Nurture threshold")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Cluster breakdown ─────────────────────────────────────────────────
    st.markdown("#### Cluster assignment of new customers")
    cl_vc = results["cluster_name"].value_counts().reset_index()
    cl_vc.columns = ["Cluster","Count"]
    fig3 = px.pie(cl_vc, names="Cluster", values="Count",
                   title="Cluster distribution of uploaded leads",
                   color_discrete_sequence=px.colors.qualitative.Bold)
    fig3.update_layout(height=340)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Full results table ────────────────────────────────────────────────
    st.markdown("#### Full scored table (sortable)")
    display_cols = ["respondent_id"] if "respondent_id" in results.columns else []
    display_cols += ["adoption_probability","predicted_class","cluster_name","recommended_action"]
    extra = [c for c in ["business_type","city_tier","annual_turnover","monthly_payment_vol"]
             if c in results.columns]
    display_cols += extra
    display_cols = [c for c in display_cols if c in results.columns]

    st.dataframe(
        results[display_cols].sort_values("adoption_probability", ascending=False),
        use_container_width=True, height=400
    )

    # ── Download scored CSV ───────────────────────────────────────────────
    st.markdown("#### Download results")
    buf = io.BytesIO()
    results.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download scored CSV",
        data=buf.getvalue(),
        file_name="bharatnxt_scored_leads.csv",
        mime="text/csv",
    )

    # ── Marketing playbook ────────────────────────────────────────────────
    st.markdown("### Marketing playbook for this batch")
    playbook_data = {
        "Priority Call (≥75%)": {
            "count": int(high),
            "message": "Call within 24 hours. Lead with: 'Pay your vendors using your existing HDFC/ICICI credit card — no new credit needed.' Offer 0% fee on first ₹1L transaction.",
            "channel": "Founder direct call or senior sales rep",
            "urgency": "Same day"
        },
        "Trust Nurture (50–74%, concern=fraud/RBI)": {
            "count": int(medium),
            "message": "Send RBI Payment Aggregator Licence proof + 3 testimonials from similar businesses. Follow up after 5 days.",
            "channel": "WhatsApp Business + Email sequence",
            "urgency": "Within 48 hours"
        },
        "Cold Nurture (<50%)": {
            "count": int(low),
            "message": "Add to 6-week drip: Week 1 category education, Week 3 ROI calculator, Week 5 case study, Week 6 soft CTA.",
            "channel": "Email drip + LinkedIn retargeting",
            "urgency": "Low — batch process weekly"
        },
    }
    for action, details in playbook_data.items():
        with st.expander(f"{action} — {details['count']} leads"):
            st.markdown(f"**Message:** {details['message']}")
            st.markdown(f"**Channel:** {details['channel']}")
            st.markdown(f"**Urgency:** {details['urgency']}")
