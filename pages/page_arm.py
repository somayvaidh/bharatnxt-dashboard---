"""Page 4 — Association Rule Mining (Apriori via mlxtend)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data

def show():
    st.title("🔗 Association Rule Mining")
    st.markdown("**Which payment types, tools and behaviours occur together?** Bundle and cross-sell intelligence.")
    st.markdown('<span class="section-tag tag-arm">Association Rule Mining</span>', unsafe_allow_html=True)

    df = load_data()

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder
        HAS_MLXTEND = True
    except ImportError:
        HAS_MLXTEND = False

    # ── Payment type co-occurrence (always available) ─────────────────────
    st.markdown("### Payment type co-occurrence heatmap")
    pay_cols = [c for c in df.columns if c.startswith("pays_")]
    pay_df = df[pay_cols].fillna(0).astype(int)
    pay_df.columns = [c.replace("pays_","") for c in pay_df.columns]
    cooc = pay_df.T.dot(pay_df)
    np.fill_diagonal(cooc.values, 0)
    fig1 = px.imshow(cooc, text_auto=True, color_continuous_scale="Blues",
                      title="Co-occurrence count — payment types made together", aspect="auto")
    fig1.update_layout(height=380)
    st.plotly_chart(fig1, use_container_width=True)

    # ── Tool co-occurrence ────────────────────────────────────────────────
    st.markdown("### Business tool co-usage heatmap")
    tool_cols = [c for c in df.columns if c.startswith("tool_")]
    tool_df = df[tool_cols].fillna(0).astype(int)
    tool_df.columns = [c.replace("tool_","") for c in tool_df.columns]
    tool_cooc = tool_df.T.dot(tool_df)
    np.fill_diagonal(tool_cooc.values, 0)
    fig2 = px.imshow(tool_cooc, text_auto=True, color_continuous_scale="Greens",
                      title="Co-usage count — SaaS tools used together", aspect="auto")
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Apriori rules ─────────────────────────────────────────────────────
    st.markdown("### Association rules — payment + tool basket analysis")

    if not HAS_MLXTEND:
        st.warning(
            "⚠️ `mlxtend` is not installed in this environment. "
            "Showing manual confidence/lift calculations instead."
        )
        _manual_rules(df, pay_cols, tool_cols)
        return

    # Build transaction basket: payment types + tool usage
    basket_cols = pay_cols + tool_cols
    basket = df[basket_cols].fillna(0).astype(bool)
    basket.columns = [c.replace("pays_","pay:").replace("tool_","tool:") for c in basket.columns]

    min_sup = st.slider("Minimum support", 0.05, 0.50, 0.15, 0.01)
    min_conf = st.slider("Minimum confidence", 0.30, 0.90, 0.50, 0.05)
    min_lift = st.slider("Minimum lift", 1.0, 3.0, 1.2, 0.1)

    freq = apriori(basket, min_support=min_sup, use_colnames=True, max_len=3)

    if freq.empty:
        st.warning("No frequent itemsets found at this support level. Try lowering minimum support.")
        return

    rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_conf].sort_values("lift", ascending=False)

    if rules.empty:
        st.warning("No rules found at these thresholds. Try relaxing confidence or lift.")
        return

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules_disp = rules[["antecedents","consequents","support","confidence","lift"]].copy()
    rules_disp["support"]    = rules_disp["support"].round(3)
    rules_disp["confidence"] = rules_disp["confidence"].round(3)
    rules_disp["lift"]       = rules_disp["lift"].round(3)

    st.markdown(f"**{len(rules_disp)} rules found** at support ≥ {min_sup}, "
                f"confidence ≥ {min_conf}, lift ≥ {min_lift}")
    st.dataframe(rules_disp.head(30), use_container_width=True)

    # ── Scatter: confidence vs lift ───────────────────────────────────────
    st.markdown("### Confidence vs lift scatter")
    fig3 = px.scatter(
        rules_disp.head(50), x="confidence", y="lift",
        size="support", color="lift",
        hover_data=["antecedents","consequents"],
        color_continuous_scale="RdYlGn",
        title="Top 50 rules — confidence vs lift (bubble size = support)"
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Top rules bar ─────────────────────────────────────────────────────
    st.markdown("### Top 10 rules by lift")
    top10 = rules_disp.head(10).copy()
    top10["rule"] = top10["antecedents"] + " → " + top10["consequents"]
    fig4 = px.bar(top10, x="lift", y="rule", orientation="h",
                   color="confidence", color_continuous_scale="Blues",
                   title="Top 10 association rules by lift")
    fig4.update_layout(height=380, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Business interpretation ───────────────────────────────────────────
    st.markdown("### Bundle recommendations from rules")
    st.success(
        "💡 **Founder interpretation:** Rules with lift > 1.5 suggest natural product bundles. "
        "E.g. 'vendor_pay + gst_tax → rent' means users who pay vendors and GST are very likely "
        "to also pay rent — price these three as a 'Core Business Bundle' at a 5% fee discount."
    )


def _manual_rules(df, pay_cols, tool_cols):
    """Fallback: compute simple 2-itemset confidence and lift manually."""
    st.markdown("#### Manual 2-itemset confidence and lift (top 20)")
    cols = [c.replace("pays_","pay:").replace("tool_","tool:") for c in pay_cols + tool_cols]
    data = df[pay_cols + tool_cols].fillna(0).astype(int)
    data.columns = cols
    n = len(data)

    results = []
    col_list = list(data.columns)
    for i, a in enumerate(col_list):
        for b in col_list[i+1:]:
            sup_a  = data[a].mean()
            sup_b  = data[b].mean()
            sup_ab = (data[a] & data[b]).mean()
            if sup_ab < 0.05 or sup_a == 0 or sup_b == 0:
                continue
            conf_ab = sup_ab / sup_a
            lift    = sup_ab / (sup_a * sup_b)
            results.append({
                "Antecedent": a, "Consequent": b,
                "Support": round(sup_ab, 3),
                "Confidence": round(conf_ab, 3),
                "Lift": round(lift, 3),
            })
    rules_df = pd.DataFrame(results).sort_values("Lift", ascending=False).head(20)
    st.dataframe(rules_df, use_container_width=True)

    fig = px.scatter(rules_df, x="Confidence", y="Lift", size="Support",
                      color="Lift", hover_data=["Antecedent","Consequent"],
                      color_continuous_scale="RdYlGn",
                      title="Confidence vs lift (manual 2-itemset)")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
