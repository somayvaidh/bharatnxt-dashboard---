"""Page 6 — Regression: GTV and Spend Prediction"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_data, preprocess, train_regression, get_feature_matrix

def show():
    st.title("📈 Regression — GTV and Spending Power Prediction")
    st.markdown(
        "**How much will this customer transact?** "
        "Predict monthly payment volume proxy to prioritise high-LTV leads."
    )
    st.markdown('<span class="section-tag tag-reg">Regression</span>', unsafe_allow_html=True)

    df_raw = load_data()
    df = preprocess(df_raw)

    with st.spinner("Training regression model..."):
        model, scaler, X_tr, X_te, y_tr, y_te, feat_cols = train_regression(df)

    y_pred = model.predict(X_te)

    # ── Metrics ───────────────────────────────────────────────────────────
    st.markdown("### Model performance")
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",  f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("R²",   f"{r2:.3f}")

    st.info(
        "Target variable is ordinal monthly payment volume (1=<1L, 2=1L-5L, 3=5L-25L, 4=>25L). "
        "R² indicates how much variance the model explains."
    )

    # ── Actual vs predicted ───────────────────────────────────────────────
    st.markdown("### Actual vs predicted — scatter")
    sample_idx = np.random.choice(len(y_te), min(400, len(y_te)), replace=False)
    act = np.array(y_te)[sample_idx]
    pred_s = y_pred[sample_idx]
    jitter = np.random.normal(0, 0.05, len(act))

    fig1 = px.scatter(
        x=act + jitter, y=pred_s,
        labels={"x": "Actual (with jitter)", "y": "Predicted"},
        color=act.astype(str),
        title="Actual vs predicted monthly payment volume (sample 400)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig1.add_trace(go.Scatter(
        x=[act.min(), act.max()], y=[act.min(), act.max()],
        mode="lines", line=dict(dash="dash", color="gray"), name="Perfect fit"
    ))
    fig1.update_layout(height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # ── Residuals ─────────────────────────────────────────────────────────
    st.markdown("### Residual plot")
    residuals = np.array(y_te) - y_pred
    fig2 = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted", "y": "Residual"},
        color=residuals,
        color_continuous_scale="RdBu",
        title="Residual plot (ideal: random scatter around y=0)"
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    fig2.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Residual histogram ────────────────────────────────────────────────
    fig3 = px.histogram(residuals, nbins=40,
                         title="Residual distribution (should be ~normal, centred at 0)",
                         labels={"value": "Residual", "count": "Count"},
                         color_discrete_sequence=["#6366f1"])
    fig3.update_layout(height=280, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Feature coefficients ──────────────────────────────────────────────
    st.markdown("### Feature coefficients — what drives payment volume?")
    coef_df = pd.DataFrame({
        "Feature": feat_cols,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False).head(20)

    fig4 = px.bar(
        coef_df, x="Coefficient", y="Feature", orientation="h",
        color="Coefficient",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Top 20 regression coefficients (positive = drives higher volume)"
    )
    fig4.update_layout(height=520, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Predicted GTV by segment ──────────────────────────────────────────
    st.markdown("### Predicted payment volume by business segment")
    X_all = get_feature_matrix(df).fillna(0)
    X_all_s = scaler.transform(X_all)
    df_raw2 = load_data()
    df_raw2["predicted_vol"] = model.predict(X_all_s)

    seg_pred = df_raw2.groupby("business_type")["predicted_vol"].mean().reset_index()
    seg_pred.columns = ["Business type", "Avg predicted volume score"]
    seg_pred = seg_pred.sort_values("Avg predicted volume score", ascending=False)

    fig5 = px.bar(seg_pred, x="Business type", y="Avg predicted volume score",
                   color="Avg predicted volume score",
                   color_continuous_scale="Greens",
                   title="Predicted avg monthly payment volume score by segment")
    fig5.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

    # ── Fee tolerance regression ──────────────────────────────────────────
    st.markdown("### Fee tolerance by predicted volume segment")
    df_raw2["vol_bucket"] = pd.cut(
        df_raw2["predicted_vol"], bins=4,
        labels=["Low (<1L)","Medium (1L-5L)","High (5L-25L)","Very high (>25L)"]
    )
    fee_vol = df_raw2.groupby("vol_bucket")["max_fee_tolerance"].value_counts(
        normalize=True).mul(100).round(1).reset_index()
    fee_vol.columns = ["Vol bucket","Fee tolerance","Pct"]
    fig6 = px.bar(fee_vol, x="Vol bucket", y="Pct", color="Fee tolerance",
                   barmode="stack",
                   title="Fee tolerance distribution by predicted volume bucket (%)",
                   color_discrete_sequence=px.colors.qualitative.Pastel)
    fig6.update_layout(height=340)
    st.plotly_chart(fig6, use_container_width=True)

    st.success(
        "💡 **Founder insight:** High predicted-volume customers (>5L/month) are significantly "
        "more tolerant of 2–2.5% fees. Target them with the Instant plan and premium positioning. "
        "Low-volume customers need the Classic 1.89% plan messaging."
    )
