"""Page 5 — Classification Models (Random Forest + Logistic Regression)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import label_binarize
from utils import (load_data, preprocess, train_classifier,
                   get_xy, INTENT_LABELS, INTENT_COLORS)

def show():
    st.title("🤖 Classification Models")
    st.markdown(
        "**Will this customer adopt BharatNXT?** "
        "Random Forest (primary) + Logistic Regression (interpretable)."
    )
    st.markdown('<span class="section-tag tag-cls">Classification</span>', unsafe_allow_html=True)

    df_raw = load_data()
    df = preprocess(df_raw)

    with st.spinner("Training models (cached after first run)..."):
        rf, lr, scaler, X_tr, X_te, y_tr, y_te, feat_cols = train_classifier(df)

    # ── Predictions ───────────────────────────────────────────────────────
    y_pred_rf = rf.predict(X_te)
    y_pred_lr = lr.predict(X_te)
    y_prob_rf = rf.predict_proba(X_te)
    y_prob_lr = lr.predict_proba(X_te)
    classes   = sorted(df["adoption_intent"].unique())

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("### Model performance summary")

    def get_metrics(y_true, y_pred, y_prob, name):
        return {
            "Model": name,
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "F1-Score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                              average="weighted"), 4),
        }

    metrics_rf = get_metrics(y_te, y_pred_rf, y_prob_rf, "Random Forest")
    metrics_lr = get_metrics(y_te, y_pred_lr, y_prob_lr, "Logistic Regression")
    metrics_df = pd.DataFrame([metrics_rf, metrics_lr])
    st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    # Metric comparison bars
    fig_m = px.bar(
        metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        title="Performance metrics comparison",
        color_discrete_sequence=["#6366f1","#22c55e"]
    )
    fig_m.update_layout(height=320, yaxis_range=[0,1])
    st.plotly_chart(fig_m, use_container_width=True)

    # ── Model selector for detailed view ─────────────────────────────────
    sel_model = st.radio("Select model for detailed analysis", ["Random Forest","Logistic Regression"],
                          horizontal=True)
    y_pred = y_pred_rf if sel_model == "Random Forest" else y_pred_lr
    y_prob = y_prob_rf if sel_model == "Random Forest" else y_prob_lr

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Confusion matrix", "ROC curves", "Feature importance", "Class report"]
    )

    # ── Tab 1: Confusion matrix ───────────────────────────────────────────
    with tab1:
        cm = confusion_matrix(y_te, y_pred, labels=classes)
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual: {INTENT_LABELS[c]}" for c in classes],
            columns=[f"Pred: {INTENT_LABELS[c]}" for c in classes],
        )
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                            title=f"{sel_model} — Confusion matrix", aspect="auto")
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class accuracy
        per_class = cm.diagonal() / cm.sum(axis=1)
        pc_df = pd.DataFrame({
            "Class": [INTENT_LABELS[c] for c in classes],
            "Per-class accuracy": np.round(per_class, 3)
        })
        st.dataframe(pc_df, use_container_width=True)

    # ── Tab 2: ROC curves ─────────────────────────────────────────────────
    with tab2:
        st.markdown(f"#### {sel_model} — One-vs-Rest ROC curves")
        y_bin = label_binarize(y_te, classes=classes)
        fig_roc = go.Figure()
        roc_colors = ["#6366f1","#f97316","#22c55e","#ef4444"]

        for i, cls in enumerate(classes):
            if y_bin.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{INTENT_LABELS[cls]} (AUC={auc:.3f})",
                    line=dict(color=roc_colors[i % len(roc_colors)], width=2),
                ))

        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(dash="dash", color="gray"), name="Random classifier"
        ))
        fig_roc.update_layout(
            title=f"{sel_model} — ROC curves (One-vs-Rest)",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            height=420,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Tab 3: Feature importance ─────────────────────────────────────────
    with tab3:
        st.markdown(f"#### {sel_model} — Feature importance")
        if sel_model == "Random Forest":
            importances = rf.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": feat_cols,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(20)

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                title="Top 20 feature importances (Random Forest)"
            )
            fig_fi.update_layout(height=520, yaxis=dict(autorange="reversed"),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        else:
            # LR: show mean abs coefficient across classes
            coef_mean = np.abs(lr.coef_).mean(axis=0)
            fi_df = pd.DataFrame({
                "Feature": feat_cols,
                "Abs coefficient": coef_mean
            }).sort_values("Abs coefficient", ascending=False).head(20)

            fig_fi = px.bar(
                fi_df, x="Abs coefficient", y="Feature", orientation="h",
                color="Abs coefficient", color_continuous_scale="Purples",
                title="Top 20 features by mean |coefficient| (Logistic Regression)"
            )
            fig_fi.update_layout(height=520, yaxis=dict(autorange="reversed"),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        st.success(
            "💡 **Founder insight:** The top features directly tell you what drives adoption. "
            "If `digital_comfort` ranks #1, your onboarding investment is the biggest lever. "
            "If `rbi_importance` ranks high, leading with 'RBI-compliant' in all ads will convert better."
        )

    # ── Tab 4: Classification report ──────────────────────────────────────
    with tab4:
        from sklearn.metrics import classification_report
        report = classification_report(
            y_te, y_pred,
            target_names=[INTENT_LABELS[c] for c in classes],
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).T
        report_df = report_df.drop(index=["accuracy"], errors="ignore")
        report_df = report_df.round(3)
        st.markdown(f"#### {sel_model} — Full classification report")
        st.dataframe(report_df, use_container_width=True)

        # Precision-Recall bar
        class_names = [INTENT_LABELS[c] for c in classes]
        pr_df = pd.DataFrame({
            "Class": class_names,
            "Precision": [report[INTENT_LABELS[c]]["precision"] for c in classes],
            "Recall":    [report[INTENT_LABELS[c]]["recall"]    for c in classes],
            "F1-Score":  [report[INTENT_LABELS[c]]["f1-score"]  for c in classes],
        })
        fig_pr = px.bar(
            pr_df.melt(id_vars="Class", var_name="Metric", value_name="Score"),
            x="Class", y="Score", color="Metric", barmode="group",
            title=f"{sel_model} — Per-class precision, recall, F1",
            color_discrete_sequence=["#6366f1","#22c55e","#f97316"]
        )
        fig_pr.update_layout(height=340)
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── SHAP (if available) ───────────────────────────────────────────────
    st.markdown("### SHAP explainability (if available)")
    try:
        import shap
        from utils import get_feature_matrix
        X_test_df = pd.DataFrame(X_te, columns=feat_cols)
        sample = X_test_df.sample(min(200, len(X_test_df)), random_state=42)
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(sample)

        # Use class 3 (Definitely interested) SHAP
        if isinstance(shap_vals, list):
            sv = shap_vals[3] if len(shap_vals) > 3 else shap_vals[-1]
        else:
            sv = shap_vals[:, :, 3] if shap_vals.ndim == 3 else shap_vals

        mean_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": feat_cols,
            "Mean |SHAP|": mean_shap
        }).sort_values("Mean |SHAP|", ascending=False).head(15)

        fig_shap = px.bar(
            shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
            color="Mean |SHAP|", color_continuous_scale="Oranges",
            title="SHAP — Top 15 features driving 'Definitely interested' prediction"
        )
        fig_shap.update_layout(height=450, yaxis=dict(autorange="reversed"),
                                coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)

    except Exception as e:
        st.info(f"SHAP not available in this environment ({type(e).__name__}). "
                "Feature importance chart above serves the same purpose.")
