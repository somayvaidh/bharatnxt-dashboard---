"""Page 3 — Clustering (K-Means)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utils import load_data, preprocess, get_feature_matrix, CLUSTER_NAMES, INTENT_LABELS

def show():
    st.title("🎯 Clustering — K-Means Customer Segmentation")
    st.markdown("**Who are our natural customer groups?** Unsupervised discovery beyond the 6 survey personas.")
    st.markdown('<span class="section-tag tag-clus">Clustering</span>', unsafe_allow_html=True)

    df_raw = load_data()
    df = preprocess(df_raw)
    X = get_feature_matrix(df)
    X_filled = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_filled)

    # ── Elbow + Silhouette ────────────────────────────────────────────────
    st.markdown("### Optimal K — elbow curve and silhouette scores")
    k_range = range(2, 11)

    @st.cache_data
    def compute_elbow(_Xs):
        inertias, sils = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(_Xs)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(_Xs, lbl, sample_size=500))
        return inertias, sils

    inertias, sils = compute_elbow(Xs)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                                   marker_color="#6366f1", name="Inertia"))
        fig1.update_layout(title="Elbow curve (inertia)", xaxis_title="K",
                            yaxis_title="Inertia", height=300)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(k_range), y=sils, mode="lines+markers",
                                   marker_color="#22c55e", name="Silhouette"))
        fig2.update_layout(title="Silhouette score", xaxis_title="K",
                            yaxis_title="Silhouette", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # ── K selector ───────────────────────────────────────────────────────
    k = st.slider("Select number of clusters (K)", 2, 10, 6)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    df_raw = df_raw.copy()
    df_raw["cluster"] = labels
    df["cluster"] = labels

    sil = silhouette_score(Xs, labels, sample_size=500)
    st.metric("Silhouette score for selected K", f"{sil:.3f}",
              delta="Good" if sil > 0.2 else "Fair")

    # ── PCA 2D scatter ────────────────────────────────────────────────────
    st.markdown("### Cluster visualisation — PCA 2D projection")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    pca_df = pd.DataFrame({
        "PC1": coords[:,0], "PC2": coords[:,1],
        "Cluster": [CLUSTER_NAMES.get(l, f"Cluster {l}") for l in labels],
        "Intent": df["adoption_intent"].map(INTENT_LABELS),
        "Persona": df_raw["persona_label"],
    })
    fig3 = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                       symbol="Intent", hover_data=["Persona"],
                       title=f"K={k} clusters projected onto 2 principal components",
                       color_discrete_sequence=px.colors.qualitative.Bold)
    fig3.update_layout(height=460)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Cluster profiles ──────────────────────────────────────────────────
    st.markdown("### Cluster profiles")
    profile_cols = ["annual_turnover","monthly_payment_vol","digital_comfort",
                    "rbi_importance","vendor_pay_usefulness","max_fee_tolerance",
                    "num_payment_types","adoption_intent"]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profile = df.groupby("cluster")[profile_cols].mean().round(2)
    profile.index = [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in profile.index]
    st.dataframe(profile, use_container_width=True)

    # ── Radar charts ──────────────────────────────────────────────────────
    st.markdown("### Radar chart — cluster characteristics")
    radar_cols = ["annual_turnover","monthly_payment_vol","digital_comfort",
                  "rbi_importance","vendor_pay_usefulness","max_fee_tolerance","adoption_intent"]
    radar_cols = [c for c in radar_cols if c in df.columns]
    prof_norm = profile[radar_cols].copy()
    for col in prof_norm.columns:
        col_range = prof_norm[col].max() - prof_norm[col].min()
        if col_range > 0:
            prof_norm[col] = (prof_norm[col] - prof_norm[col].min()) / col_range

    fig4 = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, (idx, row) in enumerate(prof_norm.iterrows()):
        fig4.add_trace(go.Scatterpolar(
            r=row.values.tolist() + [row.values[0]],
            theta=radar_cols + [radar_cols[0]],
            fill="toself", name=str(idx),
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))
    fig4.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                        title="Normalised cluster profiles", height=460)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Adoption intent by cluster ────────────────────────────────────────
    st.markdown("### Adoption intent distribution per cluster")
    ci = df_raw.groupby(["cluster","adoption_intent"]).size().reset_index(name="count")
    ci["intent_label"] = ci["adoption_intent"].map(INTENT_LABELS)
    ci["cluster_name"] = ci["cluster"].map(lambda x: CLUSTER_NAMES.get(x, f"Cluster {x}"))
    fig5 = px.bar(ci, x="cluster_name", y="count", color="intent_label",
                   barmode="stack", title="Adoption intent breakdown per cluster",
                   color_discrete_sequence=["#ef4444","#f97316","#22c55e","#16a34a"])
    fig5.update_layout(height=340, xaxis_title="")
    st.plotly_chart(fig5, use_container_width=True)

    # ── Prescriptive actions ──────────────────────────────────────────────
    st.markdown("### Prescriptive actions per cluster")
    ACTIONS = {
        0: ("High-Intent Tech Users",       "🟢 Fast-track onboarding + 0% fee first transaction"),
        1: ("Pain-Aware RE Professionals",  "🔵 Highlight GST + vendor pay bundle, CA-partner referral"),
        2: ("Fee-Sensitive Undecideds",     "🟡 Emphasise Classic (T+3) 1.89% plan, ROI calculator"),
        3: ("Trust-Blocked Skeptics",       "🔵 Send RBI regulation proof + testimonials first"),
        4: ("Traditional Low-Digital",      "🔴 Deprioritise — CA channel only, no digital ads"),
        5: ("Power Transactors",            "🟢 Premium plan pitch + dedicated account manager"),
    }
    act_rows = []
    for cid in range(k):
        n = int((labels == cid).sum())
        name, action = ACTIONS.get(cid, (f"Cluster {cid}", "Review manually"))
        avg_intent = df[df["cluster"]==cid]["adoption_intent"].mean()
        act_rows.append({
            "Cluster": f"C{cid}: {name}",
            "Size": n,
            "Avg intent": round(avg_intent, 2),
            "Recommended action": action,
        })
    st.dataframe(pd.DataFrame(act_rows), use_container_width=True)
