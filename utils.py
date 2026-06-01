"""
utils.py — shared data loading, preprocessing, model training helpers
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import pickle, os, io

DATA_PATH = "bharatnxt_survey_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Numeric mappings ──────────────────────────────────────────────────────────
ORDINAL_MAPS = {
    "age_group":           {"18-25":1,"26-35":2,"36-45":3,"46-55":4,"56+":5},
    "city_tier":           {"Metro":4,"Tier1":3,"Tier2":2,"Tier3":1},
    "business_age":        {"<1yr":1,"1-3yr":2,"3-7yr":3,"7+yr":4},
    "annual_turnover":     {"<25L":1,"25L-1Cr":2,"1Cr-5Cr":3,">5Cr":4},
    "adoption_cycle":      {"<1week":1,"2-4weeks":2,"1-3months":3,">3months":4},
    "monthly_payment_vol": {"<1L":1,"1L-5L":2,"5L-25L":3,">25L":4},
    "cash_flow_crisis_freq":{"Never":1,"Rarely":2,"Sometimes":3,"Frequently":4},
    "rbi_importance":      {1:1,2:2,3:3,4:4,5:5},
    "digital_comfort":     {"Very_comfortable":4,"Comfortable":3,"Cautious":2,"Not_comfortable":1},
    "vendor_pay_usefulness":{1:1,2:2,3:3,4:4,5:5},
    "max_fee_tolerance":   {"No_fee":0,"<1pct":1,"1-1.5pct":2,"1.5-2pct":3,"2-2.5pct":4,">2.5pct":5},
    "credit_card_limit":   {"No_card":0,"<1L":1,"1L-5L":2,"5L-15L":3,">15L":4},
    "saas_budget":         {"Nothing":0,"<500":1,"500-1500":2,"1500-3000":3,">3000":4},
    "gst_importance":      {1:1,2:2,3:3,4:4,5:5},
    "peer_network_size":   {"None":0,"1-5":1,"6-20":2,"20+":3},
    "social_influence_score":{1:1,2:2,3:3,4:4,5:5},
}

CAT_COLS = [
    "business_type","decision_authority","payment_methods","working_capital_crisis",
    "switching_trigger","biggest_concern","fraud_experience","bad_fintech_exp",
    "onboarding_pref","settlement_pref","fx_interest","accounting_software",
    "pain_point","cash_pressure_months","festival_spike","bc1_fee_tradeoff",
    "bc2_disqualifier","has_credit_card",
]

BINARY_COLS = [c for c in [
    "pays_vendor_pay","pays_gst_tax","pays_rent","pays_utility","pays_salary",
    "pays_saas_sub","pays_insurance","pays_loan_emi",
    "tried_Razorpay_PayU","tried_BharatPe","tried_Instamojo",
    "tried_Bank_netbanking","tried_NBFC_apps","tried_None",
    "tool_CRM","tool_Cloud_storage","tool_HR_payroll",
    "tool_Project_mgmt","tool_Accounting_sw","tool_None",
]]

FEATURE_COLS = (
    list(ORDINAL_MAPS.keys()) +
    ["feat_reconciliation","feat_analytics","feat_virtual_cards",
     "feat_gst_filing","feat_invoice_finance","feat_payroll_bridge",
     "num_payment_types"] +
    BINARY_COLS
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Apply ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Label encode remaining categoricals
    le = LabelEncoder()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col].astype(str))

    # Fill numeric NaN with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df

def get_feature_matrix(df_processed: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df_processed.columns]
    X = df_processed[available].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X

def get_xy(df_processed: pd.DataFrame):
    X = get_feature_matrix(df_processed)
    y = df_processed["adoption_intent"].fillna(1).astype(int)
    return X, y

@st.cache_resource
def train_classifier(df_processed: pd.DataFrame):
    X, y = get_xy(df_processed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                             multi_class="ovr", random_state=42)
    lr.fit(X_tr_s, y_tr)

    # Save models
    with open(f"{MODEL_DIR}/rf_model.pkl","wb") as f: pickle.dump(rf, f)
    with open(f"{MODEL_DIR}/lr_model.pkl","wb") as f: pickle.dump(lr, f)
    with open(f"{MODEL_DIR}/scaler.pkl","wb") as f: pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/feature_cols.pkl","wb") as f:
        pickle.dump(list(X.columns), f)

    return rf, lr, scaler, X_tr_s, X_te_s, y_tr, y_te, list(X.columns)

@st.cache_resource
def train_kmeans(df_processed: pd.DataFrame, k=6):
    X = get_feature_matrix(df_processed)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    with open(f"{MODEL_DIR}/kmeans.pkl","wb") as f: pickle.dump(km, f)
    with open(f"{MODEL_DIR}/cluster_scaler.pkl","wb") as f: pickle.dump(scaler, f)
    return km, labels, Xs

@st.cache_resource
def train_regression(df_processed: pd.DataFrame):
    X = get_feature_matrix(df_processed)
    # Target: numeric monthly payment volume proxy
    y = df_processed["monthly_payment_vol"].fillna(2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LinearRegression()
    model.fit(X_tr_s, y_tr)
    return model, scaler, X_tr_s, X_te_s, y_tr, y_te, list(X.columns)

def score_new_data(new_df: pd.DataFrame):
    """Score a newly uploaded CSV using saved models."""
    try:
        with open(f"{MODEL_DIR}/rf_model.pkl","rb") as f: rf = pickle.load(f)
        with open(f"{MODEL_DIR}/scaler.pkl","rb") as f: scaler = pickle.load(f)
        with open(f"{MODEL_DIR}/feature_cols.pkl","rb") as f: feat_cols = pickle.load(f)
        with open(f"{MODEL_DIR}/kmeans.pkl","rb") as f: km = pickle.load(f)
        with open(f"{MODEL_DIR}/cluster_scaler.pkl","rb") as f: cs = pickle.load(f)
    except FileNotFoundError:
        return None, "Models not trained yet. Please visit the Classification and Clustering pages first."

    df_p = preprocess(new_df)
    available = [c for c in feat_cols if c in df_p.columns]
    missing = [c for c in feat_cols if c not in df_p.columns]

    X = df_p[available].fillna(0)
    # Add missing cols as 0
    for mc in missing:
        X[mc] = 0
    X = X[feat_cols]

    X_s = scaler.transform(X)
    proba = rf.predict_proba(X_s)
    pred_class = rf.predict(X_s)
    adopt_prob = proba[:, 2] + proba[:, 3]  # P(probably + definitely)

    # Cluster assignment
    X_cs = cs.transform(X)
    cluster = km.predict(X_cs)

    CLUSTER_NAMES = {
        0: "High-Intent Tech Users",
        1: "Pain-Aware RE Professionals",
        2: "Fee-Sensitive Undecideds",
        3: "Trust-Blocked Skeptics",
        4: "Traditional Low-Digital",
        5: "Power Transactors",
    }

    def recommend_action(prob, cls, bc2=""):
        if prob >= 0.75:
            return "🟢 Priority Call — offer Instant plan trial"
        elif prob >= 0.55 and "fraud" in str(bc2).lower():
            return "🔵 Trust Nurture — send RBI compliance email"
        elif prob >= 0.50:
            return "🔵 Feature Education — demo reconciliation + GST"
        elif prob >= 0.35:
            return "🟡 Cold Nurture — 6-week WhatsApp drip"
        else:
            return "🔴 Deprioritise — revisit in 6 months"

    bc2_col = new_df.get("bc2_disqualifier", pd.Series([""] * len(new_df)))
    results = new_df.copy()
    results["adoption_probability"] = np.round(adopt_prob, 3)
    results["predicted_class"] = pred_class
    results["cluster_id"] = cluster
    results["cluster_name"] = [CLUSTER_NAMES.get(c, f"Cluster {c}") for c in cluster]
    results["recommended_action"] = [
        recommend_action(p, c, bc2_col.iloc[i] if i < len(bc2_col) else "")
        for i, (p, c) in enumerate(zip(adopt_prob, pred_class))
    ]
    return results, None

INTENT_LABELS = {0: "Definitely Not", 1: "Probably Not", 2: "Probably Yes", 3: "Definitely Yes"}
INTENT_COLORS = {0: "#ef4444", 1: "#f97316", 2: "#22c55e", 3: "#16a34a"}
CLUSTER_NAMES = {
    0: "High-Intent Tech Users",
    1: "Pain-Aware RE Professionals",
    2: "Fee-Sensitive Undecideds",
    3: "Trust-Blocked Skeptics",
    4: "Traditional Low-Digital",
    5: "Power Transactors",
}
