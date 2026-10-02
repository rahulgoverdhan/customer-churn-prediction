"""
STAGE 8: Streamlit Business Dashboard
Customer Churn Prediction — interactive prediction + analytics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# ── Load model once (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/final_model.joblib")

model = load_model()

# ── Helpers ──────────────────────────────────────────────────────────────────
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same CustomerChurnFeatureEngineer used during training."""
    from src.features import CustomerChurnFeatureEngineer
    fe = CustomerChurnFeatureEngineer()
    return fe.fit_transform(df)

def predict_churn(raw_row: pd.DataFrame):
    """Return (prediction, probability, risk_level) for a single raw customer row."""
    engineered = apply_feature_engineering(raw_row)
    # Drop customerID if present (pipeline drop it anyway via remainder='drop')
    if "customerID" in engineered.columns:
        engineered = engineered.drop(columns=["customerID"])
    proba = model.predict_proba(engineered)[0, 1]
    pred = int(proba >= 0.5)
    if proba < 0.35:
        risk = "Low"
    elif proba < 0.65:
        risk = "Medium"
    else:
        risk = "High"
    return pred, proba, risk

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Customer Churn Predictor
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔍 Customer Churn Predictor")

with st.form("customer_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("SeniorCitizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)

    with c2:
        phone = st.selectbox("PhoneService", ["No", "Yes"])
        multiple = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])

    with c3:
        device = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
        tech = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    c4, c5 = st.columns(2)
    with c4:
        paperless = st.selectbox("PaperlessBilling", ["No", "Yes"])
        payment = st.selectbox("PaymentMethod", [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check"
        ])
    with c5:
        monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        total_charges = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    row = pd.DataFrame([{
        "customerID": "DASHBOARD",
        "gender": gender,
        "SeniorCitizen": int(senior == "Yes"),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    pred, proba, risk = predict_churn(row)

    # Display results
    result_col1, result_col2, result_col3 = st.columns(3)
    with result_col1:
        st.metric("Prediction", "Churn" if pred == 1 else "No Churn")
    with result_col2:
        st.metric("Churn Probability", f"{proba:.1%}")
    with result_col3:
        st.metric("Risk Level", risk, delta_color="normal")

    # Risk explanation
    if risk == "High":
        st.warning("⚠️ This customer has a HIGH risk of churning based on the model's estimate.")
    elif risk == "Medium":
        st.info("ℹ️ This customer has a MEDIUM risk of churning.")
    else:
        st.success("✅ This customer has a LOW risk of churning.")

    st.caption("Probability is the model's estimate, not a guarantee of churn.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Business Analytics
# ══════════════════════════════════════════════════════════════════════════════
st.header("📈 Business Analytics")

@st.cache_data
def load_analytics_data():
    df = pd.read_csv("data/processed/cleaned.csv")
    return df

df = load_analytics_data()
total = len(df)
churned = int((df["Churn"] == "Yes").sum())
churn_rate = churned / total * 100
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Customers", f"{total:,}")
with k2:
    st.metric("Churned Customers", f"{churned:,}")
with k3:
    st.metric("Churn Rate", f"{churn_rate:.1f}%")
with k4:
    st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
with k5:
    st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

st.divider()

# Churn breakdown charts
st.subheader("Churn by Category")
def churn_by(column, title, data=None):
    """Show churn rate by a categorical column."""
    if data is None:
        data = df

    grp = data.groupby(column)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).sort_values(ascending=False)
    st.markdown(f"**{title}**")
    for label, rate in grp.items():
        st.progress(min(rate / 100, 1.0), text=f"{label} — {rate:.1f}%")
    st.caption("")

c1, c2 = st.columns(2)
with c1:
    churn_by("Contract", "Churn by Contract")
    churn_by("InternetService", "Churn by Internet Service")
    churn_by("PaymentMethod", "Churn by Payment Method")
with c2:
    churn_by("PaperlessBilling", "Churn by Paperless Billing")
    # TenureGroup derived from EDA findings
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
    df_tmp = df.copy()
    df_tmp["TenureGroup"] = pd.cut(df_tmp["tenure"], bins=bins, labels=labels, ordered=True)
    churn_by("TenureGroup", "Churn by Tenure Group", df_tmp)

# ══════════════════════════════════════════════════════════════════════════════
# Business Insights (based ONLY on existing EDA)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("💡 Business Insights (from EDA / explainability)")
st.markdown("""
- **Month-to-month contracts** have the highest churn rate (~42.7%). Migrating to longer contracts is the #1 retention lever.
- **Fiber optic customers** churn at ~41.9% — investigate service quality and competitive pricing.
- **Electronic check users** churn at ~45.3% — incentivize migration to automated payments.
- **First-year customers** are the most vulnerable (0-12 months: 57.3%). Early retention programs have the highest ROI.
- **Customers with more services** churn less — cross-selling and bundling is an effective retention strategy.
""")
