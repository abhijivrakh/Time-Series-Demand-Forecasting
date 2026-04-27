import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Forecasting ML Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
}
.header-box {
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/xgb_model.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-box">
    <h1>📈 Time Series Sales Forecasting Dashboard</h1>
    <h4>Developed by: Abhishek Jivrakh</h4>
    <p>
    A production-ready Machine Learning system designed to forecast future sales trends 
    using historical data, feature engineering, and advanced regression modeling.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- BUSINESS PROBLEM ----------------
with st.expander("🏢 Business Problem & Impact", expanded=True):
    st.write("""
    This project addresses a critical business challenge — **accurate sales forecasting**.

    Businesses often face:
    - Overstocking / Understocking
    - Demand uncertainty
    - Revenue fluctuations
    - Inefficient supply chain decisions

    🎯 **Solution:**
    This ML system predicts future sales trends and helps businesses:
    - Optimize inventory
    - Improve demand planning
    - Make proactive decisions
    - Increase operational efficiency

    📌 **Impact:**
    - Reduced inventory costs
    - Improved revenue predictability
    - Data-driven decision making
    """)

# ---------------- DATASET ----------------
with st.expander("📊 Dataset Understanding (Important)", expanded=False):
    st.write("""
    The model is trained on a **time-series structured dataset**.

    📌 **Required Columns:**
    - `date` → Time dimension
    - `sales` → Actual sales (target variable)
    - `product_category` → Product segmentation

    📌 **Engineered Features:**
    - Lag features (previous sales)
    - Rolling averages
    - Calendar features (day, month, seasonality)

    ⚠️ The model expects the same feature structure during prediction.
    """)

# ---------------- MODEL INFO ----------------
with st.expander("🤖 Model Used & Justification"):
    st.write("""
    **Model Used:** XGBoost Regressor

    ✔ Handles structured/tabular data efficiently  
    ✔ Captures non-linear patterns  
    ✔ Works well with engineered time-series features  
    ✔ Production-friendly and fast  

    This model learns patterns from historical data and predicts future sales behavior.
    """)

# ---------------- USE CASES ----------------
with st.expander("🌍 Real-World Applications"):
    st.write("""
    🛒 Retail & E-commerce → Demand forecasting  
    🚚 Supply Chain → Inventory optimization  
    📦 FMCG → Seasonal planning  
    📊 Business Strategy → Revenue forecasting  
    📈 Marketing → Campaign planning  
    """)

# ---------------- MLOPS ----------------
with st.expander("⚙️ Production & MLOps"):
    st.write("""
    This system is designed with production in mind.

    🔧 Technologies:
    - XGBoost
    - Streamlit
    - Joblib

    🚀 MLOps:
    - Model versioning
    - Pipeline-based preprocessing
    - Docker-ready architecture
    - CI/CD ready deployment

    📌 Ensures scalability, reliability, and real-world usability.
    """)

# ---------------- FILE UPLOAD ----------------
st.subheader("📤 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload featured_dataset.csv",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])

    st.success("Dataset uploaded successfully.")

    # ---------------- PREVIEW ----------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------- BASIC INFO ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")

    # ---------------- FEATURES ----------------
    X = df.drop(columns=["date", "sales", "product_category"], errors="ignore")

    # ---------------- PREDICTION ----------------
    predictions = model.predict(X)
    df["Predicted Sales"] = predictions

    # ---------------- METRICS ----------------
    if "sales" in df.columns:
        mae = mean_absolute_error(df["sales"], df["Predicted Sales"])
        rmse = np.sqrt(mean_squared_error(df["sales"], df["Predicted Sales"]))
        r2 = r2_score(df["sales"], df["Predicted Sales"])

        st.subheader("📌 Model Performance")

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", round(mae, 2))
        m2.metric("RMSE", round(rmse, 2))
        m3.metric("R² Score", round(r2, 3))

    # ---------------- GRAPH ----------------
    st.subheader("📉 Actual vs Predicted Sales")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["sales"], label="Actual", linewidth=2)
    ax.plot(df["date"], df["Predicted Sales"], label="Predicted", linewidth=2)
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # ---------------- INSIGHTS ----------------
    st.subheader("💡 Business Insight")

    avg_actual = df["sales"].mean()
    avg_pred = df["Predicted Sales"].mean()

    if avg_pred > avg_actual:
        st.info("Predicted trend shows growth. Business can scale operations.")
    else:
        st.info("Predicted trend shows possible decline. Review strategy.")

    # ---------------- DOWNLOAD ----------------
    st.subheader("📥 Download Results")

    st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        "predictions.csv"
    )

else:
    st.warning("Upload dataset to start forecasting.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "🚀 Production-Ready ML System | XGBoost | Time Series Forecasting | Docker + CI/CD Ready | Abhishek Jivrakh"
)
