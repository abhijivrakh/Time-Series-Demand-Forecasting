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
    <h4>Built by: Abhishek Jivrakh</h4>
    <p>
    An end-to-end Machine Learning dashboard that predicts future sales using historical business data,
    engineered features, and an XGBoost regression model.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- ABOUT PROJECT ----------------
with st.expander("📌 About This Project", expanded=True):
    st.write("""
    This project is designed to forecast sales using time-series-based features such as lag values,
    rolling averages, date-based patterns, and business-level indicators.

    The goal is to help businesses understand expected sales trends, compare actual vs predicted values,
    and support data-driven planning.
    """)

with st.expander("🤖 Model Used and Why"):
    st.write("""
    **Model Used:** XGBoost Regressor

    **Why XGBoost?**
    - Handles tabular and engineered time-series features very well
    - Captures non-linear relationships better than simple linear models
    - Performs strongly on structured business datasets
    - Supports feature importance and production deployment
    - Faster and easier to deploy compared to deep learning models for this use case

    In this project, time-series behavior is captured through feature engineering, and XGBoost learns patterns from those features.
    """)

# ---------------- FILE UPLOAD ----------------
st.subheader("📤 Upload Featured Dataset")

uploaded_file = st.file_uploader(
    "Upload your featured_dataset.csv file",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])

    st.success("Dataset uploaded successfully.")

    # ---------------- RAW DATA ----------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ---------------- BASIC DATA INFO ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Features", df.shape[1])

    with col3:
        st.metric("Date Range", f"{df['date'].min().date()} → {df['date'].max().date()}")

    # ---------------- FEATURE PREPARATION ----------------
    X = df.drop(columns=["date", "sales", "product_category"], errors="ignore")

    # ---------------- PREDICTION ----------------
    predictions = model.predict(X)
    df["Predicted Sales"] = predictions

    # ---------------- METRICS ----------------
    if "sales" in df.columns:
        mae = mean_absolute_error(df["sales"], df["Predicted Sales"])
        rmse = np.sqrt(mean_squared_error(df["sales"], df["Predicted Sales"]))
        r2 = r2_score(df["sales"], df["Predicted Sales"])

        st.subheader("📌 Model Performance on Uploaded Data")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("MAE", round(mae, 2))

        with m2:
            st.metric("RMSE", round(rmse, 2))

        with m3:
            st.metric("R² Score", round(r2, 3))

    # ---------------- FORECAST VISUALIZATION ----------------
    st.subheader("📉 Actual vs Predicted Sales")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], df["sales"], label="Actual Sales", linewidth=2)
    ax.plot(df["date"], df["Predicted Sales"], label="Predicted Sales", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Actual Sales vs Predicted Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # ---------------- BUSINESS INSIGHTS ----------------
    st.subheader("💡 Business Insights")

    avg_actual = df["sales"].mean()
    avg_predicted = df["Predicted Sales"].mean()

    if avg_predicted > avg_actual:
        insight = "The model predicts an upward sales trend. This may indicate strong future demand or positive business momentum."
    elif avg_predicted < avg_actual:
        insight = "The model predicts a possible sales decline. The business should monitor demand, inventory, and marketing activity."
    else:
        insight = "The predicted sales are close to actual averages, showing stable business behavior."

    st.info(insight)

    # ---------------- OUTPUT TABLE ----------------
    st.subheader("📋 Prediction Output")
    st.dataframe(df, use_container_width=True)

    # ---------------- DOWNLOAD ----------------
    st.subheader("📥 Download Prediction Results")

    csv_output = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_output,
        file_name="sales_forecast_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.warning("Please upload a CSV file to generate sales forecasts.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "Project: Time Series Sales Forecasting | Model: XGBoost Regressor | Developed by Abhishek Jivrakh"
)
