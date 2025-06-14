# app/streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/xgb_model.pkl")

st.title("📈 Time Series Sales Forecast Dashboard")

uploaded_file = st.file_uploader("featured_dataset.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])

    # Drop unwanted columns
    X = df.drop(columns=['date', 'sales', 'product_category'])

    st.subheader("📊 Raw Input Data")
    st.write(df.head())

    # Predict
    predictions = model.predict(X)
    df['Predicted Sales'] = predictions

    st.subheader("📉 Forecast Visualization")
    fig, ax = plt.subplots()
    df.plot(x='date', y=['sales', 'Predicted Sales'], ax=ax)
    st.pyplot(fig)

    st.subheader("📥 Download Prediction Results")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
