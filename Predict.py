# predict.py

import pandas as pd
import joblib
import os

def load_model(filepath="xgb_model.pkl"):
    """Load the trained XGBoost model"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)

def load_new_data(filepath):
    """Load new data for prediction (already feature-engineered)"""
    return pd.read_csv(filepath, parse_dates=['date'])

def predict(model, df):
    """Generate predictions from model"""
    X_new = df.drop(columns=['date', 'sales', 'product_category'])  # Drop string column
    predictions = model.predict(X_new)
    df['predicted_sales'] = predictions
    return df

def save_predictions(df, filepath="predicted_output.csv"):
    """Save predictions to CSV"""
    df.to_csv(filepath, index=False)
    print(f"✅ Predictions saved to {filepath}")

# Main
if __name__ == "__main__":
    model = load_model("xgb_model.pkl")
    new_data = load_new_data("featured_dataset.csv")  # You can replace this with new unseen data
    results = predict(model, new_data)
    save_predictions(results)
