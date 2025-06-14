# model_training.py

import pandas as pd
import joblib
import sys
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Check and import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    print("❌ XGBoost is not installed. Please install it using one of the following commands:")
    print("\n▶ Conda (recommended): conda install -c conda-forge xgboost")
    print("▶ Pip: pip install xgboost")
    sys.exit(1)

def load_data(filepath):
    """Load feature-engineered data"""
    return pd.read_csv(filepath, parse_dates=['date'])

def train_model(df):
    """Train an XGBoost time series model with basic hyperparameter tuning"""

    # Define features and target
    X = df.drop(columns=['date', 'sales', 'product_category'])  # Drop string column
    y = df['sales']

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    # Model
    model = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
    }

    grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_

    # Evaluation
    preds = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"✅ Model Trained:\n→ RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    return best_model

def save_model(model, filepath="xgb_model.pkl"):
    """Save trained model using joblib"""
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")

# Run training
if __name__ == "__main__":
    df = load_data("featured_dataset.csv")
    model = train_model(df)
    save_model(model)

save_model(model, "xgb_model.pkl")
