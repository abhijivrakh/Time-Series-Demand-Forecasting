# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load cleaned data from CSV"""
    return pd.read_csv(filepath, parse_dates=['date'])

def create_features(df):
    """Create time-based, lag, and rolling features"""

    # Sort by date to avoid issues in rolling/lags
    df = df.sort_values('date').reset_index(drop=True)

    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Lag features
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['lag_30'] = df['sales'].shift(30)

    # Rolling averages
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['sales'].shift(1).rolling(window=7).std()

    # Categorical encoding
    le = LabelEncoder()
    df['product_category_encoded'] = le.fit_transform(df['product_category'])

    # Drop rows with NA (from lag/rolling)
    df = df.dropna().reset_index(drop=True)

    return df

def save_data(df, filepath):
    """Save processed data to CSV"""
    df.to_csv(filepath, index=False)

# Run the script
if __name__ == "__main__":
    # Since everything is in your desktop "Time Series" folder:
    input_path = "cleaned_dataset.csv"
    output_path = "featured_dataset.csv"

    df = load_data(input_path)
    df = create_features(df)
    save_data(df, output_path)

    print(" Feature engineering complete. Saved as 'featured_dataset.csv'")
