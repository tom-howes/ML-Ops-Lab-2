"""
data.py - Data loading and preprocessing for Loan Amount Prediction

This module handles:
- Loading data (synthetic or from CSV)
- Feature engineering
- Data preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic loan application data.
    Use this if you don't have a real dataset.
    """
    np.random.seed(42)
    
    data = {
        'annual_income': np.random.normal(60000, 25000, n_samples).clip(20000, 200000),
        'employment_years': np.random.exponential(5, n_samples).clip(0, 35),
        'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, n_samples),
        'num_credit_lines': np.random.poisson(5, n_samples).clip(1, 20),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples),
        'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 
                                          'business', 'education', 'other'], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (loan_amount) with realistic relationships
    base_amount = (
        df['annual_income'] * 0.3 +
        df['credit_score'] * 50 +
        df['employment_years'] * 1000 -
        df['debt_to_income_ratio'] * 20000 +
        df['num_credit_lines'] * 500
    )
    
    # Add some noise
    noise = np.random.normal(0, 5000, n_samples)
    df['loan_amount'] = (base_amount + noise).clip(1000, 100000).round(2)
    
    return df


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load data from CSV or generate synthetic data.
    
    Args:
        filepath: Path to CSV file. If None, generates synthetic data.
    
    Returns:
        DataFrame with loan application data
    """
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
    else:
        df = generate_synthetic_data(1000)
        print(f"Generated {len(df)} synthetic records")
    
    return df


def preprocess_data(df: pd.DataFrame, fit_encoders: bool = True):
    """
    Preprocess the data for model training.
    
    Args:
        df: Raw DataFrame
        fit_encoders: If True, fit new encoders. If False, load existing ones.
    
    Returns:
        X: Feature matrix
        y: Target vector (if 'loan_amount' in df)
        feature_names: List of feature names
    """
    df = df.copy()
    
    # Separate features and target
    target_col = 'loan_amount'
    y = df[target_col].values if target_col in df.columns else None
    
    # Define feature columns
    numeric_features = ['annual_income', 'employment_years', 'credit_score',
                        'debt_to_income_ratio', 'num_credit_lines', 'loan_term_months']
    categorical_features = ['home_ownership', 'loan_purpose']
    
    # Encode categorical variables
    label_encoders = {}
    
    if fit_encoders:
        for col in categorical_features:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Save encoders
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    else:
        # Load existing encoders
        label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
        for col in categorical_features:
            df[col + '_encoded'] = label_encoders[col].transform(df[col])
    
    # Create feature matrix
    feature_cols = numeric_features + [f + '_encoded' for f in categorical_features]
    X = df[feature_cols].values
    
    # Scale features
    if fit_encoders:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    else:
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        X = scaler.transform(X)
    
    return X, y, feature_cols


def get_train_test_split(test_size: float = 0.2):
    """
    Load data and return train/test split.
    """
    df = load_data()
    X, y, feature_names = preprocess_data(df, fit_encoders=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    # Test data generation and preprocessing
    df = load_data()
    print("\nSample data:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"\nTarget statistics:")
    print(df['loan_amount'].describe())