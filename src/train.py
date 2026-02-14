"""
train.py - Model training for Loan Amount Prediction

This module handles:
- Training a regression model
- Evaluating model performance
- Saving the trained model
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import get_train_test_split

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'loan_model.pkl')


def train_model():
    """
    Train a Gradient Boosting Regressor for loan amount prediction.
    
    Returns:
        model: Trained model
        metrics: Dictionary of evaluation metrics
    """
    print("=" * 50)
    print("LOAN AMOUNT PREDICTION - MODEL TRAINING")
    print("=" * 50)
    
    # Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = get_train_test_split()
    
    # Initialize model
    print("\n[2/4] Initializing Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        verbose=0
    )
    
    # Train model
    print("\n[3/4] Training model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate model
    print("\n[4/4] Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train': {
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test)
        }
    }
    
    # Print metrics
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    print(f"\nTraining Set:")
    print(f"  RMSE: ${metrics['train']['rmse']:,.2f}")
    print(f"  MAE:  ${metrics['train']['mae']:,.2f}")
    print(f"  R²:   {metrics['train']['r2']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  RMSE: ${metrics['test']['rmse']:,.2f}")
    print(f"  MAE:  ${metrics['test']['mae']:,.2f}")
    print(f"  R²:   {metrics['test']['r2']:.4f}")
    
    # Feature importance
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE")
    print("=" * 50)
    importance = list(zip(feature_names, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in importance:
        print(f"  {feat}: {imp:.4f}")
    
    # Save model
    print("\n" + "=" * 50)
    print("SAVING MODEL")
    print("=" * 50)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # Save feature names for reference
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    return model, metrics


def load_trained_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Please run train.py first."
        )
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    model, metrics = train_model()
    print("\n✓ Training complete!")