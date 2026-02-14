"""
predict.py - Prediction logic for Loan Amount Prediction

This module handles:
- Loading the trained model
- Making predictions on new data
- Formatting prediction results
"""

import os
import joblib
import numpy as np
import pandas as pd

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'loan_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'label_encoders.pkl')


class LoanPredictor:
    """
    Handles loading model artifacts and making predictions.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.is_loaded = False
    
    def load_artifacts(self):
        """Load all model artifacts from disk."""
        if self.is_loaded:
            return
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please train the model first by running: python train.py"
            )
        
        # Load model and preprocessing artifacts
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.label_encoders = joblib.load(ENCODERS_PATH)
        self.is_loaded = True
        print("Model artifacts loaded successfully!")
    
    def preprocess_input(self, input_data: dict) -> np.ndarray:
        """
        Preprocess a single input for prediction.
        
        Args:
            input_data: Dictionary with loan application features
        
        Returns:
            Preprocessed feature array ready for prediction
        """
        # Define feature order (must match training)
        numeric_features = ['annual_income', 'employment_years', 'credit_score',
                           'debt_to_income_ratio', 'num_credit_lines', 'loan_term_months']
        categorical_features = ['home_ownership', 'loan_purpose']
        
        # Build feature vector
        features = []
        
        # Add numeric features
        for feat in numeric_features:
            features.append(float(input_data[feat]))
        
        # Add encoded categorical features
        for feat in categorical_features:
            encoded_value = self.label_encoders[feat].transform([input_data[feat]])[0]
            features.append(encoded_value)
        
        # Convert to numpy array and scale
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, input_data: dict) -> dict:
        """
        Make a loan amount prediction.
        
        Args:
            input_data: Dictionary with loan application features
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            self.load_artifacts()
        
        # Preprocess input
        X = self.preprocess_input(input_data)
        
        # Make prediction
        predicted_amount = self.model.predict(X)[0]
        
        # Ensure prediction is reasonable
        predicted_amount = max(1000, min(predicted_amount, 100000))
        
        # Build response
        result = {
            'predicted_loan_amount': round(predicted_amount, 2),
            'loan_range': {
                'minimum': round(predicted_amount * 0.9, 2),
                'maximum': round(predicted_amount * 1.1, 2)
            },
            'input_summary': {
                'annual_income': input_data['annual_income'],
                'credit_score': input_data['credit_score'],
                'loan_term_months': input_data['loan_term_months']
            }
        }
        
        return result
    
    def validate_input(self, input_data: dict) -> tuple[bool, str]:
        """
        Validate input data before prediction.
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = [
            'annual_income', 'employment_years', 'credit_score',
            'debt_to_income_ratio', 'num_credit_lines', 'loan_term_months',
            'home_ownership', 'loan_purpose'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in input_data:
                return False, f"Missing required field: {field}"
        
        # Validate ranges
        if not (20000 <= input_data['annual_income'] <= 500000):
            return False, "annual_income must be between 20,000 and 500,000"
        
        if not (0 <= input_data['employment_years'] <= 50):
            return False, "employment_years must be between 0 and 50"
        
        if not (300 <= input_data['credit_score'] <= 850):
            return False, "credit_score must be between 300 and 850"
        
        if not (0 <= input_data['debt_to_income_ratio'] <= 1):
            return False, "debt_to_income_ratio must be between 0 and 1"
        
        valid_home = ['RENT', 'OWN', 'MORTGAGE']
        if input_data['home_ownership'] not in valid_home:
            return False, f"home_ownership must be one of: {valid_home}"
        
        valid_purpose = ['debt_consolidation', 'home_improvement', 'business', 'education', 'other']
        if input_data['loan_purpose'] not in valid_purpose:
            return False, f"loan_purpose must be one of: {valid_purpose}"
        
        return True, ""


# Global predictor instance
predictor = LoanPredictor()


def get_prediction(input_data: dict) -> dict:
    """
    Main prediction function called by the API.
    """
    return predictor.predict(input_data)


if __name__ == "__main__":
    # Test prediction
    test_input = {
        'annual_income': 75000,
        'employment_years': 5,
        'credit_score': 720,
        'debt_to_income_ratio': 0.3,
        'num_credit_lines': 4,
        'loan_term_months': 36,
        'home_ownership': 'MORTGAGE',
        'loan_purpose': 'home_improvement'
    }
    
    print("Test Input:")
    for k, v in test_input.items():
        print(f"  {k}: {v}")
    
    result = get_prediction(test_input)
    
    print("\nPrediction Result:")
    print(f"  Predicted Loan Amount: ${result['predicted_loan_amount']:,.2f}")
    print(f"  Range: ${result['loan_range']['minimum']:,.2f} - ${result['loan_range']['maximum']:,.2f}")