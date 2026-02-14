"""
main.py - FastAPI Application for Loan Amount Prediction

This module defines:
- API endpoints
- Request/Response models
- Application configuration
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

from predict import predictor, get_prediction


# PYDANTIC MODELS (Request/Response Schemas)

class HomeOwnership(str, Enum):
    """Valid home ownership types."""
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"


class LoanPurpose(str, Enum):
    """Valid loan purposes."""
    DEBT_CONSOLIDATION = "debt_consolidation"
    HOME_IMPROVEMENT = "home_improvement"
    BUSINESS = "business"
    EDUCATION = "education"
    OTHER = "other"


class LoanApplication(BaseModel):
    """
    Request model for loan prediction.
    
    This defines the expected JSON structure for incoming requests.
    FastAPI will automatically validate the data against these constraints.
    """
    annual_income: float = Field(
        ...,
        ge=20000, le=500000,
        description="Annual income in dollars",
        example=75000
    )
    employment_years: float = Field(
        ...,
        ge=0, le=50,
        description="Years of employment",
        example=5
    )
    credit_score: int = Field(
        ...,
        ge=300, le=850,
        description="Credit score (300-850)",
        example=720
    )
    debt_to_income_ratio: float = Field(
        ...,
        ge=0, le=1,
        description="Debt-to-income ratio (0-1)",
        example=0.3
    )
    num_credit_lines: int = Field(
        ...,
        ge=0, le=30,
        description="Number of credit lines",
        example=4
    )
    loan_term_months: int = Field(
        ...,
        description="Loan term in months",
        example=36
    )
    home_ownership: HomeOwnership = Field(
        ...,
        description="Home ownership status",
        example="MORTGAGE"
    )
    loan_purpose: LoanPurpose = Field(
        ...,
        description="Purpose of the loan",
        example="home_improvement"
    )


class LoanRange(BaseModel):
    """Loan amount range in the response."""
    minimum: float
    maximum: float


class InputSummary(BaseModel):
    """Summary of key inputs."""
    annual_income: float
    credit_score: int
    loan_term_months: int


class LoanPredictionResponse(BaseModel):
    """
    Response model for loan prediction.
    
    Specifying response_model in the endpoint tells FastAPI to:
    - Serialize output to this format
    - Include this schema in the API documentation
    """
    predicted_loan_amount: float = Field(
        ...,
        description="Predicted loan amount in dollars"
    )
    loan_range: LoanRange = Field(
        ...,
        description="Estimated range for the loan amount"
    )
    input_summary: InputSummary = Field(
        ...,
        description="Summary of key input values"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    message: str


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Loan Amount Prediction API",
    description="""
    A machine learning API that predicts loan amounts based on applicant information.
    
    ## Features
    - Predict eligible loan amounts based on financial profile
    - Input validation with helpful error messages
    - Automatic API documentation
    
    ## How to Use
    1. Send a POST request to `/predict` with applicant details
    2. Receive predicted loan amount and range
    
    ## Model Information
    - Algorithm: Gradient Boosting Regressor
    - Features: Income, credit score, employment history, debt ratio, etc.
    """,
    version="1.0.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com"
    }
)


# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
async def load_model():
    """Load the ML model when the application starts."""
    try:
        predictor.load_artifacts()
        print("✓ Model loaded successfully on startup!")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("  The model will need to be trained before predictions can be made.")


# API ENDPOINTS

@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint - welcome message.
    """
    return {
        "message": "Welcome to the Loan Amount Prediction API!",
        "docs": "Visit /docs for interactive API documentation",
        "health": "Check /health for API status"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Use this to verify the API is running and the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        message="API is running" if predictor.is_loaded else "Model not loaded - run train.py first"
    )


@app.post("/predict", response_model=LoanPredictionResponse, tags=["Prediction"])
async def predict_loan_amount(application: LoanApplication):
    """
    Predict the loan amount for a given application.
    
    Submit applicant details and receive a predicted loan amount.
    
    **Example Request:**
    ```json
    {
        "annual_income": 75000,
        "employment_years": 5,
        "credit_score": 720,
        "debt_to_income_ratio": 0.3,
        "num_credit_lines": 4,
        "loan_term_months": 36,
        "home_ownership": "MORTGAGE",
        "loan_purpose": "home_improvement"
    }
    ```
    """
    # Check if model is loaded
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained first."
        )
    
    try:
        # Convert Pydantic model to dict for prediction
        input_data = {
            'annual_income': application.annual_income,
            'employment_years': application.employment_years,
            'credit_score': application.credit_score,
            'debt_to_income_ratio': application.debt_to_income_ratio,
            'num_credit_lines': application.num_credit_lines,
            'loan_term_months': application.loan_term_months,
            'home_ownership': application.home_ownership.value,
            'loan_purpose': application.loan_purpose.value
        }
        
        # Get prediction
        result = get_prediction(input_data)
        
        return LoanPredictionResponse(
            predicted_loan_amount=result['predicted_loan_amount'],
            loan_range=LoanRange(**result['loan_range']),
            input_summary=InputSummary(**result['input_summary'])
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the model and valid input values.
    """
    return {
        "model_type": "Gradient Boosting Regressor",
        "valid_inputs": {
            "annual_income": {"min": 20000, "max": 500000, "type": "float"},
            "employment_years": {"min": 0, "max": 50, "type": "float"},
            "credit_score": {"min": 300, "max": 850, "type": "integer"},
            "debt_to_income_ratio": {"min": 0, "max": 1, "type": "float"},
            "num_credit_lines": {"min": 0, "max": 30, "type": "integer"},
            "loan_term_months": {"options": [12, 24, 36, 48, 60], "type": "integer"},
            "home_ownership": {"options": ["RENT", "OWN", "MORTGAGE"], "type": "string"},
            "loan_purpose": {
                "options": ["debt_consolidation", "home_improvement", "business", "education", "other"],
                "type": "string"
            }
        }
    }


# RUN APPLICATION (for development)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)