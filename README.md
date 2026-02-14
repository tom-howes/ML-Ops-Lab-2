# Loan Amount Prediction API

A FastAPI app that predicts loan amounts using a Gradient Boosting model.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python train.py              # Train the model
uvicorn main:app --reload    # Start the API
```

## Test

Open **http://127.0.0.1:8000/docs** and try the `/predict` endpoint with:

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

## Endpoints

- `GET /` - Welcome message
- `GET /health` - API status
- `POST /predict` - Get loan prediction
