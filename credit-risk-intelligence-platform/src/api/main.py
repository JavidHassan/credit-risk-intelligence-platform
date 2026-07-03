"""
FastAPI Credit Risk API
Endpoints for prediction, batch prediction, model metrics, and explainability.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Intelligence API",
    description="API for credit card default prediction, risk scoring, and model explainability.",
    version="1.0.0",
)

# Global state
predictor = None
explainer_instance = None
model_metrics = {}


class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""
    customer_id: str = Field(..., example="CUST_00001")
    avg_utilization: float = Field(0.3, ge=0, le=1)
    max_utilization: float = Field(0.8, ge=0, le=1)
    current_utilization: float = Field(0.4, ge=0, le=1)
    avg_payment_to_balance: float = Field(0.6, ge=0)
    late_payment_count: float = Field(0, ge=0)
    late_payment_ratio: float = Field(0.0, ge=0, le=1)
    avg_monthly_payment: float = Field(500, ge=0)
    payment_volatility: float = Field(100, ge=0)
    spend_mean_3m: float = Field(1000, ge=0)
    spend_mean_6m: float = Field(1000, ge=0)
    spend_mean_12m: float = Field(1000, ge=0)
    delinquency_severity: float = Field(0, ge=0, le=4)
    income_to_debt_ratio: float = Field(3.0, ge=0)
    debt_burden: float = Field(0.3, ge=0)
    avg_merchant_risk: float = Field(0.3, ge=0)
    avg_transaction_amount: float = Field(80, ge=0)


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class PredictionResponse(BaseModel):
    customer_id: str
    default_probability: float
    prediction: int
    risk_level: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    global predictor, model_metrics
    try:
        from src.models.predict import CreditRiskPredictor
        model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
        feature_path = os.getenv("FEATURE_PATH", "models/feature_names.pkl")

        if os.path.exists(model_path):
            predictor = CreditRiskPredictor(model_path, feature_path)
            logger.info("Model loaded successfully.")
        else:
            logger.warning(f"Model not found at {model_path}. Run training first.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """Predict default probability for a single customer."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    feature_dict = features.dict()
    customer_id = feature_dict.pop("customer_id")
    df = pd.DataFrame([feature_dict])

    result = predictor.predict(df)
    return PredictionResponse(
        customer_id=customer_id,
        default_probability=result["default_probability"],
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Predict default probabilities for multiple customers."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    records = [c.dict() for c in request.customers]
    df = pd.DataFrame(records)
    customer_ids = df.pop("customer_id")

    results = predictor.predict_batch(df)
    results["customer_id"] = customer_ids.values

    return {
        "predictions": results.to_dict(orient="records"),
        "count": len(results),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model_metrics")
async def get_model_metrics():
    """Return model performance metrics."""
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            return json.load(f)
    return {"message": "No metrics available. Run model evaluation first."}


@app.get("/customer_explanation/{customer_id}")
async def customer_explanation(customer_id: str):
    """Get SHAP-based explanation for a customer's risk prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "customer_id": customer_id,
        "message": "Run SHAP analysis module for detailed explanations.",
        "note": "This endpoint requires SHAP explainer initialization with training data.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
