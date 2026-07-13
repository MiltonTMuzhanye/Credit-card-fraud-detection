from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging
from typing import List, Optional

from .schemas import Transaction, PredictionResponse, BatchPredictionRequest
from .inference import FraudPredictor

app = FastAPI(title="Fraud Detection API", 
              description="Real-time credit card fraud detection",
              version="1.0.0")

# Initialize predictor
predictor = FraudPredictor()

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor.model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    """Predict fraud for a single transaction"""
    try:
        # Convert to DataFrame for prediction
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Make prediction
        prediction = predictor.predict(df)
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=prediction['fraud_probability'],
            is_fraud=prediction['is_fraud'],
            threshold=prediction['threshold'],
            timestamp=datetime.now().isoformat(),
            explanation=prediction.get('explanation', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(batch_request: BatchPredictionRequest):
    """Predict fraud for multiple transactions"""
    try:
        # Convert to DataFrame
        transactions = [t.dict() for t in batch_request.transactions]
        df = pd.DataFrame(transactions)
        
        # Make predictions
        predictions = predictor.predict_batch(df)
        
        return {
            "predictions": predictions,
            "total_transactions": len(predictions),
            "fraud_count": sum(p['is_fraud'] for p in predictions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
    """Get information about the current model"""
    if predictor.metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    
    return predictor.metadata