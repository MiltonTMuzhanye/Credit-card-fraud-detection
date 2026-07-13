from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
import pandas as pd
from .schemas import (
    TransactionRequest, TransactionResponse,
    BatchTransactionRequest, BatchTransactionResponse,
    PredictionMetrics
)
from ..inference.predictor import FraudPredictor
from ..utils.logger import logger

router = APIRouter()

# Initialize predictor
predictor = FraudPredictor()

@router.post("/predict", response_model=TransactionResponse)
async def predict_transaction(transaction: TransactionRequest):
    """Predict fraud for a single transaction"""
    try:
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Make prediction
        result = predictor.predict(transaction_dict)
        
        return TransactionResponse(
            transaction_id=result['transaction_id'],
            is_fraud=result['is_fraud'],
            fraud_probability=result['probability'],
            risk_level=result['risk_level'],
            features=transaction_dict
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchTransactionResponse)
async def predict_batch(transactions: BatchTransactionRequest):
    """Predict fraud for multiple transactions"""
    try:
        transaction_list = transactions.transactions
        results = predictor.predict_batch([t.dict() for t in transaction_list])
        
        return BatchTransactionResponse(
            results=results,
            total=len(results),
            fraud_count=sum(1 for r in results if r['is_fraud'])
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=PredictionMetrics)
async def get_metrics():
    """Get model performance metrics"""
    try:
        metrics = predictor.get_metrics()
        return PredictionMetrics(**metrics)
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))