from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Transaction(BaseModel):
    """Transaction data schema"""
    transaction_id: str
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_12345",
                "time": 0.0,
                "v1": -1.359807,
                "v2": -0.072781,
                "amount": 149.62,
                # ... other features
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    threshold: float
    timestamp: str
    explanation: Optional[Dict[str, Any]] = None

class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    transactions: List[Transaction]

class ModelMetrics(BaseModel):
    """Model metrics schema"""
    roc_auc: float
    precision: float
    recall: float
    f2_score: float
    business_cost: float
    optimal_threshold: float