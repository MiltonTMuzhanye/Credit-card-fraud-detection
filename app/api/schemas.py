from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class TransactionRequest(BaseModel):
    """Request schema for single transaction prediction"""
    transaction_id: Optional[str] = None
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    
    @validator('Amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

class TransactionResponse(BaseModel):
    """Response schema for single transaction prediction"""
    transaction_id: Optional[str]
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    features: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchTransactionRequest(BaseModel):
    """Request schema for batch prediction"""
    transactions: List[TransactionRequest]

class BatchTransactionResponse(BaseModel):
    """Response schema for batch prediction"""
    results: List[Dict[str, Any]]
    total: int
    fraud_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PredictionMetrics(BaseModel):
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    last_updated: datetime