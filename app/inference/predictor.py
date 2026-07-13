import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from ...src.fraud_detection.utils.logger import logger
from ...src.fraud_detection.pipelines.inference_pipeline import InferencePipeline

class FraudPredictor:
    """Fraud prediction service"""
    
    def __init__(self):
        self.model_path = "artifacts/trained_models/best_model.pkl"
        self.scaler_path = "artifacts/scalers/amount_scaler.pkl"
        self.pipeline = InferencePipeline(self.model_path, self.scaler_path)
        self.metrics = {}
        
    def predict(self, transaction: Dict) -> Dict:
        """Predict fraud for a single transaction"""
        try:
            # Validate transaction
            self._validate_transaction(transaction)
            
            # Make prediction
            result = self.pipeline.predict_single(transaction)
            
            # Add additional info
            result['features'] = transaction
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for multiple transactions"""
        results = []
        for transaction in transactions:
            try:
                result = self.predict(transaction)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                results.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'error': str(e)
                })
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        # Load metrics from file
        metrics_path = Path("artifacts/thresholds/performance_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                import json
                self.metrics = json.load(f)
        
        return self.metrics
    
    def _validate_transaction(self, transaction: Dict):
        """Validate transaction data"""
        required_fields = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        
        for field in required_fields:
            if field not in transaction:
                raise ValueError(f"Missing required field: {field}")
        
        if transaction['Amount'] < 0:
            raise ValueError("Amount must be non-negative")