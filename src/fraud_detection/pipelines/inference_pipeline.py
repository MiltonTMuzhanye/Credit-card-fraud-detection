import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Optional
from pathlib import Path
from ..utils.logger import logger
from ..utils.helpers import load_object
from ..data.preprocessing import DataPreprocessor

class InferencePipeline:
    """Inference pipeline for fraud detection"""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        self.model = self.load_model(model_path)
        self.scaler = self.load_scaler(scaler_path) if scaler_path else None
        self.feature_names = None
    
    def load_model(self, model_path: str):
        """Load the trained model"""
        try:
            model = load_object(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_scaler(self, scaler_path: str):
        """Load the scaler"""
        try:
            scaler = load_object(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
            return scaler
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return None
    
    def preprocess_transaction(self, transaction: Dict) -> np.ndarray:
        """Preprocess a single transaction"""
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Scale Amount if scaler exists
        if self.scaler is not None and 'Amount' in df.columns:
            df['Amount_scaled'] = self.scaler.transform(df[['Amount']])
            df = df.drop('Amount', axis=1)
        
        return df.values
    
    def predict_single(self, transaction: Dict) -> Dict:
        """Make prediction for a single transaction"""
        try:
            # Preprocess
            features = self.preprocess_transaction(transaction)
            
            # Predict
            probability = self.model.predict_proba(features)[:, 1][0]
            prediction = int(probability >= 0.5)
            
            # Calculate risk level
            if probability >= 0.7:
                risk_level = 'high'
            elif probability >= 0.3:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            result = {
                'transaction_id': transaction.get('transaction_id'),
                'prediction': prediction,
                'probability': float(probability),
                'risk_level': risk_level,
                'is_fraud': bool(prediction)
            }
            
            logger.info(f"Prediction for transaction {result['transaction_id']}: {result['is_fraud']}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Make predictions for multiple transactions"""
        results = []
        for transaction in transactions:
            try:
                result = self.predict_single(transaction)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for transaction: {e}")
                results.append({
                    'transaction_id': transaction.get('transaction_id'),
                    'error': str(e)
                })
        
        return results