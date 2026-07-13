import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class FraudPredictor:
    def __init__(self, model_path="models/model.pkl", threshold=0.5):
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.parent / "model_metadata.json"
        self.threshold = threshold
        
        # Load model and metadata
        self.model = self._load_model()
        self.metadata = self._load_metadata()
        
        if self.model:
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("No model found. Please train a model first.")
    
    def _load_model(self):
        """Load trained model from disk"""
        if self.model_path.exists():
            return joblib.load(self.model_path)
        return None
    
    def _load_metadata(self):
        """Load model metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess incoming data"""
        # Ensure correct column order
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
        
        # Ensure all required features are present
        required_features = self.metadata.get('features', []) if self.metadata else []
        
        if required_features:
            for feature in required_features:
                if feature not in df.columns:
                    df[feature] = 0  # Fill missing with 0
        
        return df
    
    def predict(self, transaction_df: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction for single transaction"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess
        processed_df = self.preprocess(transaction_df)
        
        # Predict
        fraud_probability = self.model.predict_proba(processed_df)[0, 1]
        is_fraud = fraud_probability > self.threshold
        
        # Generate explanation
        explanation = self.explain_prediction(processed_df)
        
        return {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'threshold': self.threshold,
            'explanation': explanation
        }
    
    def predict_batch(self, transactions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Make predictions for batch of transactions"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess
        processed_df = self.preprocess(transactions_df)
        
        # Predict
        fraud_probabilities = self.model.predict_proba(processed_df)[:, 1]
        predictions = fraud_probabilities > self.threshold
        
        results = []
        for idx, (prob, is_fraud) in enumerate(zip(fraud_probabilities, predictions)):
            transaction_id = transactions_df.iloc[idx].get('transaction_id', f'txn_{idx}')
            
            results.append({
                'transaction_id': transaction_id,
                'fraud_probability': float(prob),
                'is_fraud': bool(is_fraud),
                'threshold': self.threshold
            })
        
        return results
    
    def explain_prediction(self, transaction_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate simple explanation for prediction"""
        if hasattr(self.model.named_steps['model'], 'feature_importances_'):
            # For tree-based models
            importances = self.model.named_steps['model'].feature_importances_
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            
            # Get top 3 features
            top_indices = np.argsort(importances)[-3:][::-1]
            top_features = {
                feature_names[i]: float(importances[i])
                for i in top_indices
            }
            
            return {
                'top_features': top_features,
                'model_type': 'tree_based'
            }
        elif hasattr(self.model.named_steps['model'], 'coef_'):
            # For linear models
            coefficients = self.model.named_steps['model'].coef_[0]
            feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            
            # Get top 3 positive and negative coefficients
            top_positive = np.argsort(coefficients)[-3:][::-1]
            top_negative = np.argsort(coefficients)[:3]
            
            explanation = {
                'increases_fraud_risk': {
                    feature_names[i]: float(coefficients[i])
                    for i in top_positive
                },
                'decreases_fraud_risk': {
                    feature_names[i]: float(coefficients[i])
                    for i in top_negative
                },
                'model_type': 'linear'
            }
            
            return explanation
        
        return {'explanation': 'Model does not support feature explanations'}
    
    def update_threshold(self, new_threshold: float):
        """Update prediction threshold"""
        if 0 <= new_threshold <= 1:
            self.threshold = new_threshold
            logger.info(f"Threshold updated to {new_threshold}")
        else:
            raise ValueError("Threshold must be between 0 and 1")