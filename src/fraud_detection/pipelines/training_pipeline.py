import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional
import mlflow
from ..utils.logger import logger
from ..utils.config import config
from ..data.preprocessing import DataPreprocessor
from ..models.xgboost_model import XGBoostModel
from ..models.random_forest import RandomForestModel
from ..models.logistic_regression import LogisticRegressionModel
from ..training.hyperparameter_tuning import HyperparameterTuner
from ..training.imbalance_handler import ImbalanceHandler

class TrainingPipeline:
    """Complete training pipeline for fraud detection"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = config
        self.preprocessor = DataPreprocessor(self.config.get_config('data'))
        self.models = {}
        self.best_model = None
        self.best_score = None
        
    def prepare_data(self, X_train, y_train, X_test, y_test, scale_amount: bool = True):
        """Prepare data with preprocessing"""
        if scale_amount:
            # Scale only the Amount column
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train['Amount_scaled'] = scaler.fit_transform(X_train[['Amount']])
            X_test['Amount_scaled'] = scaler.transform(X_test[['Amount']])
            
            # Drop original Amount column
            X_train = X_train.drop('Amount', axis=1)
            X_test = X_test.drop('Amount', axis=1)
            
            # Store scaler for inference
            self.scaler = scaler
        
        logger.info(f"Prepared training data: {X_train.shape}, {y_train.shape}")
        return X_train, y_train, X_test, y_test
    
    def create_models(self):
        """Create all model instances"""
        self.models = {
            'logistic_regression': LogisticRegressionModel(self.config.get_config('models')).create_model(),
            'random_forest': RandomForestModel(self.config.get_config('models')).create_model(),
            'xgboost': XGBoostModel(self.config.get_config('models')).create_model()
        }
        logger.info(f"Created {len(self.models)} models")
        return self.models
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            with mlflow.start_run(run_name=f"{name}_training"):
                # Log parameters
                mlflow.log_params(model.get_params())
                
                # Train model
                model.fit(X_train, y_train)
                
                # Store model
                results[name] = {
                    'model': model,
                    'params': model.get_params()
                }
                
                mlflow.sklearn.log_model(model, f"model_{name}")
        
        return results
    
    def evaluate_models(self, X_test, y_test, results):
        """Evaluate all trained models"""
        from sklearn.metrics import roc_auc_score, classification_report
        
        for name, result in results.items():
            model = result['model']
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            result['metrics'] = {
                'roc_auc': auc,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score']
            }
            
            logger.info(f"{name} - ROC AUC: {auc:.4f}")
        
        return results
    
    def run_full_pipeline(self, X_train, y_train, X_test, y_test):
        """Run the complete training pipeline"""
        # Prepare data
        X_train, y_train, X_test, y_test = self.prepare_data(X_train, y_train, X_test, y_test)
        
        # Handle imbalance
        imbalance_handler = ImbalanceHandler(self.config.get_config('training'))
        X_train_resampled, y_train_resampled = imbalance_handler.fit_resample(
            X_train, y_train, method='smote'
        )
        
        # Create models
        self.create_models()
        
        # Train models
        training_results = self.train_models(X_train_resampled, y_train_resampled)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test, training_results)
        
        # Select best model
        best_model_name = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['metrics']['roc_auc'])
        self.best_model = evaluation_results[best_model_name]['model']
        self.best_score = evaluation_results[best_model_name]['metrics']['roc_auc']
        
        logger.info(f"Best model: {best_model_name} with ROC AUC: {self.best_score:.4f}")
        
        return evaluation_results, self.best_model