import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FraudModelTrainer:
    def __init__(self, config_path="config/model_config.yaml"):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def get_models(self, X_train, y_train):
        """Define model configurations"""
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        models = {
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                class_weight='balanced_subsample',
                n_estimators=100,
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                n_estimators=100,
                eval_metric='logloss',
                random_state=42
            ),
            'LightGBM': LGBMClassifier(
                class_weight='balanced',
                n_estimators=100,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        return models
    
    def train_model(self, model, preprocessor, X_train, y_train, use_smote=False):
        """Train a single model"""
        if use_smote:
            pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42, sampling_strategy=0.1)),
                ('model', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def train_all_models(self, X_train, y_train, preprocessor, use_smote=False):
        """Train all models and select the best one"""
        models = self.get_models(X_train, y_train)
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            pipeline = self.train_model(model, preprocessor, X_train, y_train, use_smote)
            trained_models[name] = pipeline
            
        return trained_models
    
    def save_model(self, model, model_name, metrics):
        """Save model and metadata"""
        # Save model
        model_path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'metrics': metrics,
            'features': model.named_steps['preprocessor'].get_feature_names_out().tolist()
        }
        
        metadata_path = self.models_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path