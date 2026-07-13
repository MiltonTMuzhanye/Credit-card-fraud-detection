from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any, Optional
from ..utils.logger import logger

class HyperparameterTuner:
    """Hyperparameter tuning for fraud detection models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.best_params = {}
        self.best_score = None
    
    def tune_grid_search(self, pipeline: Pipeline, param_grid: Dict, X_train, y_train, cv: int = 5, scoring: str = 'roc_auc') -> Dict:
        """Perform grid search hyperparameter tuning"""
        logger.info(f"Starting Grid Search with {len(param_grid)} parameters")
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_random_search(self, pipeline: Pipeline, param_distributions: Dict, X_train, y_train, 
                          n_iter: int = 20, cv: int = 5, scoring: str = 'roc_auc') -> Dict:
        """Perform random search hyperparameter tuning"""
        logger.info(f"Starting Random Search with {n_iter} iterations")
        
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return random_search.best_estimator_