import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional
from ..utils.logger import logger

class FeatureSelector:
    """Feature selection methods for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.selected_features = []
    
    def select_by_correlation(self, df: pd.DataFrame, target_col: str, n_features: int = 10) -> List[str]:
        """Select features with highest absolute correlation to target"""
        correlations = df.corr()[target_col].abs().sort_values(ascending=False)
        
        # Remove target and any identifier columns
        correlations = correlations[~correlations.index.isin([target_col, 'Time'])]
        
        selected = correlations.head(n_features).index.tolist()
        logger.info(f"Selected {len(selected)} features by correlation")
        
        return selected
    
    def select_by_statistical(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', n_features: int = 20) -> List[str]:
        """Select features using statistical methods"""
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector.fit(X, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()
        
        # Get scores for selected features
        scores = selector.scores_[mask]
        feature_scores = dict(zip(selected, scores))
        
        logger.info(f"Selected {len(selected)} features using {method}")
        self.selected_features = selected
        return selected
    
    def select_by_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> List[str]:
        """Select features using Recursive Feature Elimination"""
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        mask = rfe.get_support()
        selected = X.columns[mask].tolist()
        
        logger.info(f"Selected {len(selected)} features using RFE")
        return selected
    
    def feature_importance_ranking(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get feature importance ranking using Random Forest"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Feature importance ranking completed")
        return importance_df