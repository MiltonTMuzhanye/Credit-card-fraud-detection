import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..utils.logger import logger

class FeatureEngineer:
    """Feature engineering for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def create_time_features(self, df: pd.DataFrame, time_col: str = 'Time') -> pd.DataFrame:
        """Create time-based features"""
        df_copy = df.copy()
        
        # Convert seconds to hours
        df_copy['hour'] = (df_copy[time_col] / 3600) % 24
        
        # Create hour bins
        df_copy['hour_bin'] = pd.cut(
            df_copy['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        # Day period
        df_copy['is_weekend'] = ((df_copy[time_col] / (3600 * 24)) % 7).isin([5, 6]).astype(int)
        
        logger.info(f"Created time features: hour, hour_bin, is_weekend")
        return df_copy
    
    def create_amount_features(self, df: pd.DataFrame, amount_col: str = 'Amount') -> pd.DataFrame:
        """Create amount-based features"""
        df_copy = df.copy()
        
        # Log transform
        df_copy['amount_log'] = np.log1p(df_copy[amount_col])
        
        # Amount bins
        bins = [0, 10, 50, 100, 500, 1000, 5000, 100000]
        labels = ['micro', 'small', 'medium', 'large', 'xlarge', 'xxlarge', 'massive']
        df_copy['amount_bin'] = pd.cut(df_copy[amount_col], bins=bins, labels=labels)
        
        # Interaction features
        df_copy['amount_squared'] = df_copy[amount_col] ** 2
        df_copy['amount_sqrt'] = np.sqrt(df_copy[amount_col])
        
        logger.info(f"Created amount features: amount_log, amount_bin, amount_squared, amount_sqrt")
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Create interaction features between selected columns"""
        df_copy = df.copy()
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                col1, col2 = features[i], features[j]
                if col1 in df.columns and col2 in df.columns:
                    df_copy[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        logger.info(f"Created {len(df_copy.columns) - len(df.columns)} interaction features")
        return df_copy
    
    def apply_aggregations(self, df: pd.DataFrame, group_col: str, agg_cols: List[str]) -> pd.DataFrame:
        """Apply aggregations to create aggregate features"""
        agg_dict = {}
        for col in agg_cols:
            agg_dict[col] = ['mean', 'std', 'min', 'max', 'median']
        
        agg_features = df.groupby(group_col).agg(agg_dict)
        agg_features.columns = [f'{col}_{agg}' for col, aggs in agg_dict.items() for agg in aggs]
        
        logger.info(f"Created {len(agg_features.columns)} aggregate features")
        return agg_features