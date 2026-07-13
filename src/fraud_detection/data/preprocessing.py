import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Optional
from ..utils.logger import logger

class DataPreprocessor:
    """Preprocess data for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scalers = {}
        self.scaler = None
    
    def create_preprocessing_pipeline(self, feature_config: Dict):
        """Create preprocessing pipeline"""
        numeric_features = feature_config.get('numeric_features', [])
        categorical_features = feature_config.get('categorical_features', [])
        scaling_method = feature_config.get('scaling_method', 'standard')
        
        # Select scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        # Create transformers
        transformers = []
        if numeric_features:
            transformers.append(('scaler', scaler, numeric_features))
        
        # Handle categorical features with one-hot encoding if needed
        if categorical_features:
            from sklearn.preprocessing import OneHotEncoder
            transformers.append(('encoder', OneHotEncoder(drop='first'), categorical_features))
        
        self.scaler = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        return self.scaler
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform the data"""
        if self.scaler is None:
            self.create_preprocessing_pipeline(self.config.get('feature_config', {}))
        
        return self.scaler.fit_transform(X)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data"""
        if self.scaler is None:
            self.create_preprocessing_pipeline(self.config.get('feature_config', {}))
        
        return self.scaler.transform(X)
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataset"""
        if strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif strategy == 'drop':
            df = df.dropna()
        else:
            logger.warning(f"Unknown strategy: {strategy}, using mean")
            df = df.fillna(df.mean())
        
        return df
    
    def scale_amount(self, df: pd.DataFrame, column: str = 'Amount', method: str = 'standard') -> pd.DataFrame:
        """Scale the Amount column specifically"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        df_copy = df.copy()
        df_copy[f'{column}_scaled'] = scaler.fit_transform(df_copy[[column]])
        logger.info(f"Scaled {column} column using {method} scaling")
        
        return df_copy