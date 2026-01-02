import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config_path="config/feature_config.yaml"):
        self.features_dir = Path("data/features")
        self.features_dir.mkdir(exist_ok=True)
        
    def create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Time features (if Time is seconds from first transaction)
        df['Hour'] = df['Time'] // 3600
        df['Hour_of_Day'] = df['Hour'] % 24
        df['Time_of_Day'] = pd.cut(df['Hour_of_Day'],
                                  bins=[0, 6, 12, 18, 24],
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Transaction frequency (simulated)
        df['Transaction_Interval'] = df['Time'].diff().fillna(0)
        
        return df
    
    def create_amount_features(self, df):
        """Create amount-based features"""
        df = df.copy()
        
        # Normalized amount
        df['Amount_Log'] = np.log1p(df['Amount'])
        
        # Amount categories
        df['Amount_Category'] = pd.cut(df['Amount'],
                                      bins=[0, 10, 100, 1000, 10000, float('inf')],
                                      labels=['Micro', 'Small', 'Medium', 'Large', 'Very Large'])
        
        return df
    
    def build_preprocessor(self):
        """Build preprocessing pipeline"""
        preprocessor = ColumnTransformer([
            ('scale_amount', StandardScaler(), ['Amount']),
            ('passthrough', 'passthrough', [f'V{i}' for i in range(1, 29)])
        ])
        
        return preprocessor
    
    def save_preprocessor(self, preprocessor, path="models/preprocessor.pkl"):
        """Save preprocessor to disk"""
        joblib.dump(preprocessor, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path="models/preprocessor.pkl"):
        """Load preprocessor from disk"""
        return joblib.load(path)