import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path="config/data_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['data']['raw_path'])
        
    def load_raw_data(self):
        """Load raw credit card data"""
        file_path = self.data_dir / "creditcard.csv"
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path)
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Fraud rate: {df['Class'].mean():.4%}")
        
        return df
    
    def validate_data(self, df):
        """Basic data validation"""
        required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Check for null values
        if df.isnull().sum().sum() > 0:
            logger.warning("Data contains null values")
            
        return True
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Stratified train-test split"""
        from sklearn.model_selection import train_test_split
        
        X = df.drop(['Class', 'Time'], axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train fraud: {y_train.mean():.4%}, Test fraud: {y_test.mean():.4%}")
        
        return X_train, X_test, y_train, y_test