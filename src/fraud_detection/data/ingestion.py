import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from ..utils.logger import logger
from ..utils.helpers import DataValidator

class DataIngestion:
    """Handle data ingestion from various sources"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load CSV data from file"""
        path = file_path or self.data_path
        try:
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded data from {path}, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def load_kaggle_data(self, dataset_name: str) -> pd.DataFrame:
        """Load data from Kaggle (requires kagglehub)"""
        try:
            import kagglehub
            path = kagglehub.dataset_download(dataset_name)
            return self.load_csv(path)
        except ImportError:
            logger.error("kagglehub not installed. Please install with: pip install kagglehub")
            raise
    
    def split_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train/test sets with stratification"""
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train class distribution: {y_train.value_counts(normalize=True)}")
        
        return X_train, X_test, y_train, y_test