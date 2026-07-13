import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ..utils.logger import logger

class DataValidator:
    """Validate data quality and schema"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate data schema"""
        expected_columns = self.config.get('columns', [])
        if not expected_columns:
            logger.warning("No schema defined for validation")
            return True
        
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        
        return True
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality metrics"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'unique_counts': df.nunique().to_dict()
        }
        
        # Check for feature drift or anomalies
        logger.info(f"Data quality report: {quality_report}")
        return quality_report
    
    def validate_target_distribution(self, y: pd.Series, min_ratio: float = 0.001) -> bool:
        """Validate target distribution for imbalance"""
        class_counts = y.value_counts(normalize=True)
        
        if 1 not in class_counts:
            logger.error("No positive class found in target")
            return False
        
        minority_ratio = class_counts.get(1, 0)
        if minority_ratio < min_ratio:
            logger.warning(f"Highly imbalanced dataset: {minority_ratio:.4%} fraud cases")
        
        logger.info(f"Target distribution: {class_counts.to_dict()}")
        return True