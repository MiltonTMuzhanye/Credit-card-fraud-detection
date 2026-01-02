import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.anomaly_thresholds = {
            'Amount': (0, 100000),
            'V1': (-10, 10),
            'V2': (-10, 10),
            # Add more thresholds as needed
        }
    
    def check_anomalies(self, df):
        """Check for anomalous values"""
        anomalies = {}
        
        for col, (min_val, max_val) in self.anomaly_thresholds.items():
            if col in df.columns:
                n_anomalies = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if n_anomalies > 0:
                    anomalies[col] = n_anomalies
                    logger.warning(f"Column {col} has {n_anomalies} anomalous values")
        
        return anomalies
    
    def check_data_drift(self, reference_df, current_df, columns=None):
        """Check for data drift"""
        if columns is None:
            columns = reference_df.select_dtypes(include=[np.number]).columns
        
        drift_report = {}
        
        for col in columns:
            if col in reference_df.columns and col in current_df.columns:
                ref_mean = reference_df[col].mean()
                cur_mean = current_df[col].mean()
                
                # Simple drift detection
                drift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-10)
                
                if drift_pct > 0.1:  # 10% drift threshold
                    drift_report[col] = {
                        'reference_mean': ref_mean,
                        'current_mean': cur_mean,
                        'drift_pct': drift_pct
                    }
                    logger.warning(f"Data drift detected in {col}: {drift_pct:.2%}")
        
        return drift_report