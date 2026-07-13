from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from typing import Dict, Optional
from ..utils.logger import logger

class ImbalanceHandler:
    """Handle class imbalance for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sampler = None
    
    def get_sampler(self, method: str = 'smote', **kwargs):
        """Get the appropriate sampling method"""
        if method == 'smote':
            self.sampler = SMOTE(random_state=42, **kwargs)
        elif method == 'adasyn':
            self.sampler = ADASYN(random_state=42, **kwargs)
        elif method == 'random_under':
            self.sampler = RandomUnderSampler(random_state=42, **kwargs)
        elif method == 'tomek':
            self.sampler = TomekLinks(**kwargs)
        elif method == 'smote_tomek':
            self.sampler = SMOTETomek(random_state=42, **kwargs)
        elif method == 'smote_enn':
            self.sampler = SMOTEENN(random_state=42, **kwargs)
        else:
            logger.warning(f"Unknown sampling method: {method}, using SMOTE")
            self.sampler = SMOTE(random_state=42)
        
        logger.info(f"Using {method} for imbalance handling")
        return self.sampler
    
    def fit_resample(self, X, y, method: str = 'smote', **kwargs):
        """Fit and resample the data"""
        sampler = self.get_sampler(method, **kwargs)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logger.info(f"Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        logger.info(f"Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        return X_resampled, y_resampled