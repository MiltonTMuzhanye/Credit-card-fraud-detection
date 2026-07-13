from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List
from ..utils.logger import logger

class RandomForestModel:
    """Random Forest model for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
    
    def create_model(self) -> RandomForestClassifier:
        """Create Random Forest model with configuration"""
        params = self.config.get('random_forest', {})
        
        self.model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 5),
            class_weight=params.get('class_weight', 'balanced_subsample'),
            random_state=params.get('random_state', 42)
        )
        
        logger.info(f"Created Random Forest model with params: {params}")
        return self.model
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict:
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))