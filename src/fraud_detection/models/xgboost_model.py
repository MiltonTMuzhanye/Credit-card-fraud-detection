import xgboost as xgb
from typing import Dict, List, Optional
from ..utils.logger import logger

class XGBoostModel:
    """XGBoost model for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scale_pos_weight = None
    
    def create_model(self, scale_pos_weight: Optional[float] = None) -> xgb.XGBClassifier:
        """Create XGBoost model with configuration"""
        params = self.config.get('xgboost', {})
        
        if scale_pos_weight:
            params['scale_pos_weight'] = scale_pos_weight
        
        self.model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            scale_pos_weight=params.get('scale_pos_weight', 1),
            eval_metric=params.get('eval_metric', 'logloss'),
            random_state=params.get('random_state', 42),
            use_label_encoder=False
        )
        
        logger.info(f"Created XGBoost model with params: {params}")
        return self.model
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict:
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))