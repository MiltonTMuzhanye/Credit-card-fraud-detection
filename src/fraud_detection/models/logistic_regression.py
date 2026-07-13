from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Dict, Any
from ..utils.logger import logger

class LogisticRegressionModel:
    """Logistic Regression model for fraud detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        
    def create_model(self) -> LogisticRegression:
        """Create Logistic Regression model with configuration"""
        params = self.config.get('logistic_regression', {})
        self.model = LogisticRegression(
            class_weight=params.get('class_weight', 'balanced'),
            max_iter=params.get('max_iter', 1000),
            C=params.get('C', 1.0),
            solver=params.get('solver', 'lbfgs'),
            random_state=params.get('random_state', 42)
        )
        logger.info("Created Logistic Regression model")
        return self.model
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict:
        """Get feature importance from coefficients"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = dict(zip(feature_names, np.abs(self.model.coef_[0])))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))