import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelMetrics:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def mape(y_true, y_pred):
     y_true = np.array(y_true)
     y_pred = np.array(y_pred)
     # avoid division by zero
     mask = y_true != 0
     return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def r2(y_true, y_pred):
        """R-squared Score"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def evaluate(y_true, y_pred, model_name="Model"):
        """Evaluate model performance"""
        metrics = {
            'MAPE': ModelMetrics.mape(y_true, y_pred),
            'RMSE': ModelMetrics.rmse(y_true, y_pred),
            'MAE': ModelMetrics.mae(y_true, y_pred),
            'R2': ModelMetrics.r2(y_true, y_pred)
        }
        
        logger.info(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics