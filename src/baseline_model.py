import pandas as pd
import numpy as np
import logging
from src.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModel:
    """Create baseline models for comparison"""
    
    def __init__(self):
        self.X_train = pd.read_csv('data/prepared/X_train.csv')
        self.X_val = pd.read_csv('data/prepared/X_val.csv')
        self.X_test = pd.read_csv('data/prepared/X_test.csv')
        
        self.y_train = pd.read_csv('data/prepared/y_train.csv').values.ravel()
        self.y_val = pd.read_csv('data/prepared/y_val.csv').values.ravel()
        self.y_test = pd.read_csv('data/prepared/y_test.csv').values.ravel()
        
        logger.info(" Loaded datasets")
    
    def simple_mean_baseline(self):
        """Baseline: always predict training mean"""
        logger.info("\n1️ BASELINE 1: SIMPLE MEAN")
        
        train_mean = self.y_train.mean()
        
        y_pred_val = np.full_like(self.y_val, train_mean, dtype=float)
        y_pred_test = np.full_like(self.y_test, train_mean, dtype=float)
        
        logger.info(f"Mean of training set: ${train_mean:.2f}")
        
        val_metrics = ModelMetrics.evaluate(self.y_val, y_pred_val, "Simple Mean - Validation")
        test_metrics = ModelMetrics.evaluate(self.y_test, y_pred_test, "Simple Mean - Test")
        
        return val_metrics, test_metrics
    
    def seasonal_baseline(self):
        """Baseline: predict using day-of-week average"""
        logger.info("\n2️ BASELINE 2: SEASONAL AVERAGE")
        
        # This would require access to day-of-week info
        # For now, using a rolling average as proxy
        
        rolling_mean = pd.Series(self.y_train).rolling(window=7).mean().iloc[-1]
        
        y_pred_val = np.full_like(self.y_val, rolling_mean, dtype=float)
        y_pred_test = np.full_like(self.y_test, rolling_mean, dtype=float)
        
        logger.info(f"7-day rolling average: ${rolling_mean:.2f}")
        
        val_metrics = ModelMetrics.evaluate(self.y_val, y_pred_val, "Seasonal - Validation")
        test_metrics = ModelMetrics.evaluate(self.y_test, y_pred_test, "Seasonal - Test")
        
        return val_metrics, test_metrics
    
    def persistence_baseline(self):
        """Baseline: predict previous value"""
        logger.info("\n3️ BASELINE 3: PERSISTENCE (Last Value)")
        
        # Last value from training
        last_value = self.y_train[-1]
        
        y_pred_val = np.full_like(self.y_val, last_value, dtype=float)
        y_pred_test = np.full_like(self.y_test, last_value, dtype=float)
        
        logger.info(f"Last training value: ${last_value:.2f}")
        
        val_metrics = ModelMetrics.evaluate(self.y_val, y_pred_val, "Persistence - Validation")
        test_metrics = ModelMetrics.evaluate(self.y_test, y_pred_test, "Persistence - Test")
        
        return val_metrics, test_metrics
    
    def run_all_baselines(self):
        """Run all baseline models"""
        logger.info("\n" + "="*70)
        logger.info("BASELINE MODELS - ESTABLISH LOWER BOUND PERFORMANCE")
        logger.info("="*70)
        
        baselines = {}
        
        baselines['simple_mean'] = self.simple_mean_baseline()
        baselines['seasonal'] = self.seasonal_baseline()
        baselines['persistence'] = self.persistence_baseline()
        
        logger.info("\n" + "="*70)
        logger.info("BASELINE SUMMARY")
        logger.info("="*70)
        
        logger.info("\nValidation Set Performance:")
        logger.info("  Simple Mean MAPE: {:.2f}%".format(baselines['simple_mean'][0]['MAPE']))
        logger.info("  Seasonal MAPE: {:.2f}%".format(baselines['seasonal'][0]['MAPE']))
        logger.info("  Persistence MAPE: {:.2f}%".format(baselines['persistence'][0]['MAPE']))
        
        logger.info("\nTest Set Performance:")
        logger.info("  Simple Mean MAPE: {:.2f}%".format(baselines['simple_mean'][1]['MAPE']))
        logger.info("  Seasonal MAPE: {:.2f}%".format(baselines['seasonal'][1]['MAPE']))
        logger.info("  Persistence MAPE: {:.2f}%".format(baselines['persistence'][1]['MAPE']))
        
        logger.info("\n Baselines established!")
        logger.info("Our models should beat these!")
        
        return baselines

# Usage
if __name__ == "__main__":
    baseline = BaselineModel()
    baseline.run_all_baselines()