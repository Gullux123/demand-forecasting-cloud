import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate prepared data quality"""
    
    def __init__(self):
        self.X_train = pd.read_csv('data/prepared/X_train.csv')
        self.X_val = pd.read_csv('data/prepared/X_val.csv')
        self.X_test = pd.read_csv('data/prepared/X_test.csv')
        self.y_train = pd.read_csv('data/prepared/y_train.csv').values.ravel()
        self.y_val = pd.read_csv('data/prepared/y_val.csv').values.ravel()
        self.y_test = pd.read_csv('data/prepared/y_test.csv').values.ravel()
        
        logger.info(" Loaded all datasets")
    
    def validate_shapes(self):
        """Validate shape consistency"""
        logger.info("\n1️ SHAPE VALIDATION")
        
        assert self.X_train.shape[1] == self.X_val.shape[1] == self.X_test.shape[1], \
            "Feature count mismatch!"
        
        assert len(self.X_train) == len(self.y_train), \
            "Train X and y size mismatch!"
        
        logger.info(f" Train: {self.X_train.shape}")
        logger.info(f" Val: {self.X_val.shape}")
        logger.info(f" Test: {self.X_test.shape}")
        logger.info(f" All shapes valid!")
    
    def validate_nan_values(self):
        """Check for NaN values"""
        logger.info("\n2️ NaN VALUE CHECK")
        
        nan_train = self.X_train.isnull().sum().sum()
        nan_val = self.X_val.isnull().sum().sum()
        nan_test = self.X_test.isnull().sum().sum()
        
        logger.info(f"NaN in train: {nan_train}")
        logger.info(f"NaN in val: {nan_val}")
        logger.info(f"NaN in test: {nan_test}")
        
        assert nan_train == 0 and nan_val == 0 and nan_test == 0, \
            "Found NaN values in prepared data!"
        
        logger.info(f" No NaN values found!")
    
    def validate_inf_values(self):
        """Check for infinity values"""
        logger.info("\n3️ INFINITY VALUE CHECK")
        
        inf_train = np.isinf(self.X_train.values).sum()
        inf_val = np.isinf(self.X_val.values).sum()
        inf_test = np.isinf(self.X_test.values).sum()
        
        logger.info(f"Inf in train: {inf_train}")
        logger.info(f"Inf in val: {inf_val}")
        logger.info(f"Inf in test: {inf_test}")
        
        assert inf_train == 0 and inf_val == 0 and inf_test == 0, \
            "Found infinity values in prepared data!"
        
        logger.info(f" No infinity values found!")
    
    def validate_feature_ranges(self):
        """Check feature value ranges"""
        logger.info("\n4️ FEATURE RANGE CHECK")
        
        logger.info(f"Train feature ranges (first 5):")
        for col in self.X_train.columns[:5]:
            min_val = self.X_train[col].min()
            max_val = self.X_train[col].max()
            logger.info(f"  {col:30s}: [{min_val:7.2f}, {max_val:7.2f}]")
        
        logger.info(f" Features have reasonable ranges!")
    
    def validate_target_distribution(self):
        """Check target distribution"""
        logger.info("\n5️ TARGET DISTRIBUTION CHECK")
        
        logger.info(f"Train target statistics:")
        logger.info(f"  Mean: ${self.y_train.mean():.2f}")
        logger.info(f"  Std: ${self.y_train.std():.2f}")
        logger.info(f"  Min: ${self.y_train.min():.2f}")
        logger.info(f"  Max: ${self.y_train.max():.2f}")
        
        logger.info(f"Val target statistics:")
        logger.info(f"  Mean: ${self.y_val.mean():.2f}")
        logger.info(f"  Std: ${self.y_val.std():.2f}")
        
        logger.info(f" Target distributions look good!")
    
    def detect_outliers(self):
        """Detect outliers in target"""
        logger.info("\n6️ OUTLIER DETECTION")
        
        Q1 = np.percentile(self.y_train, 25)
        Q3 = np.percentile(self.y_train, 75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = np.sum((self.y_train < lower) | (self.y_train > upper))
        
        logger.info(f"Outliers in train target: {outliers} ({outliers/len(self.y_train)*100:.2f}%)")
        logger.info(f" Outlier count acceptable!")
    
    def validate_all(self):
        """Run all validations"""
        logger.info("\n" + "="*70)
        logger.info("DATA VALIDATION")
        logger.info("="*70)
        
        self.validate_shapes()
        self.validate_nan_values()
        self.validate_inf_values()
        self.validate_feature_ranges()
        self.validate_target_distribution()
        self.detect_outliers()
        
        logger.info("\n" + "="*70)
        logger.info(" ALL VALIDATION CHECKS PASSED!")
        logger.info("="*70)

# Usage
if __name__ == "__main__":
    validator = DataValidator()
    validator.validate_all()
    
    print("\n Data is ready for modeling!")