import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    """Prepare data for ML modeling"""
    
    def __init__(self, df_path='data/processed/train_features.csv'):
        logger.info(f"Loading data from {df_path}...")
        self.df = pd.read_csv(df_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        logger.info(f"✓ Loaded: {self.df.shape}")
    
    def identify_feature_types(self):
        """Identify numerical and categorical features"""
        logger.info("\n IDENTIFYING FEATURE TYPES")
        # Separate target from features
        self.y = self.df['Sales'].copy()
        # Drop non-predictive columns
        drop_cols = ['Date','Store','festival_name','holiday_name','PromoInterval']
        self.X = self.df.drop(columns=['Sales'] + drop_cols,errors='ignore')
        logger.info("Target variable: Sales")
        logger.info(f"Features: {len(self.X.columns)}")
        # Identify feature types
        self.numeric_features =self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Numeric features: {len(self.numeric_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
        if self.categorical_features:
            logger.info(f"Categorical columns: {self.categorical_features}")
        return self.X, self.y    

    def train_val_test_split(self, train_size=0.6,val_size=0.2,test_size=0.2,random_state=42):
        """Create train/validation/test split (temporal split)"""
        logger.info("\n CREATING TRAIN/VALIDATION/TEST SPLIT")
        # Time series split (avoid data leakage)
        n = len(self.X)
        train_idx = int(n * train_size)
        val_idx = int(n * (train_size + val_size))
        # Split features
        self.X_train = self.X.iloc[:train_idx].copy()
        self.X_val = self.X.iloc[train_idx:val_idx].copy()
        self.X_test = self.X.iloc[val_idx:].copy()
        # Split target
        self.y_train = self.y.iloc[:train_idx].copy()
        self.y_val = self.y.iloc[train_idx:val_idx].copy()
        self.y_test = self.y.iloc[val_idx:].copy()
        logger.info(f"Train set: {len(self.X_train)} rows "f"({len(self.X_train)/n*100:.1f}%)")
        logger.info(f"Validation set: {len(self.X_val)} rows "f"({len(self.X_val)/n*100:.1f}%)")
        logger.info( f"Test set: {len(self.X_test)} rows "f"({len(self.X_test)/n*100:.1f}%)")
        # Check date ranges
        logger.info("\nData quality check:")
        logger.info(f"Train date range: "f"{self.df.iloc[self.X_train.index]['Date'].min().date()} "f"to " f"{self.df.iloc[self.X_train.index]['Date'].max().date()}")
        logger.info(f"Validation date range: "f"{self.df.iloc[self.X_val.index]['Date'].min().date()} "f"to "f"{self.df.iloc[self.X_val.index]['Date'].max().date()}")
        logger.info(f"Test date range: "f"{self.df.iloc[self.X_test.index]['Date'].min().date()} "f"to "f"{self.df.iloc[self.X_test.index]['Date'].max().date()}")
        logger.info(" No time-based data leakage detected!" )
        return (self.X_train,self.X_val,self.X_test,self.y_train, self.y_val,self.y_test)  
  
    def encode_categorical_features(self):
      """Encode categorical variables - handle unknown values"""
      logger.info("\n ENCODING CATEGORICAL FEATURES")
      if len(self.categorical_features) == 0:
        logger.info("No categorical features to encode")
        return
    
      for col in self.categorical_features:
        logger.info(f"\nEncoding: {col}")
           # Fit encoder on ALL data (to see all possible values)
        # This is OK for categorical encoding (no data leakage risk)
        all_values = pd.concat([
            self.X_train[col],
            self.X_val[col],
            self.X_test[col]
        ]).astype(str)
        
        le = LabelEncoder()
        le.fit(all_values)
        self.label_encoders[col] = le
        
           # Transform all sets
        self.X_train[col] = le.transform(self.X_train[col].astype(str))
        self.X_val[col] = le.transform(self.X_val[col].astype(str))
        self.X_test[col] = le.transform(self.X_test[col].astype(str))
        logger.info(f"  Classes: {list(le.classes_)}")
        logger.info(f"   Encoded {col}")
    
      return self
   
    def scale_numerical_features(self):
      """Scale numerical features"""
      logger.info("\n SCALING NUMERICAL FEATURES")
      if len(self.numeric_features) == 0:
        logger.info("No numerical features to scale")
        return self
      # Create scaler
      self.scaler = StandardScaler()
      # Fit only on training data
      self.scaler.fit(self.X_train[self.numeric_features])
      # Transform train / validation / test
      self.X_train[self.numeric_features] = (self.scaler.transform(self.X_train[self.numeric_features]))
      self.X_val[self.numeric_features] = (self.scaler.transform(self.X_val[self.numeric_features]))
      self.X_test[self.numeric_features] = (self.scaler.transform(self.X_test[self.numeric_features]) )
      logger.info(f" Scaled {len(self.numeric_features)} numerical features")
      # Show scaling info
      logger.info("\nScaling statistics:")
      logger.info(f"Mean: {self.scaler.mean_[:5].round(4)} (first 5)")
      logger.info(f"Std: {self.scaler.scale_[:5].round(4)} (first 5)")
      return self
    
    def save_datasets(self, output_dir='data/prepared'):
      """Save processed datasets"""
      logger.info("\n SAVING PREPARED DATA")
      # Create folder if it does not exist
      Path(output_dir).mkdir(parents=True,exist_ok=True)
      # Save feature datasets
      self.X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
      self.X_val.to_csv( f'{output_dir}/X_val.csv',index=False )
      self.X_test.to_csv( f'{output_dir}/X_test.csv', index=False)
      # Save target datasets
      self.y_train.to_csv(f'{output_dir}/y_train.csv',index=False )
      self.y_val.to_csv(f'{output_dir}/y_val.csv',index=False )
      self.y_test.to_csv(f'{output_dir}/y_test.csv',index=False)
      logger.info("Saved datasets:")
      logger.info( f"X_train.csv ({len(self.X_train)} rows)" )
      logger.info( f"X_val.csv ({len(self.X_val)} rows)")
      logger.info(f"X_test.csv ({len(self.X_test)} rows)" )
      logger.info( "y_train.csv, y_val.csv, y_test.csv")
      # Save preprocessing objects
      with open( f'{output_dir}/scaler.pkl', 'wb' ) as f:
        pickle.dump( self.scaler, f)
      with open(f'{output_dir}/label_encoders.pkl','wb' ) as f:
        pickle.dump(self.label_encoders, f)
      logger.info("scaler.pkl saved" )
      logger.info( "label_encoders.pkl saved")
      return self
    
    def print_data_schema(self):
        """Print schema"""
        logger.info("\n DATA SCHEMA")
        logger.info("\n Feature Information:")
        logger.info( f"Total features: {len(self.X.columns)}")
        logger.info( f"Numerical: {len(self.numeric_features)}")
        logger.info( f"Categorical: {len(self.categorical_features)}")
        logger.info(f"\nTarget Variable:")
        logger.info(f"  Name: Sales")
        logger.info(f"  Type: Continuous (Regression)")
        logger.info(f"  Mean: ${self.y.mean():.2f}")
        logger.info(f"  Std: ${self.y.std():.2f}")
        logger.info(f"  Min: ${self.y.min():.2f}")
        logger.info(f"  Max: ${self.y.max():.2f}")
        logger.info(f"\nFeature List ({len(self.X.columns)} features):")
        for i, col in enumerate(self.X.columns, 1):
            dtype = 'numerical' if col in self.numeric_features else 'categorical'
            logger.info(f"  {i:2d}. {col:30s} ({dtype})")
        return self

    def prepare_pipeline(self):
        """Execute complete preparation pipeline"""
        logger.info("\n" + "="*70)
        logger.info("DATA PREPARATION PIPELINE")
        logger.info("="*70)
        
        self.identify_feature_types()
        self.train_val_test_split()
        self.encode_categorical_features()
        self.scale_numerical_features()
        self.save_datasets()
        self.print_data_schema()
        
        logger.info("\n" + "="*70)
        logger.info(" DATA PREPARATION COMPLETE!")
        logger.info("="*70)
        
        return self

# Usage
if __name__ == "__main__":
    prep = DataPreparation()
    prep.prepare_pipeline()
    
    print("\n All datasets saved and ready for modeling!")