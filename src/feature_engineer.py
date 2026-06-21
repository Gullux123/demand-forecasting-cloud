import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineer 30+ features for demand forecasting"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        logger.info(f"Initialized with {len(self.df)} rows")
    
    def add_time_features(self):
        """Extract date/time based features"""
        logger.info("Adding time features...")
        
        self.df['year'] = self.df['Date'].dt.year
        self.df['month'] = self.df['Date'].dt.month
        self.df['quarter'] = self.df['Date'].dt.quarter
        self.df['day'] = self.df['Date'].dt.day
        self.df['dayofweek'] = self.df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['dayofyear'] = self.df['Date'].dt.dayofyear
        self.df['weekofyear'] = self.df['Date'].dt.isocalendar().week
        self.df['is_month_start'] = self.df['Date'].dt.is_month_start.astype(int)
        self.df['is_month_end'] = self.df['Date'].dt.is_month_end.astype(int)
        self.df['is_quarter_start'] = self.df['Date'].dt.is_quarter_start.astype(int)
        self.df['is_quarter_end'] = self.df['Date'].dt.is_quarter_end.astype(int)
        self.df['is_year_start'] = self.df['Date'].dt.is_year_start.astype(int)
        self.df['is_year_end'] = self.df['Date'].dt.is_year_end.astype(int)
        
        logger.info(f" Added 13 time features")
        return self
    
    def add_cyclical_features(self):
        """Convert cyclical time features to sine/cosine (prevents misleading distance)"""
        logger.info("Adding cyclical features...")
        
        # Day of week (7-day cycle)
        self.df['dow_sin'] = np.sin(2 * np.pi * self.df['dayofweek'] / 7)
        self.df['dow_cos'] = np.cos(2 * np.pi * self.df['dayofweek'] / 7)
        
        # Month (12-month cycle)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Day of year (365-day cycle)
        self.df['doy_sin'] = np.sin(2 * np.pi * self.df['dayofyear'] / 365)
        self.df['doy_cos'] = np.cos(2 * np.pi * self.df['dayofyear'] / 365)
        
        # Week of year (52-week cycle)
        self.df['woy_sin'] = np.sin(2 * np.pi * self.df['weekofyear'] / 52)
        self.df['woy_cos'] = np.cos(2 * np.pi * self.df['weekofyear'] / 52)
        
        logger.info(f" Added 8 cyclical features")
        return self
    
    def add_binary_time_features(self):
        """Weekend, weekday, etc."""
        logger.info("Adding binary time features...")
        
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
        self.df['is_weekday'] = (self.df['dayofweek'] < 5).astype(int)
        self.df['is_monday'] = (self.df['dayofweek'] == 0).astype(int)
        self.df['is_friday'] = (self.df['dayofweek'] == 4).astype(int)
        self.df['is_saturday'] = (self.df['dayofweek'] == 5).astype(int)
        self.df['is_sunday'] = (self.df['dayofweek'] == 6).astype(int)
        
        logger.info(f" Added 6 binary time features")
        return self
    
    def add_external_signal_features(self):
        """Process weather and event signals"""
        logger.info("Adding external signal features...")
        
        # Weather features (already in df from merge)
        if 'temp_max' in self.df.columns:
            self.df['temp_range'] = self.df['temp_max'] - self.df['temp_min']
            self.df['is_hot'] = (self.df['temp_max'] > 25).astype(int)
            self.df['is_cold'] = (self.df['temp_max'] < 5).astype(int)
        
        if 'rainfall' in self.df.columns:
            self.df['is_rainy'] = (self.df['rainfall'] > 5).astype(int)
            self.df['is_heavy_rain'] = (self.df['rainfall'] > 20).astype(int)
            self.df['rainfall_category'] = pd.cut(
                self.df['rainfall'], 
                bins=[-np.inf, 0, 5, 20, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(int)
        
        # Event flags (already in df from merge)
        if 'is_festival' in self.df.columns:
            self.df['is_festival'] = self.df['is_festival'].fillna(0).astype(int)
        
        if 'is_ipl_match' in self.df.columns:
            self.df['is_ipl_match'] = self.df['is_ipl_match'].fillna(0).astype(int)
        
        if 'is_holiday' in self.df.columns:
            self.df['is_holiday'] = self.df['is_holiday'].fillna(0).astype(int)
        
        logger.info(f" Added external signal features")
        return self
    
    def add_lag_features(self, lags=[1, 7, 14, 30]):
        """Sales from previous days/weeks/months"""
        logger.info(f"Adding lag features: {lags}...")
        
        for lag in lags:
            self.df[f'sales_lag_{lag}'] = self.df.groupby('Store')['Sales'].shift(lag)
            self.df[f'customers_lag_{lag}'] = self.df.groupby('Store')['Customers'].shift(lag)
        
        logger.info(f" Added {len(lags)*2} lag features")
        return self
    
    def add_rolling_features(self, windows=[7, 14, 30]):
        """Rolling average and std dev"""
        logger.info(f"Adding rolling features: {windows}...")
        
        for window in windows:
            self.df[f'sales_ma_{window}'] = (
                self.df.groupby('Store')['Sales']
                .rolling(window=window, min_periods=1).mean().reset_index(drop=True)
            )
            
            self.df[f'sales_std_{window}'] = (
                self.df.groupby('Store')['Sales']
                .rolling(window=window, min_periods=1).std().reset_index(drop=True)
            )
            
            self.df[f'customers_ma_{window}'] = (
                self.df.groupby('Store')['Customers']
                .rolling(window=window, min_periods=1).mean().reset_index(drop=True)
            )
        
        logger.info(f" Added {len(windows)*3} rolling features")
        return self
    
    def add_trend_features(self):
        """Trend and store-specific features"""
        logger.info("Adding trend features...")

        # Global trend (overall row number)
        self.df['trend'] = np.arange(len(self.df))

        # Store-specific trend
        self.df['store_trend'] = (
         self.df.groupby('Store').cumcount()
        )

        # Days since last event
        for col in ['is_festival', 'is_holiday']:

         if col in self.df.columns:

            self.df[f'days_since_{col}'] = (
                self.df.groupby('Store')[col]
                .transform(
                    lambda x: (~x.astype(bool)).cumsum()
                )
            )

        logger.info("Added trend features")

        return self
    def fill_nan_values(self):
         """Fill NaN values instead of removing rows"""
         initial_nans = self.df.isnull().sum().sum()
    
         # Fill lag features with forward fill (carry previous value)
         lag_cols = [col for col in self.df.columns if 'lag' in col]
         for col in lag_cols:
            self.df[col] = self.df.groupby('Store')[col].fillna(method='ffill')
    
         # Fill rolling stats with forward fill
         rolling_cols = [col for col in self.df.columns if 'ma_' in col or 'std_' in col]
         for col in rolling_cols:
            self.df[col] = self.df.groupby('Store')[col].fillna(method='ffill')
    
         # Fill remaining NaN with 0 (for trend, cyclical features)
         self.df = self.df.fillna(0)
    
         final_nans = self.df.isnull().sum().sum()
         logger.info(f"✓ Filled {initial_nans} NaN values (kept all rows)")
         logger.info(f"✓ Remaining NaN: {final_nans}")
         return self
    
    def get_features(self):
        """Execute complete feature engineering"""
        return (self.add_time_features()
        .add_cyclical_features()
        .add_binary_time_features()
        .add_external_signal_features()
        .add_lag_features()
        .add_rolling_features()
        .add_trend_features()
        .fill_nan_values()  # ← CHANGED!
        .df)
    
    def get_feature_summary(self):
        """Summary of features created"""
        logger.info("\n" + "="*70)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total features created: {len(self.df.columns)}")
        logger.info(f"Total rows: {len(self.df)}")
        logger.info(f"\nFeatures by category:")
        logger.info(f"  - Time features: 13")
        logger.info(f"  - Cyclical features: 8")
        logger.info(f"  - Binary features: 6")
        logger.info(f"  - External signals: 10+")
        logger.info(f"  - Lag features: 8")
        logger.info(f"  - Rolling features: 9")
        logger.info(f"  - Trend features: 3+")
        logger.info(f"  - Original features: ~5")
        logger.info("="*70 + "\n")
        
        return self

# Usage
if __name__ == "__main__":
    from data_loader import DataLoader
    from external_data import ExternalDataCollector
    
    # Load data
    logger.info("\n Loading original data...")
    loader = DataLoader(sample_size=100000)  # Use 100K for feature engineering
    loader.run_pipeline()
    train, test, store = loader.get_data()
    
    # Get external data
    logger.info("\n Getting external data...")
    collector = ExternalDataCollector()
    
    weather = collector.get_weather_data(
        lat=50.0, lon=10.0,
        start_date=pd.to_datetime('2013-01-01'),
        end_date=pd.to_datetime('2015-12-31')
    )
    
    festivals = collector.create_festival_calendar()
    ipl = collector.create_ipl_schedule()
    holidays = collector.create_school_holidays()
    
    # Merge external data (for train)
    logger.info("\n Merging external data...")
    train = train.merge(weather, on='Date', how='left')
    train = train.merge(festivals, on='Date', how='left')
    train = train.merge(ipl, on='Date', how='left')
    train = train.merge(holidays, on='Date', how='left')
    
    # Engineer features
    logger.info("\n Engineering features...")
    fe = FeatureEngineer(train)
    df_features = fe.get_features()
    fe.get_feature_summary()
    
    # Save
    logger.info("\n Saving processed data...")
    df_features.to_csv('data/processed/train_features.csv', index=False)
    logger.info(f" Saved to data/processed/train_features.csv")
    
    print(f"\n FEATURE ENGINEERING COMPLETE!")
    print(f"Output shape: {df_features.shape}")
    print(f"\nFirst few rows:")
    print(df_features.head())
    print(f"\nColumn names ({len(df_features.columns)} total):")
    print(df_features.columns.tolist())