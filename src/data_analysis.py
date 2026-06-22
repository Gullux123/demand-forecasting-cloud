import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Deep statistical analysis of demand forecasting data"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        logger.info(f"Initialized with {len(self.df)} rows")
    
    def analyze_store_performance(self):
        """Analyze sales by store"""
        logger.info("\n STORE PERFORMANCE ANALYSIS")
        
        store_stats = self.df.groupby('Store').agg({
            'Sales': ['mean', 'median', 'std', 'min', 'max'],
            'Customers': ['mean', 'median'],
            'Store': 'count'
        }).round(2)
        
        store_stats.columns = ['Sales_Mean', 'Sales_Median', 'Sales_Std', 
                               'Sales_Min', 'Sales_Max', 'Customers_Mean', 
                               'Customers_Median', 'Transaction_Count']
        
        logger.info("\nTop 10 stores by average sales:")
        top_stores = store_stats['Sales_Mean'].nlargest(10)
        for i, (store, sales) in enumerate(top_stores.items(), 1):
            logger.info(f"  {i:2d}. Store {store:4d} → ${sales:7.2f} avg sales")
        
        logger.info("\nBottom 10 stores by average sales:")
        bottom_stores = store_stats['Sales_Mean'].nsmallest(10)
        for i, (store, sales) in enumerate(bottom_stores.items(), 1):
            logger.info(f"  {i:2d}. Store {store:4d} → ${sales:7.2f} avg sales")
        
        return store_stats
    
    def analyze_temporal_patterns(self):
        """Analyze patterns over time"""
        logger.info("\n TEMPORAL PATTERNS ANALYSIS")
        
            # Daily sales
        daily_sales = self.df.groupby('Date')['Sales'].sum()
        logger.info(f"\nDaily Sales Statistics:")
        logger.info(f"  Mean: ${daily_sales.mean():,.2f}")
        logger.info(f"  Median: ${daily_sales.median():,.2f}")
        logger.info(f"  Std: ${daily_sales.std():,.2f}")
        logger.info(f"  Min: ${daily_sales.min():,.2f}")
        logger.info(f"  Max: ${daily_sales.max():,.2f}")
        
            # By day of week
        dow_sales = self.df.groupby('dayofweek')['Sales'].mean()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        logger.info(f"\nAverage Sales by Day of Week:")
        for i, (dow, sales) in enumerate(dow_sales.items()):
          logger.info(f"  {day_names[int(dow)]} → ${sales:7.2f}")
        # By month
        monthly_sales = self.df.groupby('month')['Sales'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        logger.info(f"\nAverage Sales by Month:")
        for month, sales in monthly_sales.items():
            logger.info(f"  {month_names[int(month)-1]} → ${sales:7.2f}")
        
        return daily_sales, dow_sales, monthly_sales
    
    def analyze_external_signals(self):
        """Analyze impact of external signals"""
        logger.info("\nEXTERNAL SIGNALS IMPACT ANALYSIS")

        signals = {
        'Festival': 'is_festival',
        'IPL Match': 'is_ipl_match',
        'Holiday': 'is_holiday',
        'Weekend': 'is_weekend',
        'Promo': 'Promo'
        }

        for signal_name, column in signals.items():
         if column in self.df.columns:

            if self.df[column].dtype in [np.float64, np.int64]:

                normal = self.df[self.df[column] == 0]['Sales'].mean()

                event = self.df[self.df[column] == 1]['Sales'].mean()

                if not np.isnan(event):

                    impact = ((event / normal) - 1) * 100

                    logger.info(f"\n{signal_name}:")
                    logger.info(f"  Normal day: ${normal:.2f}")
                    logger.info(f"  {signal_name} day: ${event:.2f}")
                    logger.info(f"  Impact: {impact:+.1f}%")

        return None
    
    def detect_outliers(self):
        """Detect outliers using IQR method"""
        logger.info("\nOUTLIER DETECTION (IQR Method)")

        Q1 = self.df['Sales'].quantile(0.25)
        Q3 = self.df['Sales'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.df[
        (self.df['Sales'] < lower_bound) |
        (self.df['Sales'] > upper_bound)
        ]

        logger.info("\nOutlier Statistics:")
        logger.info(f"  Q1 (25th percentile): ${Q1:.2f}")
        logger.info(f"  Q3 (75th percentile): ${Q3:.2f}")
        logger.info(f"  IQR: ${IQR:.2f}")
        logger.info(f"  Lower bound: ${lower_bound:.2f}")
        logger.info(f"  Upper bound: ${upper_bound:.2f}")
        logger.info(
         f"  Total outliers: {len(outliers)} "
         f"({len(outliers)/len(self.df)*100:.2f}%)"
        )

        if len(outliers) > 0:
         logger.info(
            f"  Outlier sales range: "
            f"${outliers['Sales'].min():.2f} - "
            f"${outliers['Sales'].max():.2f}"
         )

        return outliers    
    def test_normality(self):
        """Test if sales distribution is normal"""
        logger.info("\nNORMALITY TEST (Shapiro-Wilk)")

        # Sample if data is large (Shapiro-Wilk is slow for huge datasets)
        sample = self.df['Sales'].sample(
        min(5000, len(self.df)),
        random_state=42
        )

        stat, p_value = stats.shapiro(sample)

        logger.info("\nShapiro-Wilk Test:")
        logger.info(f"  Test statistic: {stat:.6f}")
        logger.info(f"  P-value: {p_value:.10f}")

        if p_value < 0.05:
         logger.info(
            " Sales NOT normally distributed (p < 0.05)"
         )
        else:
         logger.info(
            " Sales approximately normal (p >= 0.05)"
         )

        return stat, p_value
    
    def analyze_skewness_kurtosis(self):
        """Analyze distribution shape"""
        logger.info("\nDISTRIBUTION SHAPE ANALYSIS")

        skewness = stats.skew(self.df['Sales'])
        kurtosis = stats.kurtosis(self.df['Sales'])

        logger.info(f"\nSkewness: {skewness:.4f}")

        if abs(skewness) < 0.5:
         logger.info("  → Fairly symmetric")
        elif skewness > 0:
         logger.info("  → Right-skewed (tail on right)")
        else:
         logger.info("  → Left-skewed (tail on left)")

        logger.info(f"\nKurtosis: {kurtosis:.4f}")

        if kurtosis > 0:
         logger.info("  → Heavy tails (leptokurtic)")
        else:
         logger.info("  → Light tails (platykurtic)")

        return skewness, kurtosis
    
    def correlation_analysis(self):
        """Top correlations with sales"""
        logger.info("\nCORRELATION ANALYSIS")

        df_numeric = self.df.select_dtypes(include=[np.number])

        correlations = df_numeric.corr()['Sales'].sort_values(
        ascending=False
        )

        logger.info("\nTop 15 Features by Correlation with Sales:")

        for i, (feature, corr) in enumerate(
         correlations.head(15).items(), 1
        ):
         if feature != 'Sales':
            logger.info(
                f"  {i:2d}. {feature:30s} → {corr:7.4f}"
            )

        return correlations
    
    def generate_report(self):
        """Generate complete analysis report"""
        logger.info("\n" + "="*70)
        logger.info("DAY 3: STATISTICAL ANALYSIS COMPLETE")
        logger.info("="*70)
        
        store_stats = self.analyze_store_performance()
        daily_sales, dow_sales, monthly_sales = self.analyze_temporal_patterns()
        self.analyze_external_signals()
        outliers = self.detect_outliers()
        stat, p_value = self.test_normality()
        skewness, kurtosis = self.analyze_skewness_kurtosis()
        correlations = self.correlation_analysis()
        
        logger.info("\n" + "="*70)
        logger.info("KEY INSIGHTS")
        logger.info("="*70)
        logger.info(f" Analyzed {len(self.df):,} transactions")
        logger.info(f" {len(self.df['Store'].unique())} unique stores")
        logger.info(f" {len(outliers)} outliers detected ({len(outliers)/len(self.df)*100:.2f}%)")
        logger.info(f" Sales distribution is {'NOT ' if p_value < 0.05 else ''}normal")
        logger.info(f" Top feature: {correlations.index[1]} (corr: {correlations.values[1]:.4f})")
        logger.info("="*70)
        
        return {
            'store_stats': store_stats,
            'daily_sales': daily_sales,
            'dow_sales': dow_sales,
            'monthly_sales': monthly_sales,
            'outliers': outliers,
            'correlations': correlations,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

# Usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed/train_features.csv')
    
    analyzer = DataAnalyzer(df)
    results = analyzer.generate_report()
    
    print("\n Analysis complete!")