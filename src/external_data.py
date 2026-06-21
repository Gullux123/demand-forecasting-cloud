import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalDataCollector:
    """Collect external data: weather, festivals, sports events"""
    
    @staticmethod
    def get_weather_data(lat, lon, start_date, end_date):
        """
        Fetch weather data from Open-Meteo API (FREE - no authentication needed!)
        
        Args:
            lat, lon: Latitude/Longitude (e.g., Rossmann stores in Germany: 50°N, 10°E)
            start_date, end_date: datetime objects
        
        Returns:
            DataFrame with weather data
        """
        logger.info(f"Fetching weather data from {start_date.date()} to {end_date.date()}...")
        
        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Open-Meteo API endpoint (FREE!)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date={start_str}&end_date={end_str}&"
            f"daily=temperature_2m_max,temperature_2m_min,"
            f"precipitation_sum,windspeed_10m_max&"
            f"timezone=Europe/Berlin"
        )
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Create DataFrame
            weather_df = pd.DataFrame({
                'Date': pd.to_datetime(data['daily']['time']),
                'temp_max': data['daily']['temperature_2m_max'],
                'temp_min': data['daily']['temperature_2m_min'],
                'rainfall': data['daily']['precipitation_sum'],
                'wind_speed': data['daily']['windspeed_10m_max']
            })
            
            logger.info(f" Fetched weather data: {len(weather_df)} days")
            return weather_df
            
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return None
    
    @staticmethod
    def create_festival_calendar():
        """
        Create calendar of important Indian festivals and holidays
        These significantly impact retail demand!
        
        Note: Actual dates vary by lunar calendar - using approximate dates
        """
        logger.info("Creating Indian festival calendar...")
        
        festivals = [
            # 2013
            ('2013-01-01', 'New Year', 'national'),
            ('2013-01-26', 'Republic Day', 'national'),
            ('2013-03-25', 'Holi', 'religious'),
            ('2013-03-29', 'Good Friday', 'religious'),
            ('2013-04-11', 'Eid ul-Fitr', 'religious'),
            ('2013-05-30', 'Buddha Purnima', 'religious'),
            ('2013-08-15', 'Independence Day', 'national'),
            ('2013-09-16', 'Onam', 'regional'),
            ('2013-10-14', 'Diwali', 'major'),  # BIG SALES SPIKE!
            ('2013-11-04', 'Diwali', 'major'),
            ('2013-12-25', 'Christmas', 'festival'),
            
            # 2014
            ('2014-01-01', 'New Year', 'national'),
            ('2014-01-26', 'Republic Day', 'national'),
            ('2014-03-17', 'Holi', 'religious'),
            ('2014-04-18', 'Good Friday', 'religious'),
            ('2014-04-29', 'Eid ul-Fitr', 'religious'),
            ('2014-05-19', 'Buddha Purnima', 'religious'),
            ('2014-08-15', 'Independence Day', 'national'),
            ('2014-09-08', 'Onam', 'regional'),
            ('2014-10-02', 'Gandhi Jayanti', 'national'),
            ('2014-10-23', 'Diwali', 'major'),  #  BIG SALES SPIKE!
            ('2014-11-06', 'Diwali', 'major'),
            ('2014-12-25', 'Christmas', 'festival'),
            
            # 2015
            ('2015-01-01', 'New Year', 'national'),
            ('2015-01-26', 'Republic Day', 'national'),
            ('2015-03-06', 'Holi', 'religious'),
            ('2015-04-10', 'Good Friday', 'religious'),
            ('2015-07-07', 'Eid ul-Fitr', 'religious'),
            ('2015-08-15', 'Independence Day', 'national'),
            ('2015-09-27', 'Onam', 'regional'),
            ('2015-10-02', 'Gandhi Jayanti', 'national'),
            ('2015-11-11', 'Diwali', 'major'),  #  BIG SALES SPIKE!
            ('2015-12-25', 'Christmas', 'festival'),
        ]
        
        festival_df = pd.DataFrame({
            'Date': pd.to_datetime([f[0] for f in festivals]),
            'festival_name': [f[1] for f in festivals],
            'festival_type': [f[2] for f in festivals]
        })
        
        festival_df['is_festival'] = 1
        
        logger.info(f" Created festival calendar: {len(festival_df)} events")
        return festival_df
    
    @staticmethod
    def create_ipl_schedule():
        """
        IPL (Indian Premier League) cricket matches cause MASSIVE demand spikes!
        
        Cold drinks, snacks, food delivery see 3-5x surge on IPL match days
        This is INDIA-SPECIFIC insight - not in typical forecasting models!
        """
        logger.info("Creating IPL cricket match schedule...")
        
        # IPL match dates (approximate - actual dates vary by season)
        ipl_matches = [
            # IPL 2013 (April-May)
            '2013-04-03', '2013-04-04', '2013-04-05', '2013-04-07', '2013-04-08',
            '2013-04-12', '2013-04-14', '2013-04-19', '2013-04-21', '2013-04-28',
            '2013-05-12', '2013-05-23', '2013-05-26',  # Finals
            
            # IPL 2014 (March-May)
            '2014-03-16', '2014-03-17', '2014-03-21', '2014-03-23', '2014-03-28',
            '2014-03-30', '2014-04-04', '2014-04-06', '2014-04-11', '2014-04-13',
            '2014-05-18', '2014-05-25', '2014-05-30',  # Finals
            
            # IPL 2015 (March-May)
            '2015-03-23', '2015-03-24', '2015-03-28', '2015-03-29', '2015-04-04',
            '2015-04-05', '2015-04-10', '2015-04-12', '2015-04-17', '2015-04-19',
            '2015-05-17', '2015-05-24', '2015-05-29',  # Finals
        ]
        
        ipl_df = pd.DataFrame({
            'Date': pd.to_datetime(ipl_matches),
            'is_ipl_match': 1
        })
        
        logger.info(f" Created IPL schedule: {len(ipl_df)} match days")
        return ipl_df
    
    @staticmethod
    def create_school_holidays():
        """School holidays in Germany affect shopping patterns"""
        logger.info("Creating school holidays calendar...")
        
        # German school holidays (approximate)
        holidays = [
            ('2013-04-27', '2013-04-30', 'Easter'),  # Easter holidays
            ('2013-05-30', '2013-05-30', 'Ascension'),  # Ascension Day
            ('2013-10-03', '2013-10-03', 'German Unity'),
            ('2013-12-23', '2014-01-06', 'Christmas'),
            
            ('2014-04-19', '2014-04-22', 'Easter'),
            ('2014-05-29', '2014-05-29', 'Ascension'),
            ('2014-10-03', '2014-10-03', 'German Unity'),
            ('2014-12-22', '2015-01-06', 'Christmas'),
            
            ('2015-04-10', '2015-04-13', 'Easter'),
            ('2015-05-14', '2015-05-14', 'Ascension'),
            ('2015-10-03', '2015-10-03', 'German Unity'),
        ]
        
        holidays_list = []
        for start, end, name in holidays:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            dates = pd.date_range(start=start_date, end=end_date)
            for date in dates:
                holidays_list.append({
                    'Date': date,
                    'holiday_name': name,
                    'is_holiday': 1
                })
        
        holidays_df = pd.DataFrame(holidays_list)
        logger.info(f" Created holiday calendar: {len(holidays_df)} holiday days")
        return holidays_df

# Usage
if __name__ == "__main__":
    collector = ExternalDataCollector()
    
    # 1. Get weather data
    weather = collector.get_weather_data(
        lat=50.0,  # Germany latitude
        lon=10.0,  # Germany longitude
        start_date=datetime(2013, 1, 1),
        end_date=datetime(2015, 12, 31)
    )
    print(f"\n Weather data shape: {weather.shape}")
    print(weather.head())
    
    # 2. Get festivals
    festivals = collector.create_festival_calendar()
    print(f"\n Festival data shape: {festivals.shape}")
    print(festivals.head())
    
    # 3. Get IPL schedule
    ipl = collector.create_ipl_schedule()
    print(f"\n IPL schedule shape: {ipl.shape}")
    print(ipl.head())
    
    # 4. Get holidays
    holidays = collector.create_school_holidays()
    print(f"\n Holiday calendar shape: {holidays.shape}")
    print(holidays.head())
    
    print("\n All external data created successfully!")