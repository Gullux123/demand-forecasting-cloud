import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:

    def __init__(self, data_dir='data/raw', sample_size=None):
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        self.train = None
        self.test = None
        self.store = None


    def load_raw_data(self):

        logger.info("Loading raw data files...")

        self.store = pd.read_csv(
            self.data_dir / 'store.csv'
        )

        logger.info(f"Loaded {len(self.store)} stores")


        nrows = self.sample_size if self.sample_size else None

        self.train = pd.read_csv(
            self.data_dir / 'train.csv',
            nrows=nrows,
            parse_dates=['Date']
        )

        logger.info(f"Loaded {len(self.train)} training records")


        self.test = pd.read_csv(
            self.data_dir / 'test.csv',
            parse_dates=['Date']
        )

        logger.info(f"Loaded {len(self.test)} test records")

        return self


    def merge_store_info(self):

        logger.info("Merging store information...")

        self.train = self.train.merge(
            self.store,
            on='Store',
            how='left'
        )

        self.test = self.test.merge(
            self.store,
            on='Store',
            how='left'
        )

        logger.info("Store info merged")

        return self


    def display_info(self):

        print("\n" + "="*60)
        print("DATASET INFORMATION")
        print("="*60)

        print("Shape:", self.train.shape)

        print(
            "Date Range:",
            self.train['Date'].min(),
            "to",
            self.train['Date'].max()
        )

        print("\nColumns:")
        print(self.train.columns.tolist())

        print("\nMissing Values:")
        print(self.train.isnull().sum())

        print("="*60)

        return self


    def get_data(self):
        return self.train, self.test, self.store


    def run_pipeline(self):

        return (
            self.load_raw_data()
            .merge_store_info()
            .display_info()
        )


if __name__ == "__main__":

    loader = DataLoader(sample_size=50000)

    loader.run_pipeline()

    train, test, store = loader.get_data()

    print("\nSUCCESS!")
    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Store:", store.shape)