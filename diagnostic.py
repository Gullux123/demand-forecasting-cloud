import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("DIAGNOSTIC TEST")
print("=" * 70)

# Test 1: Load raw data
print("\n1️⃣ Loading raw data...")
try:
    train = pd.read_csv('data/raw/train.csv', nrows=50000, parse_dates=['Date'])
    store = pd.read_csv('data/raw/store.csv')
    print(f"✓ Loaded train: {train.shape}, store: {store.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit()

# Test 2: Merge
print("\n2️⃣ Merging...")
try:
    train = train.merge(store, on='Store', how='left')
    print(f"✓ Merged: {train.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit()

# Test 3: Add time features
print("\n3️⃣ Adding time features...")
try:
    train['year'] = train['Date'].dt.year
    train['month'] = train['Date'].dt.month
    train['dayofweek'] = train['Date'].dt.dayofweek
    print(f"✓ Time features added: {train.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit()

# Test 4: Add lag features
print("\n4️⃣ Adding lag features...")
try:
    train['sales_lag_1'] = train.groupby('Store')['Sales'].shift(1)
    print(f"✓ Lag features added")
    print(f"  Shape before dropna: {train.shape}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit()

# Test 5: Remove NaN
print("\n5️⃣ Removing NaN rows...")
try:
    initial_rows = len(train)
    train = train.dropna()
    final_rows = len(train)
    print(f"✓ Removed {initial_rows - final_rows} NaN rows")
    print(f"  Remaining rows: {final_rows}")
    
    if final_rows == 0:
        print(f"\n⚠️ ERROR: ALL ROWS REMOVED!")
        print(f"This is the problem! Check for missing values in external data.")
        exit()
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit()

# Test 6: Save
print("\n6️⃣ Saving to CSV...")
try:
    train.to_csv('data/processed/diagnose_output.csv', index=False)
    size = Path('data/processed/diagnose_output.csv').stat().st_size / (1024*1024)
    print(f"✓ Saved: {size:.2f} MB")
    print(f"  Rows: {len(train)}")
    print(f"  Columns: {len(train.columns)}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit()

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)