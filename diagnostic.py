# diagnostic.py
"""Quick diagnostic script to check data statistics."""

import pandas as pd
import numpy as np
from data.btc_data_loader import load_btc_csv
from core.regime_detector import wilder_atr
from pathlib import Path

# Load data
csv_paths = [
    "data/btcusd_4H.csv",
    "data/btcusd_1min.csv",
    "data/btc_1m.csv",
]

df = None
for path in csv_paths:
    if Path(path).exists():
        df = load_btc_csv(path)
        print(f"Loaded from: {path}")
        break

if df is None:
    data_dir = Path("data")
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        df = load_btc_csv(str(csv_files[0]))
        print(f"Loaded from: {csv_files[0]}")
    else:
        raise FileNotFoundError("No CSV files found in data/ directory")

print(f"\nData shape: {df.shape}")
print(f"Data index range: {df.index[0]} to {df.index[-1]}")

# 1. Percentage changes
print("\n" + "="*60)
print("1. ABSOLUTE PERCENTAGE CHANGES")
print("="*60)
pct_changes = df['close'].pct_change().abs()
print(pct_changes.describe())

# 2. ATR
print("\n" + "="*60)
print("2. ATR (Wilder's, period=14)")
print("="*60)
atr = wilder_atr(df, 14)
print(atr.describe())

# 3. Example thresholds
print("\n" + "="*60)
print("3. EXAMPLE THRESHOLDS (ATR * 0.5)")
print("="*60)
print("Index | ATR | Threshold (ATR * 0.5)")
print("-" * 60)
for i in range(min(10, len(atr))):
    atr_val = atr.iloc[i]
    threshold = atr_val * 0.5
    idx = atr.index[i]
    print(f"{i:3d} | {atr_val:.6f} | {threshold:.6f} | {idx}")

# Additional diagnostics
print("\n" + "="*60)
print("4. ADDITIONAL DIAGNOSTICS")
print("="*60)
print(f"ATR NaN count: {atr.isna().sum()}")
print(f"ATR first non-NaN index: {atr.first_valid_index()}")
print(f"ATR first non-NaN value: {atr[atr.first_valid_index()] if atr.first_valid_index() else 'N/A'}")
print(f"\nClose price range:")
print(f"  Min: {df['close'].min():.2f}")
print(f"  Max: {df['close'].max():.2f}")
print(f"  Mean: {df['close'].mean():.2f}")

