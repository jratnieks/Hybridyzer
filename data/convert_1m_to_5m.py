import pandas as pd
import numpy as np
from pathlib import Path

# Get the script's directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Define paths relative to project root
INPUT_FILE = PROJECT_ROOT / "data" / "btcusd_1min.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "btcusd_5min.csv"

def load_1m_data(path):
    """
    Load 1-minute OHLCV data from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    
    print(f"Loading raw 1m data from {path}...")
    df = pd.read_csv(path)

    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Fix timestamp format
    if 'timestamp' not in df.columns:
        raise ValueError("CSV must have 'timestamp' column")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce')

    # Remove rows with invalid timestamps
    df = df[df["timestamp"].notna()]
    
    if df.empty:
        raise ValueError("No valid timestamps found in CSV")

    # Sort by time
    df = df.sort_values("timestamp")

    # Remove duplicates (keep first occurrence)
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep='first')
    n_after = len(df)
    if n_before != n_after:
        print(f"  Removed {n_before - n_after:,} duplicate timestamps")

    # Set timestamp as index
    df = df.set_index("timestamp")
    
    # Ensure OHLCV columns are numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN in OHLCV
    df = df.dropna(subset=required_cols)
    
    if df.empty:
        raise ValueError("No valid OHLCV data after cleaning")
    
    # Validate OHLC relationships
    invalid_mask = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close']) |
        (df['volume'] < 0)
    )
    
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        print(f"  Warning: Found {n_invalid:,} rows with invalid OHLC relationships. Removing...")
        df = df[~invalid_mask]
    
    if df.empty:
        raise ValueError("No valid data after removing invalid OHLC relationships")
    
    print(f"  Loaded {len(df):,} valid 1m bars")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def resample_to_5m(df):
    """
    Resample 1-minute OHLCV data to 5-minute bars.
    
    Args:
        df: DataFrame with 1-minute OHLCV data (datetime index)
        
    Returns:
        DataFrame with 5-minute OHLCV data
    """
    print("Resampling to 5m OHLCV...")
    
    # Check required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for resampling: {missing_cols}")

    # Resample to 5-minute bars
    # OHLCV aggregation rules:
    # - open: first value in the period
    # - high: maximum value in the period
    # - low: minimum value in the period
    # - close: last value in the period
    # - volume: sum of volumes in the period
    df_5m = df.resample("5T").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    # Drop any rows with missing OHLC (gaps in 1m data)
    n_before = len(df_5m)
    df_5m = df_5m.dropna(subset=["open", "high", "low", "close"])
    n_after = len(df_5m)
    
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after:,} rows with missing OHLC data")
    
    if df_5m.empty:
        raise ValueError("No valid 5m bars after resampling")
    
    # Validate OHLC relationships in resampled data
    invalid_mask = (
        (df_5m['high'] < df_5m['low']) |
        (df_5m['high'] < df_5m['open']) |
        (df_5m['high'] < df_5m['close']) |
        (df_5m['low'] > df_5m['open']) |
        (df_5m['low'] > df_5m['close']) |
        (df_5m['volume'] < 0)
    )
    
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        print(f"  Warning: Found {n_invalid:,} invalid 5m bars. Removing...")
        df_5m = df_5m[~invalid_mask]
    
    if df_5m.empty:
        raise ValueError("No valid 5m bars after validation")
    
    print(f"  Generated {len(df_5m):,} valid 5m bars")
    print(f"  Date range: {df_5m.index.min()} to {df_5m.index.max()}")
    
    return df_5m


def main():
    """
    Main function to convert 1-minute data to 5-minute bars.
    """
    try:
        # Convert Path objects to strings for compatibility
        input_file = str(INPUT_FILE)
        output_file = str(OUTPUT_FILE)
        
        # Load 1-minute data
        df_1m = load_1m_data(input_file)
        print(f"\n1m rows loaded: {len(df_1m):,}")

        # Resample to 5-minute bars
        df_5m = resample_to_5m(df_1m)
        print(f"\n5m rows after resampling: {len(df_5m):,}")

        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to {output_path}...")
        df_5m.to_csv(output_path, index=True)
        
        print(f"\n✓ Successfully converted {len(df_1m):,} 1m bars to {len(df_5m):,} 5m bars")
        print(f"✓ Output saved to {output_path}")
        print("Done.")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()
