# data/btc_data_loader.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class BTCDataLoader:
    """
    Data loader for BTC price data from CSV files.
    Loads, parses, and returns clean OHLCV DataFrame.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Optional path to CSV file (if None, searches data/ folder)
        """
        self.data_path = data_path

    def load_data(
        self,
        csv_path: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load BTC price data from CSV file.
        
        Args:
            csv_path: Path to CSV file (if None, searches data/ folder)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            Clean OHLCV DataFrame with datetime index, sorted by time.
            Columns: open, high, low, close, volume
        """
        # Determine file path
        file_path = csv_path or self.data_path
        
        if file_path is None:
            # Search data/ folder for CSV files
            data_dir = Path("data")
            csv_files = list(data_dir.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in {data_dir}. "
                    "Please provide csv_path or place CSV file in data/ folder."
                )
            
            # Use first CSV found, or prefer specific names
            preferred_names = [
                "btcusd_1min_volspikes.csv",
                "btc_data.csv",
                "btcusdt.csv",
                "btc.csv"
            ]
            
            file_path = None
            for name in preferred_names:
                candidate = data_dir / name
                if candidate.exists():
                    file_path = candidate
                    break
            
            if file_path is None:
                file_path = csv_files[0]
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise IOError(f"Error loading CSV from {file_path}: {e}")
        
        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")
        
        # Standardize column names (case-insensitive mapping)
        col_mapping = {
            'time': 'timestamp', 'Time': 'timestamp', 'TIME': 'timestamp',
            'date': 'timestamp', 'Date': 'timestamp', 'DATE': 'timestamp',
            'datetime': 'timestamp', 'DateTime': 'timestamp', 'DATETIME': 'timestamp',
            'Open': 'open', 'OPEN': 'open',
            'High': 'high', 'HIGH': 'high',
            'Low': 'low', 'LOW': 'low',
            'Close': 'close', 'CLOSE': 'close',
            'Volume': 'volume', 'VOLUME': 'volume', 'vol': 'volume', 'VOL': 'volume'
        }
        
        # Rename columns (only if they exist)
        rename_dict = {old: new for old, new in col_mapping.items() if old in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Check for required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Found columns: {df.columns.tolist()}"
            )
        
        # Parse timestamp column
        if 'timestamp' in df.columns:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Set as index
            df = df.set_index('timestamp')
        else:
            # Try to parse index as datetime
            df.index = pd.to_datetime(df.index, errors='coerce')
        
        # Remove rows with invalid timestamps
        df = df[df.index.notna()]
        
        if df.empty:
            raise ValueError("No valid timestamps found in CSV")
        
        # Ensure correct dtypes for OHLCV
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
            print(f"Warning: Found {n_invalid} rows with invalid OHLC relationships. Removing...")
            df = df[~invalid_mask]
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df.index <= end_dt]
        
        # Sort by time (ascending)
        df = df.sort_index()
        
        # Select only OHLCV columns (in correct order)
        df = df[required_cols]
        
        # Ensure final dtypes
        df = df.astype({
            'open': np.float64,
            'high': np.float64,
            'low': np.float64,
            'close': np.float64,
            'volume': np.float64
        })
        
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate that dataframe has required structure and valid data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Index must be DatetimeIndex, got {type(df.index)}")
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for NaN values
        if df[required_cols].isna().any().any():
            raise ValueError("DataFrame contains NaN values in OHLCV columns")
        
        # Check OHLC relationships
        invalid = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['volume'] < 0)
        )
        
        if invalid.any():
            n_invalid = invalid.sum()
            raise ValueError(f"Found {n_invalid} rows with invalid OHLC relationships")
        
        # Check index is sorted
        if not df.index.is_monotonic_increasing:
            raise ValueError("Index is not sorted in ascending order")
        
        return True


def load_btc_csv(csv_path: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load BTC CSV data.
    
    Args:
        csv_path: Path to CSV file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        Clean OHLCV DataFrame with datetime index
    """
    loader = BTCDataLoader()
    return loader.load_data(csv_path=csv_path, start_date=start_date, end_date=end_date)
