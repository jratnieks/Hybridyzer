"""
Synthetic feature data generators for testing.

These generators produce tiny, deterministic DataFrames that simulate
the feature matrices produced by FeatureStore without heavy GPU computation.

WHY THIS EXISTS:
- Tests need feature matrices but must not run GPU feature engineering
- Feature alignment with labels is critical and must be tested
- Model input shapes must be validated without actual model loading

WHAT IT PROTECTS:
- Ensures feature/label alignment in train.py
- Validates that models receive correctly shaped inputs
- Tests that feature columns match expected schemas

HOW IT WORKS:
- Each generator returns a small pandas DataFrame (10-20 rows)
- Feature names follow the actual naming conventions in the codebase
- Data is deterministic and numerically stable (no NaN explosions)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List


def make_synthetic_features(
    n_rows: int = 15,
    n_features: int = 20,
    start_time: Optional[datetime] = None,
    seed: int = 42,
    feature_prefix: str = "feature",
) -> pd.DataFrame:
    """
    Generate synthetic feature matrix with random values.
    
    Args:
        n_rows: Number of samples (default: 15)
        n_features: Number of feature columns (default: 20)
        start_time: Starting timestamp (default: 2024-01-01)
        seed: Random seed (default: 42)
        feature_prefix: Prefix for feature column names (default: "feature")
        
    Returns:
        DataFrame with n_features columns, DatetimeIndex of n_rows
        
    Example:
        >>> X = make_synthetic_features(n_rows=10, n_features=5)
        >>> assert X.shape == (10, 5)
        >>> assert all(col.startswith('feature') for col in X.columns)
    """
    np.random.seed(seed)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    # Generate random features (standardized: mean=0, std=1)
    data = np.random.randn(n_rows, n_features)
    
    columns = [f"{feature_prefix}_{i}" for i in range(n_features)]
    
    return pd.DataFrame(data, index=timestamps, columns=columns)


def make_aligned_features(
    df_ohlcv: pd.DataFrame,
    n_features: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate features aligned to an OHLCV DataFrame's index.
    
    WHY: Tests that features align correctly with price data.
    WHAT: Returns features with identical index to input DataFrame.
    
    Args:
        df_ohlcv: OHLCV DataFrame with DatetimeIndex
        n_features: Number of feature columns (default: 15)
        seed: Random seed (default: 42)
        
    Returns:
        DataFrame with features aligned to df_ohlcv.index
    """
    np.random.seed(seed)
    
    n_rows = len(df_ohlcv)
    data = np.random.randn(n_rows, n_features)
    columns = [f"aligned_feature_{i}" for i in range(n_features)]
    
    return pd.DataFrame(data, index=df_ohlcv.index, columns=columns)


def make_regime_features(
    n_rows: int = 15,
    start_time: Optional[datetime] = None,
    include_linreg: bool = True,
    include_atr: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic features matching regime detector expectations.
    
    WHY: Tests that rule_based_regime() in train.py works correctly.
    WHAT: Produces features with columns used by regime classification logic.
    
    Args:
        n_rows: Number of samples (default: 15)
        start_time: Starting timestamp
        include_linreg: Include linreg_lr_slope, linreg_lr_mid columns
        include_atr: Include ATR-related features
        
    Returns:
        DataFrame with regime-detection compatible features
        
    Columns included (matching train.py expectations):
        - linreg_lr_slope: Linear regression slope
        - linreg_lr_mid: Linear regression midline
        - linreg_lr_width: Channel width
        - close: Price (needed for regime logic)
    """
    np.random.seed(42)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    # Base price around 100
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    
    features = {'close': close}
    
    if include_linreg:
        # Slope: small values around 0, occasionally large for trends
        slope = np.random.uniform(-0.002, 0.002, n_rows)
        # Make first half trend up, second half trend down for variety
        slope[:n_rows//2] = np.abs(slope[:n_rows//2]) + 0.001
        slope[n_rows//2:] = -np.abs(slope[n_rows//2:]) - 0.001
        
        # Mid line tracks close approximately
        lr_mid = close + np.random.randn(n_rows) * 0.5
        
        # Width (channel width)
        lr_width = np.abs(np.random.randn(n_rows)) * 2 + 1
        
        features['linreg_lr_slope'] = slope
        features['linreg_lr_mid'] = lr_mid
        features['linreg_lr_width'] = lr_width
    
    if include_atr:
        # ATR: positive values, roughly 1-3% of price
        atr = close * np.random.uniform(0.01, 0.03, n_rows)
        features['atr_14'] = atr
    
    return pd.DataFrame(features, index=timestamps)


def make_blender_features(
    n_rows: int = 15,
    start_time: Optional[datetime] = None,
    signal_modules: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate synthetic features matching SignalBlender expectations.
    
    WHY: Tests that SignalBlender receives correctly formatted inputs.
    WHAT: Produces features with module signal columns and regime encoding.
    
    Args:
        n_rows: Number of samples (default: 15)
        start_time: Starting timestamp
        signal_modules: Module names to include signals for
                       (default: ['superma', 'trendmagic', 'pvt'])
        
    Returns:
        DataFrame with blender-compatible features including:
        - {module}_signal columns for each module
        - regime column (encoded as int)
        - Various base features
    """
    np.random.seed(42)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    if signal_modules is None:
        signal_modules = ['superma', 'trendmagic', 'pvt']
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    features = {}
    
    # Add module signal columns (values: -1, 0, 1)
    for module in signal_modules:
        features[f"{module}_signal"] = np.random.choice([-1, 0, 1], size=n_rows)
    
    # Add regime column (encoded: 0=trend_up, 1=trend_down, 2=chop, etc.)
    features['regime'] = np.random.choice([0, 1, 2, 3, 4], size=n_rows)
    
    # Add some base features
    for i in range(10):
        features[f"base_feature_{i}"] = np.random.randn(n_rows)
    
    return pd.DataFrame(features, index=timestamps)

