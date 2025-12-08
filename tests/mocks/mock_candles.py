"""
Synthetic OHLCV candle data generators for testing.

These generators produce tiny, deterministic DataFrames that simulate
various market conditions without loading real data.

WHY THIS EXISTS:
- Tests need OHLCV data but must not load large CSVs
- Different market conditions (trend, chop, volatile) require different test data
- Deterministic data ensures reproducible test results

WHAT IT PROTECTS:
- Ensures backtest.py logic handles all market condition types
- Validates that regime detection works on edge cases
- Tests that feature generators handle various price patterns

HOW IT WORKS:
- Each generator returns a small pandas DataFrame (10-20 rows)
- Data uses a datetime index starting from a fixed timestamp
- Values are deterministic for reproducibility
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def make_synthetic_ohlcv(
    n_rows: int = 10,
    start_price: float = 100.0,
    volatility: float = 0.02,
    start_time: Optional[datetime] = None,
    freq_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with random walk prices.
    
    Args:
        n_rows: Number of candles to generate (default: 10)
        start_price: Starting close price (default: 100.0)
        volatility: Return standard deviation per bar (default: 0.02 = 2%)
        start_time: Starting timestamp (default: 2024-01-01 00:00)
        freq_minutes: Minutes between bars (default: 5)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex
        
    Example:
        >>> df = make_synthetic_ohlcv(n_rows=5, start_price=100.0)
        >>> assert len(df) == 5
        >>> assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    """
    np.random.seed(seed)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    # Generate timestamps
    timestamps = pd.date_range(
        start=start_time,
        periods=n_rows,
        freq=f"{freq_minutes}min"
    )
    
    # Generate close prices via random walk
    returns = np.random.normal(0, volatility, n_rows)
    close = start_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close
    # Open is previous close (or start_price for first bar)
    open_prices = np.concatenate([[start_price], close[:-1]])
    
    # High is max of open/close + random positive excursion
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, volatility/2, n_rows)))
    
    # Low is min of open/close - random negative excursion
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, volatility/2, n_rows)))
    
    # Ensure high >= low always
    high = np.maximum(high, low + 0.01)
    
    # Volume: random positive values
    volume = np.random.uniform(100, 1000, n_rows)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=timestamps)


def make_trending_candles(
    n_rows: int = 15,
    direction: int = 1,
    start_price: float = 100.0,
    trend_strength: float = 0.01,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate synthetic candles with a clear trend direction.
    
    WHY: Tests that regime detection correctly identifies trends.
    WHAT: Produces consistent up/down trending price action.
    
    Args:
        n_rows: Number of candles (default: 15)
        direction: 1 for uptrend, -1 for downtrend (default: 1)
        start_price: Starting price (default: 100.0)
        trend_strength: Per-bar expected return (default: 0.01 = 1%)
        start_time: Starting timestamp (default: 2024-01-01)
        
    Returns:
        DataFrame with trending OHLCV data
    """
    np.random.seed(42)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    # Strong trend with small noise
    base_return = direction * trend_strength
    returns = base_return + np.random.normal(0, 0.002, n_rows)
    close = start_price * np.cumprod(1 + returns)
    
    open_prices = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_prices, close) * 1.002
    low = np.minimum(open_prices, close) * 0.998
    volume = np.random.uniform(100, 1000, n_rows)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=timestamps)


def make_choppy_candles(
    n_rows: int = 15,
    start_price: float = 100.0,
    chop_range: float = 0.005,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate synthetic candles with choppy/sideways price action.
    
    WHY: Tests that regime detection correctly identifies chop/range markets.
    WHAT: Produces oscillating prices with no clear direction.
    
    Args:
        n_rows: Number of candles (default: 15)
        start_price: Starting price (default: 100.0)
        chop_range: Max percentage deviation from mean (default: 0.005)
        start_time: Starting timestamp (default: 2024-01-01)
        
    Returns:
        DataFrame with choppy OHLCV data
    """
    np.random.seed(42)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    # Mean-reverting noise around start_price
    noise = np.cumsum(np.random.normal(0, 0.001, n_rows))
    # Mean revert by subtracting cumulative mean
    noise = noise - np.cumsum(noise) / np.arange(1, n_rows + 1)
    close = start_price * (1 + noise * chop_range)
    
    open_prices = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_prices, close) * 1.001
    low = np.minimum(open_prices, close) * 0.999
    volume = np.random.uniform(100, 1000, n_rows)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=timestamps)


def make_volatile_candles(
    n_rows: int = 15,
    start_price: float = 100.0,
    high_volatility: float = 0.05,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate synthetic candles with high volatility.
    
    WHY: Tests that volatility-based regime detection works correctly.
    WHAT: Produces large price swings typical of volatile market conditions.
    
    Args:
        n_rows: Number of candles (default: 15)
        start_price: Starting price (default: 100.0)
        high_volatility: Return std per bar (default: 0.05 = 5%)
        start_time: Starting timestamp (default: 2024-01-01)
        
    Returns:
        DataFrame with volatile OHLCV data
    """
    np.random.seed(42)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    # Large random returns
    returns = np.random.normal(0, high_volatility, n_rows)
    close = start_price * np.cumprod(1 + returns)
    
    open_prices = np.concatenate([[start_price], close[:-1]])
    # Wide high-low range for volatile bars
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, high_volatility/2, n_rows)))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, high_volatility/2, n_rows)))
    volume = np.random.uniform(500, 2000, n_rows)  # Higher volume in volatile periods
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=timestamps)

