# core/labeling.py
"""
Functions to generate labels for training regime detector and signal blender.

NEW LABEL SYSTEM (v2):
- Forward return over configurable horizon (default: 12 bars = 1 hour for 5-min data)
- 12-bar rolling mean smoothing to remove micro-noise
- Fixed threshold classification (default: 0.0005)
- No ATR normalization or volatility adjustments
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from core.regime_detector import wilder_atr


def make_direction_labels(
    df: pd.DataFrame,
    horizon_bars: int = 12,
    label_threshold: float = 0.0005,
    smoothing_window: int = 12,
    debug: bool = True
) -> pd.DataFrame:
    """
    Generate clean macro-direction labels for SignalBlender training.
    
    This is the NEW label system that replaces ATR-based thresholding.
    Uses longer horizon and smoothing to capture macro trends, not micro-noise.
    
    Args:
        df: OHLCV DataFrame with 'close' column
        horizon_bars: Forward horizon in bars (default: 12 = 1 hour for 5-min data)
        label_threshold: Threshold for label assignment (default: 0.0005 = 0.05%)
        smoothing_window: Rolling window for smoothing returns (default: 12)
        debug: Whether to print label statistics (default: True)
        
    Returns:
        DataFrame with added columns:
        - future_return: Raw future return over horizon (close.shift(-H) / close - 1)
        - smoothed_return: 12-bar rolling mean of future_return
        - blend_label: Direction label (-1, 0, +1) based on smoothed return vs threshold
    """
    df = df.copy()
    close = df['close']
    
    # 1. Compute forward return over horizon
    # future_return_H = close.shift(-H) / close - 1
    future_return = (close.shift(-horizon_bars) / close.replace(0, np.nan)) - 1.0
    df['future_return'] = future_return
    
    # 2. Smooth the return series to remove micro-noise
    # 12-bar rolling mean
    smoothed_return = future_return.rolling(window=smoothing_window, min_periods=1).mean()
    df['smoothed_return'] = smoothed_return
    
    # 3. Map smoothed return to clean direction labels
    # if smoothed > +threshold:   label = 1
    # elif smoothed < -threshold: label = -1
    # else:                       label = 0
    blend_label = pd.Series(0, index=df.index, dtype=int)
    
    # Only label when we have valid smoothed values
    valid_mask = ~smoothed_return.isna()
    if valid_mask.any():
        blend_label[valid_mask & (smoothed_return > label_threshold)] = 1
        blend_label[valid_mask & (smoothed_return < -label_threshold)] = -1
    
    df['blend_label'] = blend_label
    
    # Store threshold for reference (constant value)
    df['threshold'] = label_threshold
    
    # Debug output
    if debug:
        _print_label_debug_stats(
            horizon_bars=horizon_bars,
            label_threshold=label_threshold,
            smoothing_window=smoothing_window,
            future_return=future_return,
            smoothed_return=smoothed_return,
            blend_label=blend_label
        )
    
    return df


def _print_label_debug_stats(
    horizon_bars: int,
    label_threshold: float,
    smoothing_window: int,
    future_return: pd.Series,
    smoothed_return: pd.Series,
    blend_label: pd.Series
) -> None:
    """
    Print debug statistics for label generation.
    
    Args:
        horizon_bars: Forward horizon used
        label_threshold: Threshold used
        smoothing_window: Smoothing window used
        future_return: Raw future return series
        smoothed_return: Smoothed return series
        blend_label: Generated labels
    """
    print("\n" + "="*60)
    print("[NEW LABELS] Direction Label Generation Statistics")
    print("="*60)
    print(f"[NEW LABELS] Horizon: {horizon_bars} bars")
    print(f"[NEW LABELS] Threshold: {label_threshold}")
    print(f"[NEW LABELS] Smoothing window: {smoothing_window} bars")
    
    # Label distribution
    total = len(blend_label)
    long_count = (blend_label == 1).sum()
    short_count = (blend_label == -1).sum()
    flat_count = (blend_label == 0).sum()
    
    print(f"\nLabel distribution:")
    print(f"  long (+1):  {long_count:7d} ({long_count/total*100:5.2f}%)")
    print(f"  short (-1): {short_count:7d} ({short_count/total*100:5.2f}%)")
    print(f"  flat (0):   {flat_count:7d} ({flat_count/total*100:5.2f}%)")
    
    # Raw future return statistics
    fr_valid = future_return.dropna()
    if len(fr_valid) > 0:
        print(f"\nRaw future return statistics (horizon={horizon_bars}):")
        print(f"  mean:   {fr_valid.mean():.6f}")
        print(f"  median: {fr_valid.median():.6f}")
        print(f"  std:    {fr_valid.std():.6f}")
        print(f"  p25:    {fr_valid.quantile(0.25):.6f}")
        print(f"  p75:    {fr_valid.quantile(0.75):.6f}")
    
    # Smoothed return statistics
    sr_valid = smoothed_return.dropna()
    if len(sr_valid) > 0:
        print(f"\nSmoothed return statistics (window={smoothing_window}):")
        print(f"  mean:   {sr_valid.mean():.6f}")
        print(f"  median: {sr_valid.median():.6f}")
        print(f"  std:    {sr_valid.std():.6f}")
        print(f"  p25:    {sr_valid.quantile(0.25):.6f}")
        print(f"  p75:    {sr_valid.quantile(0.75):.6f}")
    
    print("="*60 + "\n")


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX) for trend strength.
    
    ADX measures trend strength regardless of direction.
    Values > 25 indicate strong trend, < 25 indicates weak trend/chop.
    
    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns
        period: Period for ADX calculation (default: 14)
        
    Returns:
        Series of ADX values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range (TR)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # +DM and -DM
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan))
    
    # Calculate DX (Directional Index)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    
    # Calculate ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx


def make_regime_labels(df: pd.DataFrame) -> pd.Series:
    """
    Clean 3-class regime labeling using:
    - MA trend slope (fast MA - slow MA)
    - ADX for trend strength
    - ATR volatility normalization
    
    NOTE: This function is UNCHANGED from the original implementation.
    Regime detection remains separate from direction labeling.
    
    Args:
        df: OHLCV DataFrame with 'close', 'high', 'low' columns
        
    Returns:
        Series of regime labels: 'trend_up', 'trend_down', 'chop'
    """
    close = df['close']
    
    # 1. Trend indicators
    sma_fast = close.rolling(window=10, min_periods=10).mean()
    sma_slow = close.rolling(window=40, min_periods=40).mean()
    slope = sma_fast - sma_slow
    
    # 2. Trend strength (ADX)
    adx = compute_adx(df, period=14)
    
    # 3. Volatility compression for chop detection (not used in final logic but computed for reference)
    atr = wilder_atr(df, 14)
    atr_pct = atr / close.replace(0, np.nan)
    compression = atr_pct.rolling(window=20, min_periods=20).mean()
    
    # Initialize labels as 'chop'
    labels = pd.Series("chop", index=df.index, dtype=object)
    
    # Vectorized labeling
    # Trend up: ADX > 25 and slope > 0
    trend_up_mask = (adx > 25) & (slope > 0) & (~adx.isna()) & (~slope.isna())
    
    # Trend down: ADX > 25 and slope < 0
    trend_down_mask = (adx > 25) & (slope < 0) & (~adx.isna()) & (~slope.isna())
    
    # Apply labels
    labels[trend_up_mask] = "trend_up"
    labels[trend_down_mask] = "trend_down"
    # Everything else remains "chop"
    
    return labels


