"""
Synthetic trade log generators for testing.

These generators produce tiny, deterministic DataFrames that simulate
trade execution results from backtest.py without running actual backtests.

WHY THIS EXISTS:
- Tests need trade logs but must not run heavy backtests
- Different trade outcome patterns require different test data
- Deterministic data ensures reproducible test results

WHAT IT PROTECTS:
- Ensures compute_metrics() handles all trade patterns
- Validates PnL calculations with known expected values
- Tests hit rate, equity curve, and drawdown calculations

HOW IT WORKS:
- Each generator returns a small pandas DataFrame (5-15 rows)
- Trade outcomes are deterministic and verifiable
- Includes all columns expected by backtest.py logic
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


def make_synthetic_trades(
    n_trades: int = 10,
    avg_return: float = 0.001,
    win_rate: float = 0.55,
    start_time: Optional[datetime] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic trade log with configurable outcomes.
    
    Args:
        n_trades: Number of trades (default: 10)
        avg_return: Average per-trade return (default: 0.001 = 0.1%)
        win_rate: Fraction of winning trades (default: 0.55)
        start_time: Starting timestamp (default: 2024-01-01)
        seed: Random seed (default: 42)
        
    Returns:
        DataFrame with columns matching backtest.py trade output:
        - signal: Trade direction (1=long, -1=short)
        - future_return: Raw return before signal applied
        - trade_pnl: Realized PnL (signal * future_return - costs)
        - trade_pnl_raw: Gross PnL before costs
        - regime: Market regime label
        - direction_confidence: Model confidence
        - ev: Expected value
        
    Example:
        >>> trades = make_synthetic_trades(n_trades=5, win_rate=0.6)
        >>> assert len(trades) == 5
        >>> assert 'trade_pnl' in trades.columns
    """
    np.random.seed(seed)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_trades, freq="5min")
    
    # Generate win/loss pattern based on win_rate
    n_wins = int(n_trades * win_rate)
    outcomes = [1] * n_wins + [-1] * (n_trades - n_wins)
    np.random.shuffle(outcomes)
    
    # Generate returns: positive for wins, negative for losses
    base_returns = np.abs(np.random.normal(avg_return, avg_return * 0.5, n_trades))
    future_returns = base_returns * np.array(outcomes)
    
    # Generate trade directions (mix of long/short)
    signals = np.random.choice([1, -1], size=n_trades)
    
    # Trade PnL = signal * future_return (simplified)
    # For winning trades, signal should match return sign
    # Adjust signals to align with outcomes for determinism
    signals = np.where(future_returns > 0, 1, -1) * outcomes
    
    # Gross PnL
    trade_pnl_raw = signals * future_returns
    
    # Net PnL (subtract small cost)
    cost_per_trade = 0.0006  # 6 bps round-trip
    trade_pnl = trade_pnl_raw - cost_per_trade
    
    # Regime labels
    regimes = np.random.choice(['trend_up', 'trend_down', 'chop'], size=n_trades)
    
    # Confidence and EV
    confidence = np.random.uniform(0.5, 0.9, n_trades)
    ev = trade_pnl_raw * confidence  # Simplified EV
    
    return pd.DataFrame({
        'signal': signals,
        'future_return': future_returns,
        'trade_pnl': trade_pnl,
        'trade_pnl_raw': trade_pnl_raw,
        'regime': regimes,
        'direction_confidence': confidence,
        'ev': ev,
        'final_signal': signals,
    }, index=timestamps)


def make_winning_trades(n_trades: int = 10, start_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Generate all-winning trade log for testing upper bounds.
    
    WHY: Tests compute_metrics() behavior with 100% win rate.
    WHAT: All trades have positive PnL, useful for max return calculations.
    
    Args:
        n_trades: Number of trades (default: 10)
        start_time: Starting timestamp
        
    Returns:
        DataFrame with all winning trades
    """
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_trades, freq="5min")
    
    # All positive returns, long positions
    returns = np.full(n_trades, 0.002)  # 0.2% per trade
    signals = np.ones(n_trades, dtype=int)
    trade_pnl_raw = signals * returns
    trade_pnl = trade_pnl_raw - 0.0006
    
    return pd.DataFrame({
        'signal': signals,
        'future_return': returns,
        'trade_pnl': trade_pnl,
        'trade_pnl_raw': trade_pnl_raw,
        'regime': ['trend_up'] * n_trades,
        'direction_confidence': np.full(n_trades, 0.8),
        'ev': trade_pnl_raw * 0.8,
        'final_signal': signals,
    }, index=timestamps)


def make_losing_trades(n_trades: int = 10, start_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Generate all-losing trade log for testing lower bounds.
    
    WHY: Tests compute_metrics() behavior with 0% win rate.
    WHAT: All trades have negative PnL, useful for max drawdown calculations.
    
    Args:
        n_trades: Number of trades (default: 10)
        start_time: Starting timestamp
        
    Returns:
        DataFrame with all losing trades
    """
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_trades, freq="5min")
    
    # All negative returns (longs in downtrend)
    returns = np.full(n_trades, -0.002)  # -0.2% per trade
    signals = np.ones(n_trades, dtype=int)  # Long positions
    trade_pnl_raw = signals * returns
    trade_pnl = trade_pnl_raw - 0.0006
    
    return pd.DataFrame({
        'signal': signals,
        'future_return': returns,
        'trade_pnl': trade_pnl,
        'trade_pnl_raw': trade_pnl_raw,
        'regime': ['trend_down'] * n_trades,
        'direction_confidence': np.full(n_trades, 0.6),
        'ev': trade_pnl_raw * 0.6,
        'final_signal': signals,
    }, index=timestamps)


def make_mixed_trades(
    n_trades: int = 12,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate mixed long/short trades across different regimes.
    
    WHY: Tests that side-by-side stats (long vs short) are computed correctly.
    WHAT: Mix of long wins, long losses, short wins, short losses.
    
    Args:
        n_trades: Number of trades (should be divisible by 4)
        start_time: Starting timestamp
        
    Returns:
        DataFrame with mixed trade outcomes
    """
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    # Ensure divisible by 4 for clean split
    n_trades = (n_trades // 4) * 4
    quarter = n_trades // 4
    
    timestamps = pd.date_range(start=start_time, periods=n_trades, freq="5min")
    
    # Pattern: long win, long loss, short win, short loss (repeated)
    signals = np.array([1, 1, -1, -1] * quarter)
    # For long wins: return > 0, for long losses: return < 0
    # For short wins: return < 0, for short losses: return > 0
    returns = np.array([0.002, -0.002, -0.002, 0.002] * quarter)
    
    trade_pnl_raw = signals * returns  # [+, -, +, -] pattern
    trade_pnl = trade_pnl_raw - 0.0006
    
    regimes = np.array(['trend_up', 'trend_down', 'trend_down', 'trend_up'] * quarter)
    confidence = np.random.RandomState(42).uniform(0.55, 0.85, n_trades)
    
    return pd.DataFrame({
        'signal': signals,
        'future_return': returns,
        'trade_pnl': trade_pnl,
        'trade_pnl_raw': trade_pnl_raw,
        'regime': regimes,
        'direction_confidence': confidence,
        'ev': trade_pnl_raw * confidence,
        'final_signal': signals,
    }, index=timestamps)

