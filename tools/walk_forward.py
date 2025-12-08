"""
Walk-Forward Evaluation Tool

Implements walk-forward optimization and validation to assess
strategy robustness on unseen data.

Walk-Forward Process:
1. Split data into sequential windows
2. For each window: train on historical data, validate on forward data
3. Aggregate out-of-sample performance across all windows
4. Compare to in-sample performance to detect overfitting

Features:
- Anchored walk-forward (expanding training window)
- Rolling walk-forward (fixed training window)
- Customizable train/test ratios
- Gap period to prevent lookahead bias
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable


@dataclass
class WalkForwardFold:
    """
    Results from a single walk-forward fold.
    
    Attributes:
        fold_idx: Index of this fold (0-based)
        train_start: Start timestamp of training period
        train_end: End timestamp of training period
        test_start: Start timestamp of test period
        test_end: End timestamp of test period
        train_sharpe: In-sample Sharpe ratio
        test_sharpe: Out-of-sample Sharpe ratio
        train_return: In-sample total return
        test_return: Out-of-sample total return
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
    """
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    n_train_samples: int
    n_test_samples: int


@dataclass
class WalkForwardResult:
    """
    Aggregated results from walk-forward evaluation.
    
    Attributes:
        n_folds: Number of walk-forward folds
        folds: List of individual fold results
        avg_train_sharpe: Average in-sample Sharpe ratio
        avg_test_sharpe: Average out-of-sample Sharpe ratio
        sharpe_decay_ratio: (avg_train - avg_test) / avg_train
        total_oos_return: Combined out-of-sample return
        oos_sharpe: Sharpe of combined OOS returns
        is_robust: True if OOS performance is acceptable
        failure_modes: List of detected issues
    """
    n_folds: int
    folds: List[WalkForwardFold]
    avg_train_sharpe: float
    avg_test_sharpe: float
    sharpe_decay_ratio: float
    total_oos_return: float
    oos_sharpe: float
    is_robust: bool
    failure_modes: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "ROBUST" if self.is_robust else "FRAGILE"
        lines = [
            f"Walk-Forward Evaluation: {status}",
            f"  Folds: {self.n_folds}",
            f"  Avg Train Sharpe: {self.avg_train_sharpe:.2f}",
            f"  Avg Test Sharpe: {self.avg_test_sharpe:.2f}",
            f"  Sharpe Decay: {self.sharpe_decay_ratio:.1%}",
            f"  Total OOS Return: {self.total_oos_return:.2%}",
            f"  OOS Sharpe: {self.oos_sharpe:.2f}"
        ]
        
        if self.failure_modes:
            lines.append("\nFailure Modes:")
            for mode in self.failure_modes:
                lines.append(f"  ⚠️  {mode}")
        
        return "\n".join(lines)


def _compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    mean_ret = np.nanmean(returns)
    std_ret = np.nanstd(returns, ddof=1)
    
    if std_ret == 0 or np.isnan(std_ret):
        return 0.0
    
    return np.sqrt(periods_per_year) * mean_ret / std_ret


def _compute_total_return(returns: np.ndarray) -> float:
    """Compute total compounded return."""
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) == 0:
        return 0.0
    return np.prod(1 + clean_returns) - 1


def walk_forward_evaluate(
    df: pd.DataFrame,
    signal_generator: Callable[[pd.DataFrame], pd.Series],
    n_folds: int = 5,
    train_ratio: float = 0.8,
    gap_periods: int = 0,
    anchored: bool = False,
    periods_per_year: int = 252,
    min_train_samples: int = 100,
    min_test_samples: int = 20
) -> WalkForwardResult:
    """
    Perform walk-forward evaluation of a trading strategy.
    
    Walk-forward evaluation tests a strategy's ability to generalize
    by repeatedly training on historical data and validating on
    subsequent unseen data.
    
    Args:
        df: OHLCV DataFrame with 'close' column and returns
        signal_generator: Function that takes df and returns signals
        n_folds: Number of walk-forward folds (default: 5)
        train_ratio: Proportion of each window for training (default: 0.8)
        gap_periods: Gap between train and test to prevent lookahead (default: 0)
        anchored: If True, training window expands; if False, rolls (default: False)
        periods_per_year: For Sharpe annualization (default: 252)
        min_train_samples: Minimum training samples per fold
        min_test_samples: Minimum test samples per fold
        
    Returns:
        WalkForwardResult with per-fold and aggregate metrics
        
    Example:
        >>> def generate_signals(df):
        ...     return pd.Series(np.sign(df['close'].pct_change(5)), index=df.index)
        >>> result = walk_forward_evaluate(ohlcv_df, generate_signals, n_folds=5)
        >>> print(result)
        
    Notes:
        - Rolling walk-forward maintains constant training window size
        - Anchored walk-forward grows training window (more stable but slower)
        - Gap period should match label horizon to prevent lookahead
    """
    n_samples = len(df)
    
    if n_samples < n_folds * (min_train_samples + min_test_samples):
        raise ValueError(
            f"Insufficient data: {n_samples} samples for {n_folds} folds "
            f"(need at least {n_folds * (min_train_samples + min_test_samples)})"
        )
    
    # Calculate fold boundaries
    fold_size = n_samples // n_folds
    folds: List[WalkForwardFold] = []
    all_oos_returns: List[np.ndarray] = []
    failure_modes: List[str] = []
    
    for fold_idx in range(n_folds):
        # Determine training and test boundaries
        if anchored:
            # Anchored: train from start up to fold boundary
            train_start_idx = 0
            train_end_idx = fold_size * (fold_idx + 1)
        else:
            # Rolling: fixed window size
            train_start_idx = fold_size * fold_idx
            train_end_idx = train_start_idx + int(fold_size * train_ratio)
        
        test_start_idx = train_end_idx + gap_periods
        test_end_idx = min(
            fold_size * (fold_idx + 2) if fold_idx < n_folds - 1 else n_samples,
            n_samples
        )
        
        # Ensure minimum samples
        if train_end_idx - train_start_idx < min_train_samples:
            continue
        if test_end_idx - test_start_idx < min_test_samples:
            continue
        
        # Split data
        train_df = df.iloc[train_start_idx:train_end_idx].copy()
        test_df = df.iloc[test_start_idx:test_end_idx].copy()
        
        # Generate signals (simulating training on train_df, applying to test_df)
        try:
            # In a real scenario, we'd fit a model on train_df
            # Here we just apply the signal generator
            test_signals = signal_generator(test_df)
            
            # Compute returns
            if 'close' in test_df.columns:
                test_returns = test_df['close'].pct_change().shift(-1).iloc[:-1]
                test_signals = test_signals.iloc[:-1]
                
                # Strategy returns
                strategy_returns = (test_signals * test_returns).dropna().values
            else:
                strategy_returns = np.array([])
            
            # Compute metrics
            train_signals = signal_generator(train_df)
            if 'close' in train_df.columns:
                train_returns = train_df['close'].pct_change().shift(-1).iloc[:-1]
                train_strategy = (train_signals.iloc[:-1] * train_returns).dropna().values
            else:
                train_strategy = np.array([])
            
            train_sharpe = _compute_sharpe(train_strategy, periods_per_year)
            test_sharpe = _compute_sharpe(strategy_returns, periods_per_year)
            train_return = _compute_total_return(train_strategy)
            test_return = _compute_total_return(strategy_returns)
            
            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_start=df.index[train_start_idx],
                train_end=df.index[train_end_idx - 1],
                test_start=df.index[test_start_idx],
                test_end=df.index[test_end_idx - 1],
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                train_return=train_return,
                test_return=test_return,
                n_train_samples=len(train_df),
                n_test_samples=len(test_df)
            )
            folds.append(fold)
            
            if len(strategy_returns) > 0:
                all_oos_returns.append(strategy_returns)
            
        except Exception as e:
            failure_modes.append(f"Fold {fold_idx} failed: {str(e)}")
            continue
    
    if len(folds) == 0:
        return WalkForwardResult(
            n_folds=0,
            folds=[],
            avg_train_sharpe=0.0,
            avg_test_sharpe=0.0,
            sharpe_decay_ratio=1.0,
            total_oos_return=0.0,
            oos_sharpe=0.0,
            is_robust=False,
            failure_modes=["All folds failed"]
        )
    
    # Aggregate metrics
    avg_train_sharpe = np.mean([f.train_sharpe for f in folds])
    avg_test_sharpe = np.mean([f.test_sharpe for f in folds])
    
    if avg_train_sharpe != 0:
        sharpe_decay_ratio = (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe)
    else:
        sharpe_decay_ratio = 0.0
    
    # Combined OOS performance
    if all_oos_returns:
        combined_oos = np.concatenate(all_oos_returns)
        total_oos_return = _compute_total_return(combined_oos)
        oos_sharpe = _compute_sharpe(combined_oos, periods_per_year)
    else:
        total_oos_return = 0.0
        oos_sharpe = 0.0
    
    # Detect failure modes
    if sharpe_decay_ratio > 0.5:
        failure_modes.append(f"High Sharpe decay: {sharpe_decay_ratio:.1%}")
    
    if avg_test_sharpe < 0:
        failure_modes.append(f"Negative OOS Sharpe: {avg_test_sharpe:.2f}")
    
    negative_folds = sum(1 for f in folds if f.test_return < 0)
    if negative_folds > len(folds) / 2:
        failure_modes.append(f"{negative_folds}/{len(folds)} folds have negative OOS returns")
    
    # Determine robustness
    is_robust = (
        len(failure_modes) == 0 and
        avg_test_sharpe > 0 and
        sharpe_decay_ratio < 0.5
    )
    
    return WalkForwardResult(
        n_folds=len(folds),
        folds=folds,
        avg_train_sharpe=avg_train_sharpe,
        avg_test_sharpe=avg_test_sharpe,
        sharpe_decay_ratio=sharpe_decay_ratio,
        total_oos_return=total_oos_return,
        oos_sharpe=oos_sharpe,
        is_robust=is_robust,
        failure_modes=failure_modes
    )


if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    n_periods = 1000
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='H')
    close = 100 * np.exp(np.cumsum(np.random.randn(n_periods) * 0.002))
    
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(n_periods) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n_periods) * 0.005)),
        'low': close * (1 - np.abs(np.random.randn(n_periods) * 0.005)),
        'close': close,
        'volume': np.random.uniform(1000, 2000, n_periods)
    }, index=dates)
    
    # Simple momentum signal generator
    def momentum_signals(data: pd.DataFrame) -> pd.Series:
        returns = data['close'].pct_change(5)
        return pd.Series(np.sign(returns), index=data.index)
    
    print("=" * 60)
    print("Walk-Forward Evaluation")
    print("=" * 60)
    
    result = walk_forward_evaluate(
        df, 
        momentum_signals, 
        n_folds=5,
        gap_periods=5
    )
    
    print(result)
    
    print("\nPer-Fold Details:")
    for fold in result.folds:
        print(f"  Fold {fold.fold_idx}: Train Sharpe={fold.train_sharpe:.2f}, "
              f"Test Sharpe={fold.test_sharpe:.2f}")

