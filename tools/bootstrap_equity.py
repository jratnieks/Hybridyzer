"""
Bootstrap Equity Analysis Tool

Detects overfitting using bootstrap resampling of trade returns.
Computes confidence intervals for key metrics (Sharpe, total return, max drawdown)
and estimates the probability that observed performance is due to chance.

Key Features:
- Circular block bootstrap for time series (preserves autocorrelation)
- Confidence intervals for Sharpe ratio, total return, max drawdown
- P-value estimation for null hypothesis (true Sharpe = 0)
- Overfitting detection via comparison with shuffled returns
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import warnings


@dataclass
class BootstrapResult:
    """
    Results from bootstrap equity analysis.
    
    Attributes:
        n_bootstraps: Number of bootstrap iterations
        metric_name: Name of the metric analyzed
        observed_value: Observed value from original data
        bootstrap_mean: Mean across bootstrap samples
        bootstrap_std: Standard deviation across bootstrap samples
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        p_value: Probability of observing value under null hypothesis
        is_significant: Whether result is statistically significant
        bootstrap_distribution: Array of bootstrap sample values
    """
    n_bootstraps: int
    metric_name: str
    observed_value: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    bootstrap_distribution: np.ndarray = field(repr=False)
    
    def __str__(self) -> str:
        sig_marker = "✓" if self.is_significant else "✗"
        return (
            f"Bootstrap Analysis: {self.metric_name}\n"
            f"  Observed: {self.observed_value:.4f}\n"
            f"  Bootstrap Mean: {self.bootstrap_mean:.4f} ± {self.bootstrap_std:.4f}\n"
            f"  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  P-value: {self.p_value:.4f} {sig_marker}\n"
            f"  Significant (α=0.05): {self.is_significant}"
        )


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
    return np.prod(1 + returns) - 1


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from returns."""
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return np.min(drawdown)


def _circular_block_bootstrap(
    returns: np.ndarray,
    block_size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate one circular block bootstrap sample.
    
    Preserves temporal structure by sampling contiguous blocks
    with wraparound at boundaries.
    
    Args:
        returns: Original return series
        block_size: Size of each block
        rng: Random number generator
        
    Returns:
        Bootstrap sample of same length as returns
    """
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    
    # Sample random starting positions
    starts = rng.integers(0, n, size=n_blocks)
    
    # Build bootstrap sample from blocks
    sample = []
    for start in starts:
        # Circular indexing
        indices = np.arange(start, start + block_size) % n
        sample.extend(returns[indices])
    
    return np.array(sample[:n])


def bootstrap_equity_curve(
    returns: pd.Series,
    n_bootstraps: int = 1000,
    block_size: Optional[int] = None,
    confidence_level: float = 0.95,
    periods_per_year: int = 252,
    seed: int = 42
) -> Dict[str, BootstrapResult]:
    """
    Perform bootstrap analysis on trading returns to detect overfitting.
    
    Uses circular block bootstrap to preserve temporal dependencies in returns.
    Computes bootstrap distributions and confidence intervals for:
    - Sharpe Ratio
    - Total Return
    - Max Drawdown
    
    Args:
        returns: Series of period returns (e.g., daily or bar returns)
        n_bootstraps: Number of bootstrap iterations (default: 1000)
        block_size: Size of blocks for block bootstrap (default: sqrt(n))
        confidence_level: Confidence level for intervals (default: 0.95)
        periods_per_year: Periods per year for annualization (default: 252)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping metric names to BootstrapResult objects
        
    Example:
        >>> results = bootstrap_equity_curve(trade_returns)
        >>> print(results['sharpe'])
        >>> if not results['sharpe'].is_significant:
        ...     print("Warning: Sharpe ratio not statistically significant")
        
    Notes:
        - Block bootstrap preserves autocorrelation in returns
        - P-value tests null hypothesis that true metric = 0
        - Non-significant results suggest potential overfitting
    """
    # Convert to numpy and clean
    returns_arr = returns.dropna().values.astype(np.float64)
    
    if len(returns_arr) < 20:
        raise ValueError("Need at least 20 returns for bootstrap analysis")
    
    # Default block size: sqrt(n)
    if block_size is None:
        block_size = max(1, int(np.sqrt(len(returns_arr))))
    
    rng = np.random.default_rng(seed)
    
    # Compute observed metrics
    observed_sharpe = _compute_sharpe(returns_arr, periods_per_year)
    observed_return = _compute_total_return(returns_arr)
    observed_drawdown = _compute_max_drawdown(returns_arr)
    
    # Bootstrap distributions
    bootstrap_sharpe = np.zeros(n_bootstraps)
    bootstrap_return = np.zeros(n_bootstraps)
    bootstrap_drawdown = np.zeros(n_bootstraps)
    
    for i in range(n_bootstraps):
        sample = _circular_block_bootstrap(returns_arr, block_size, rng)
        bootstrap_sharpe[i] = _compute_sharpe(sample, periods_per_year)
        bootstrap_return[i] = _compute_total_return(sample)
        bootstrap_drawdown[i] = _compute_max_drawdown(sample)
    
    # Compute confidence intervals and p-values
    alpha = 1 - confidence_level
    
    results = {}
    
    for name, observed, bootstrap in [
        ('sharpe', observed_sharpe, bootstrap_sharpe),
        ('total_return', observed_return, bootstrap_return),
        ('max_drawdown', observed_drawdown, bootstrap_drawdown)
    ]:
        # Remove NaN/Inf from bootstrap samples
        valid_mask = np.isfinite(bootstrap)
        bootstrap_clean = bootstrap[valid_mask]
        
        if len(bootstrap_clean) < n_bootstraps * 0.5:
            warnings.warn(f"More than 50% of {name} bootstrap samples are invalid")
        
        # Confidence interval
        ci_lower = np.percentile(bootstrap_clean, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_clean, (1 - alpha / 2) * 100)
        
        # P-value: proportion of bootstrap samples <= 0 (for Sharpe/return)
        # or >= 0 (for drawdown)
        if name == 'max_drawdown':
            # Drawdown is negative, test if significantly different from 0
            p_value = np.mean(bootstrap_clean >= 0)
        else:
            # For Sharpe/return, test if significantly > 0
            p_value = np.mean(bootstrap_clean <= 0)
        
        results[name] = BootstrapResult(
            n_bootstraps=n_bootstraps,
            metric_name=name,
            observed_value=observed,
            bootstrap_mean=np.mean(bootstrap_clean),
            bootstrap_std=np.std(bootstrap_clean, ddof=1),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            is_significant=(p_value < alpha),
            bootstrap_distribution=bootstrap_clean
        )
    
    return results


def detect_overfitting(
    in_sample_returns: pd.Series,
    out_sample_returns: pd.Series,
    n_bootstraps: int = 500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Detect overfitting by comparing in-sample and out-of-sample performance.
    
    Uses the Probability of Backtest Overfitting (PBO) framework to estimate
    the likelihood that in-sample performance is due to overfitting.
    
    Args:
        in_sample_returns: Returns from training/in-sample period
        out_sample_returns: Returns from validation/out-of-sample period
        n_bootstraps: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        Dictionary with overfitting diagnostics:
        - is_sharpe_ratio: In-sample Sharpe ratio
        - oos_sharpe_ratio: Out-of-sample Sharpe ratio
        - sharpe_decay: (IS - OOS) / IS
        - performance_degradation: True if OOS significantly worse
        - pbo_estimate: Estimated probability of backtest overfitting
        
    Notes:
        - High sharpe_decay (> 0.5) suggests overfitting
        - PBO > 0.5 indicates strategy is likely overfit
    """
    rng = np.random.default_rng(seed)
    
    is_returns = in_sample_returns.dropna().values
    oos_returns = out_sample_returns.dropna().values
    
    # Compute observed Sharpe ratios
    is_sharpe = _compute_sharpe(is_returns)
    oos_sharpe = _compute_sharpe(oos_returns)
    
    # Sharpe decay
    if is_sharpe != 0:
        sharpe_decay = (is_sharpe - oos_sharpe) / abs(is_sharpe)
    else:
        sharpe_decay = np.nan
    
    # Bootstrap OOS performance under null hypothesis
    # (shuffling destroys any true signal)
    combined = np.concatenate([is_returns, oos_returns])
    null_oos_sharpe = np.zeros(n_bootstraps)
    
    for i in range(n_bootstraps):
        shuffled = rng.permutation(combined)
        n_oos = len(oos_returns)
        null_oos_sharpe[i] = _compute_sharpe(shuffled[-n_oos:])
    
    # PBO estimate: probability that OOS Sharpe is worse than random
    pbo_estimate = np.mean(null_oos_sharpe >= oos_sharpe)
    
    return {
        'is_sharpe_ratio': is_sharpe,
        'oos_sharpe_ratio': oos_sharpe,
        'sharpe_decay': sharpe_decay,
        'performance_degradation': sharpe_decay > 0.3,
        'pbo_estimate': pbo_estimate,
        'is_overfit': pbo_estimate > 0.5,
        'null_sharpe_mean': np.mean(null_oos_sharpe),
        'null_sharpe_std': np.std(null_oos_sharpe)
    }


if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    
    # Generate synthetic returns with slight positive drift
    n_periods = 500
    returns = pd.Series(np.random.randn(n_periods) * 0.01 + 0.0003)
    
    print("=" * 60)
    print("Bootstrap Equity Analysis")
    print("=" * 60)
    
    results = bootstrap_equity_curve(returns, n_bootstraps=1000)
    
    for name, result in results.items():
        print(f"\n{result}")

