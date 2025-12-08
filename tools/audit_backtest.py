"""
Backtest Audit Tool

Performs invariant checks on trades_df and equity_curve to ensure
backtest integrity and catch common issues before deployment.

Invariants Checked:
- Signal values in {-1, 0, 1}
- Regime values in valid set
- No NaN/Inf in numeric columns
- Timestamps monotonically increasing
- PnL consistency with returns and signals
- Equity curve non-negative
- Transaction costs properly applied
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class BacktestAuditResult:
    """
    Results from backtest audit.
    
    Attributes:
        passed: Whether all invariants passed
        errors: List of critical errors (invariant violations)
        warnings: List of non-critical warnings
        stats: Summary statistics from the audit
    """
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Backtest Audit: {status}"]
        
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  ❌ {err}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  ⚠️  {warn}")
        
        if self.stats:
            lines.append("\nStats:")
            for key, val in self.stats.items():
                lines.append(f"  {key}: {val}")
        
        return "\n".join(lines)


def audit_backtest(
    trades_df: pd.DataFrame,
    equity_curve: Optional[pd.Series] = None,
    cost_per_trade: float = 0.0006,
    strict: bool = True
) -> BacktestAuditResult:
    """
    Audit a backtest for invariant violations and data integrity issues.
    
    This function performs comprehensive checks on backtest outputs to ensure
    data integrity and catch common issues that could indicate bugs or leakage.
    
    Args:
        trades_df: DataFrame with trade data. Expected columns:
            - signal: Trade direction (-1, 0, 1)
            - future_return: Forward return
            - trade_pnl: Realized PnL per trade
            - regime: Market regime label
        equity_curve: Optional Series of cumulative equity values
        cost_per_trade: Expected transaction cost per trade (default: 6 bps round-trip)
        strict: If True, treat warnings as errors
        
    Returns:
        BacktestAuditResult with pass/fail status, errors, warnings, and stats
        
    Raises:
        ValueError: If trades_df is empty or missing required columns
        
    Example:
        >>> result = audit_backtest(trades_df, equity_curve)
        >>> if not result.passed:
        ...     print(result)
        ...     raise ValueError("Backtest audit failed")
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}
    
    # =========================================================================
    # 1. Basic Structure Checks
    # =========================================================================
    
    if trades_df.empty:
        errors.append("trades_df is empty")
        return BacktestAuditResult(passed=False, errors=errors, warnings=warnings, stats=stats)
    
    required_cols = {'signal', 'future_return', 'trade_pnl'}
    missing_cols = required_cols - set(trades_df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return BacktestAuditResult(passed=False, errors=errors, warnings=warnings, stats=stats)
    
    stats['n_rows'] = len(trades_df)
    stats['n_cols'] = len(trades_df.columns)
    
    # =========================================================================
    # 2. Signal Value Invariant
    # =========================================================================
    
    valid_signals = {-1, 0, 1}
    unique_signals = set(trades_df['signal'].dropna().unique())
    invalid_signals = unique_signals - valid_signals
    
    if invalid_signals:
        errors.append(f"Invalid signal values: {invalid_signals}")
    
    stats['signal_distribution'] = trades_df['signal'].value_counts().to_dict()
    
    # =========================================================================
    # 3. Regime Value Invariant
    # =========================================================================
    
    if 'regime' in trades_df.columns:
        valid_regimes = {'trend_up', 'trend_down', 'chop'}
        unique_regimes = set(trades_df['regime'].dropna().unique())
        invalid_regimes = unique_regimes - valid_regimes
        
        if invalid_regimes:
            errors.append(f"Invalid regime values: {invalid_regimes}")
        
        stats['regime_distribution'] = trades_df['regime'].value_counts().to_dict()
    
    # =========================================================================
    # 4. Numeric Data Quality
    # =========================================================================
    
    numeric_cols = ['future_return', 'trade_pnl']
    for col in numeric_cols:
        if col in trades_df.columns:
            # Check for NaN
            nan_count = trades_df[col].isna().sum()
            if nan_count > 0:
                nan_pct = nan_count / len(trades_df) * 100
                if nan_pct > 10:
                    warnings.append(f"{col}: {nan_count} NaN values ({nan_pct:.1f}%)")
            
            # Check for Inf
            inf_count = np.isinf(trades_df[col].dropna()).sum()
            if inf_count > 0:
                errors.append(f"{col}: {inf_count} Inf values detected")
            
            # Check for extreme values (> 100% return per bar)
            extreme_mask = trades_df[col].abs() > 1.0
            extreme_count = extreme_mask.sum()
            if extreme_count > 0:
                warnings.append(f"{col}: {extreme_count} extreme values (abs > 100%)")
    
    # =========================================================================
    # 5. Timestamp Monotonicity
    # =========================================================================
    
    if isinstance(trades_df.index, pd.DatetimeIndex):
        if not trades_df.index.is_monotonic_increasing:
            errors.append("Index is not monotonically increasing")
        
        if trades_df.index.duplicated().any():
            dup_count = trades_df.index.duplicated().sum()
            errors.append(f"Index has {dup_count} duplicate timestamps")
        
        # Check for gaps
        if len(trades_df) > 1:
            deltas = trades_df.index.to_series().diff()
            median_delta = deltas.median()
            large_gaps = deltas > median_delta * 3
            if large_gaps.sum() > 0:
                warnings.append(f"Index has {large_gaps.sum()} large gaps (>3x median)")
    
    # =========================================================================
    # 6. PnL Consistency Check
    # =========================================================================
    
    trade_mask = trades_df['signal'] != 0
    if trade_mask.any():
        # Expected PnL: signal * future_return - cost
        expected_pnl = (
            trades_df.loc[trade_mask, 'signal'] * 
            trades_df.loc[trade_mask, 'future_return'] - 
            cost_per_trade
        )
        actual_pnl = trades_df.loc[trade_mask, 'trade_pnl']
        
        # Check if they're approximately equal
        pnl_diff = (expected_pnl - actual_pnl).abs()
        large_diff_count = (pnl_diff > 0.001).sum()  # > 0.1% difference
        
        if large_diff_count > 0:
            warnings.append(
                f"PnL inconsistency: {large_diff_count} trades differ from expected by >0.1%"
            )
        
        stats['n_trades'] = trade_mask.sum()
        stats['avg_pnl'] = actual_pnl.mean()
        stats['total_pnl'] = actual_pnl.sum()
    
    # =========================================================================
    # 7. Equity Curve Checks
    # =========================================================================
    
    if equity_curve is not None:
        if equity_curve.isna().any():
            errors.append(f"Equity curve has {equity_curve.isna().sum()} NaN values")
        
        if np.isinf(equity_curve).any():
            errors.append(f"Equity curve has {np.isinf(equity_curve).sum()} Inf values")
        
        if (equity_curve <= 0).any():
            warnings.append("Equity curve has non-positive values")
        
        # Check if equity is computable from PnL
        if 'trade_pnl' in trades_df.columns:
            computed_equity = (1 + trades_df['trade_pnl'].fillna(0)).cumprod()
            
            if len(computed_equity) == len(equity_curve):
                correlation = computed_equity.corr(equity_curve)
                if correlation < 0.99:
                    warnings.append(
                        f"Equity curve doesn't match computed from PnL (corr={correlation:.3f})"
                    )
        
        stats['final_equity'] = equity_curve.iloc[-1]
        stats['max_drawdown'] = ((equity_curve / equity_curve.expanding().max()) - 1).min()
    
    # =========================================================================
    # 8. Direction Confidence Checks (if present)
    # =========================================================================
    
    if 'direction_confidence' in trades_df.columns:
        conf = trades_df['direction_confidence']
        
        if (conf < 0).any() or (conf > 1).any():
            out_of_range = ((conf < 0) | (conf > 1)).sum()
            errors.append(f"direction_confidence has {out_of_range} values outside [0, 1]")
        
        stats['avg_confidence'] = conf.mean()
        stats['min_confidence'] = conf.min()
        stats['max_confidence'] = conf.max()
    
    # =========================================================================
    # 9. EV Checks (if present)
    # =========================================================================
    
    if 'ev' in trades_df.columns:
        ev = trades_df['ev'].dropna()
        
        if len(ev) > 0:
            stats['avg_ev'] = ev.mean()
            stats['median_ev'] = ev.median()
            stats['ev_positive_pct'] = (ev > 0).mean() * 100
    
    # =========================================================================
    # Compile Result
    # =========================================================================
    
    if strict:
        errors.extend(warnings)
        warnings = []
    
    passed = len(errors) == 0
    
    return BacktestAuditResult(
        passed=passed,
        errors=errors,
        warnings=warnings,
        stats=stats
    )


def audit_from_csv(
    trades_csv_path: str,
    equity_csv_path: Optional[str] = None,
    **kwargs
) -> BacktestAuditResult:
    """
    Convenience function to audit backtest from CSV files.
    
    Args:
        trades_csv_path: Path to trades CSV file
        equity_csv_path: Optional path to equity curve CSV
        **kwargs: Additional arguments passed to audit_backtest
        
    Returns:
        BacktestAuditResult
    """
    trades_path = Path(trades_csv_path)
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_csv_path}")
    
    trades_df = pd.read_csv(trades_path, index_col=0, parse_dates=True)
    
    equity_curve = None
    if equity_csv_path:
        equity_path = Path(equity_csv_path)
        if equity_path.exists():
            equity_df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
            if 'equity' in equity_df.columns:
                equity_curve = equity_df['equity']
    
    return audit_backtest(trades_df, equity_curve, **kwargs)


if __name__ == "__main__":
    # Example usage with results directory
    results_dir = Path("results")
    
    if results_dir.exists():
        # Find latest breakdown file
        breakdown_files = sorted(results_dir.glob("backtest_breakdown_*.csv"))
        if breakdown_files:
            latest = breakdown_files[-1]
            print(f"Auditing: {latest}")
            
            trades_df = pd.read_csv(latest, index_col=0, parse_dates=True)
            result = audit_backtest(trades_df, strict=False)
            print(result)
        else:
            print("No backtest breakdown files found")
    else:
        print("Results directory not found")

