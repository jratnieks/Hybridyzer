"""
OHLCV Data Profiler Tool

Profiles BTC OHLCV data to detect drift, anomalies, and data quality issues
that could corrupt backtest results or model training.

Key Checks:
- Statistical drift detection (rolling vs global statistics)
- Price gap anomalies (extreme moves)
- Volume spike detection
- Missing data patterns
- Timestamp regularity
- Distribution shifts over time
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class DataProfileResult:
    """
    Results from OHLCV data profiling.
    
    Attributes:
        n_rows: Total number of rows
        date_range: Tuple of (start, end) timestamps
        passed: Whether all quality checks passed
        errors: List of critical errors
        warnings: List of non-critical warnings
        drift_metrics: Dictionary of drift statistics
        anomaly_counts: Dictionary of anomaly counts by type
        quality_score: Overall quality score (0-100)
    """
    n_rows: int
    date_range: Tuple[pd.Timestamp, pd.Timestamp]
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    drift_metrics: Dict[str, float] = field(default_factory=dict)
    anomaly_counts: Dict[str, int] = field(default_factory=dict)
    quality_score: float = 100.0
    
    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Data Profile: {status} (Quality Score: {self.quality_score:.1f}/100)",
            f"  Rows: {self.n_rows}",
            f"  Date Range: {self.date_range[0]} to {self.date_range[1]}"
        ]
        
        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  ❌ {err}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  ⚠️  {warn}")
        
        if self.drift_metrics:
            lines.append("\nDrift Metrics:")
            for key, val in self.drift_metrics.items():
                lines.append(f"  {key}: {val:.4f}")
        
        if self.anomaly_counts:
            lines.append("\nAnomaly Counts:")
            for key, val in self.anomaly_counts.items():
                lines.append(f"  {key}: {val}")
        
        return "\n".join(lines)


def _compute_returns(close: pd.Series) -> pd.Series:
    """Compute log returns from close prices."""
    return np.log(close / close.shift(1))


def _detect_price_gaps(
    df: pd.DataFrame,
    threshold_std: float = 5.0
) -> Tuple[pd.Series, int]:
    """
    Detect extreme price gaps (potential anomalies).
    
    Args:
        df: OHLCV DataFrame
        threshold_std: Number of std deviations to flag as anomaly
        
    Returns:
        Tuple of (anomaly_mask, count)
    """
    returns = _compute_returns(df['close'])
    returns_std = returns.std()
    returns_mean = returns.mean()
    
    # Z-score of returns
    z_scores = (returns - returns_mean) / returns_std
    
    # Flag extreme moves
    anomaly_mask = np.abs(z_scores) > threshold_std
    
    return anomaly_mask, anomaly_mask.sum()


def _detect_volume_spikes(
    df: pd.DataFrame,
    threshold_multiplier: float = 5.0
) -> Tuple[pd.Series, int]:
    """
    Detect extreme volume spikes.
    
    Args:
        df: OHLCV DataFrame
        threshold_multiplier: Multiplier of rolling mean to flag as spike
        
    Returns:
        Tuple of (spike_mask, count)
    """
    if 'volume' not in df.columns:
        return pd.Series(False, index=df.index), 0
    
    volume = df['volume']
    rolling_mean = volume.rolling(window=100, min_periods=20).mean()
    
    spike_mask = volume > (rolling_mean * threshold_multiplier)
    
    return spike_mask, spike_mask.sum()


def _detect_ohlc_violations(df: pd.DataFrame) -> Tuple[pd.Series, int]:
    """
    Detect OHLC constraint violations (high < low, close outside range, etc.).
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Tuple of (violation_mask, count)
    """
    violations = pd.Series(False, index=df.index)
    
    # High should be >= Low
    violations |= df['high'] < df['low']
    
    # Close should be between Low and High
    violations |= (df['close'] < df['low']) | (df['close'] > df['high'])
    
    # Open should be between Low and High
    violations |= (df['open'] < df['low']) | (df['open'] > df['high'])
    
    return violations, violations.sum()


def _compute_drift_metrics(
    df: pd.DataFrame,
    window: int = 500
) -> Dict[str, float]:
    """
    Compute statistical drift metrics comparing recent vs historical data.
    
    Args:
        df: OHLCV DataFrame
        window: Window size for rolling statistics
        
    Returns:
        Dictionary of drift metrics
    """
    close = df['close']
    returns = _compute_returns(close)
    
    # Split into first half and second half
    mid_point = len(df) // 2
    first_half = returns.iloc[:mid_point]
    second_half = returns.iloc[mid_point:]
    
    drift_metrics = {}
    
    # Mean drift (shift in average return)
    mean_drift = second_half.mean() - first_half.mean()
    drift_metrics['return_mean_drift'] = mean_drift
    
    # Volatility drift (change in std dev)
    vol_drift = second_half.std() / first_half.std() - 1.0 if first_half.std() > 0 else 0.0
    drift_metrics['volatility_drift'] = vol_drift
    
    # Skewness drift
    skew_first = first_half.skew()
    skew_second = second_half.skew()
    drift_metrics['skewness_drift'] = skew_second - skew_first
    
    # Kurtosis drift
    kurt_first = first_half.kurtosis()
    kurt_second = second_half.kurtosis()
    drift_metrics['kurtosis_drift'] = kurt_second - kurt_first
    
    # Rolling volatility ratio (recent vs historical)
    recent_vol = returns.iloc[-window:].std() if len(returns) > window else returns.std()
    historical_vol = returns.iloc[:-window].std() if len(returns) > window else returns.std()
    if historical_vol > 0:
        drift_metrics['recent_vol_ratio'] = recent_vol / historical_vol
    else:
        drift_metrics['recent_vol_ratio'] = 1.0
    
    return drift_metrics


def _check_timestamp_regularity(
    df: pd.DataFrame
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Check timestamp regularity and detect gaps.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        Tuple of (errors, warnings, gap_counts)
    """
    errors: List[str] = []
    warnings: List[str] = []
    gap_counts: Dict[str, int] = {}
    
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex")
        return errors, warnings, gap_counts
    
    # Check for duplicates
    n_duplicates = df.index.duplicated().sum()
    if n_duplicates > 0:
        errors.append(f"Found {n_duplicates} duplicate timestamps")
        gap_counts['duplicates'] = n_duplicates
    
    # Check monotonicity
    if not df.index.is_monotonic_increasing:
        errors.append("Timestamps are not monotonically increasing")
    
    # Detect gaps
    deltas = df.index.to_series().diff()
    if len(deltas) > 1:
        median_delta = deltas.median()
        
        # Large gaps (> 3x median)
        large_gaps = deltas > median_delta * 3
        n_large_gaps = large_gaps.sum()
        if n_large_gaps > 0:
            warnings.append(f"Found {n_large_gaps} large gaps (>3x median interval)")
            gap_counts['large_gaps'] = n_large_gaps
        
        # Very large gaps (> 10x median, likely data issues)
        very_large_gaps = deltas > median_delta * 10
        n_very_large = very_large_gaps.sum()
        if n_very_large > 0:
            errors.append(f"Found {n_very_large} very large gaps (>10x median interval)")
            gap_counts['very_large_gaps'] = n_very_large
    
    return errors, warnings, gap_counts


def _check_data_quality(
    df: pd.DataFrame
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Check general data quality (NaN, Inf, zeros).
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Tuple of (errors, warnings, counts)
    """
    errors: List[str] = []
    warnings: List[str] = []
    counts: Dict[str, int] = {}
    
    # Check for NaN in required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                pct = nan_count / len(df) * 100
                if pct > 5:
                    errors.append(f"{col}: {nan_count} NaN values ({pct:.1f}%)")
                else:
                    warnings.append(f"{col}: {nan_count} NaN values ({pct:.1f}%)")
                counts[f'{col}_nan'] = nan_count
    
    # Check for Inf values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            errors.append(f"{col}: {inf_count} Inf values")
            counts[f'{col}_inf'] = inf_count
    
    # Check for zero close prices (suspicious)
    if 'close' in df.columns:
        zero_close = (df['close'] == 0).sum()
        if zero_close > 0:
            errors.append(f"Found {zero_close} zero close prices")
            counts['zero_close'] = zero_close
    
    # Check for negative prices (impossible)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                errors.append(f"{col}: {neg_count} negative values (impossible)")
                counts[f'{col}_negative'] = neg_count
    
    return errors, warnings, counts


def profile_ohlcv(
    df: pd.DataFrame,
    price_gap_threshold: float = 5.0,
    volume_spike_threshold: float = 5.0,
    strict: bool = False
) -> DataProfileResult:
    """
    Profile OHLCV data for drift, anomalies, and quality issues.
    
    This function performs comprehensive data quality checks to identify
    issues that could corrupt backtest results or model training.
    
    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume (optional)
            Index should be DatetimeIndex
        price_gap_threshold: Std deviations for price gap anomaly detection
        volume_spike_threshold: Multiplier for volume spike detection
        strict: If True, treat warnings as errors
        
    Returns:
        DataProfileResult with comprehensive quality assessment
        
    Example:
        >>> result = profile_ohlcv(btc_df)
        >>> if not result.passed:
        ...     print(result)
        ...     raise ValueError("Data quality check failed")
        
    Checks Performed:
        - Timestamp regularity (gaps, duplicates, monotonicity)
        - NaN/Inf detection in all columns
        - OHLC constraint validation (high >= low, etc.)
        - Price gap anomaly detection
        - Volume spike detection
        - Statistical drift (mean, volatility, skewness, kurtosis)
        - Zero/negative price detection
    """
    errors: List[str] = []
    warnings: List[str] = []
    anomaly_counts: Dict[str, int] = {}
    
    # Basic validation
    if df.empty:
        return DataProfileResult(
            n_rows=0,
            date_range=(pd.NaT, pd.NaT),
            passed=False,
            errors=["DataFrame is empty"],
            quality_score=0.0
        )
    
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        return DataProfileResult(
            n_rows=len(df),
            date_range=(df.index[0], df.index[-1]) if len(df) > 0 else (pd.NaT, pd.NaT),
            passed=False,
            errors=[f"Missing required columns: {missing_cols}"],
            quality_score=0.0
        )
    
    n_rows = len(df)
    date_range = (df.index[0], df.index[-1])
    
    # 1. Timestamp regularity checks
    ts_errors, ts_warnings, ts_counts = _check_timestamp_regularity(df)
    errors.extend(ts_errors)
    warnings.extend(ts_warnings)
    anomaly_counts.update(ts_counts)
    
    # 2. Data quality checks (NaN, Inf, zeros, negatives)
    dq_errors, dq_warnings, dq_counts = _check_data_quality(df)
    errors.extend(dq_errors)
    warnings.extend(dq_warnings)
    anomaly_counts.update(dq_counts)
    
    # 3. OHLC constraint violations
    violation_mask, violation_count = _detect_ohlc_violations(df)
    if violation_count > 0:
        pct = violation_count / n_rows * 100
        if pct > 1:
            errors.append(f"OHLC violations: {violation_count} rows ({pct:.2f}%)")
        else:
            warnings.append(f"OHLC violations: {violation_count} rows ({pct:.2f}%)")
        anomaly_counts['ohlc_violations'] = violation_count
    
    # 4. Price gap anomalies
    gap_mask, gap_count = _detect_price_gaps(df, threshold_std=price_gap_threshold)
    if gap_count > 0:
        pct = gap_count / n_rows * 100
        warnings.append(f"Price gap anomalies (>{price_gap_threshold}σ): {gap_count} ({pct:.3f}%)")
        anomaly_counts['price_gaps'] = gap_count
    
    # 5. Volume spikes
    spike_mask, spike_count = _detect_volume_spikes(df, threshold_multiplier=volume_spike_threshold)
    if spike_count > 0:
        pct = spike_count / n_rows * 100
        warnings.append(f"Volume spikes (>{volume_spike_threshold}x avg): {spike_count} ({pct:.3f}%)")
        anomaly_counts['volume_spikes'] = spike_count
    
    # 6. Statistical drift metrics
    drift_metrics = _compute_drift_metrics(df)
    
    # Flag significant drift
    if abs(drift_metrics.get('volatility_drift', 0)) > 0.5:
        warnings.append(f"High volatility drift: {drift_metrics['volatility_drift']:.2%}")
    
    if abs(drift_metrics.get('return_mean_drift', 0)) > 0.001:
        warnings.append(f"Mean return drift: {drift_metrics['return_mean_drift']:.6f}")
    
    # Calculate quality score
    quality_score = 100.0
    
    # Deduct for errors (10 points each)
    quality_score -= len(errors) * 10
    
    # Deduct for warnings (2 points each)
    quality_score -= len(warnings) * 2
    
    # Deduct for anomaly density
    total_anomalies = sum(anomaly_counts.values())
    anomaly_rate = total_anomalies / n_rows if n_rows > 0 else 0
    quality_score -= min(anomaly_rate * 1000, 20)  # Max 20 point deduction
    
    quality_score = max(0, quality_score)
    
    # Handle strict mode
    if strict:
        errors.extend(warnings)
        warnings = []
    
    passed = len(errors) == 0
    
    return DataProfileResult(
        n_rows=n_rows,
        date_range=date_range,
        passed=passed,
        errors=errors,
        warnings=warnings,
        drift_metrics=drift_metrics,
        anomaly_counts=anomaly_counts,
        quality_score=quality_score
    )


def profile_from_csv(
    csv_path: str,
    **kwargs
) -> DataProfileResult:
    """
    Profile OHLCV data from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        **kwargs: Additional arguments passed to profile_ohlcv
        
    Returns:
        DataProfileResult
    """
    path = Path(csv_path)
    if not path.exists():
        return DataProfileResult(
            n_rows=0,
            date_range=(pd.NaT, pd.NaT),
            passed=False,
            errors=[f"File not found: {csv_path}"],
            quality_score=0.0
        )
    
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        
        # Standardize column names (lowercase)
        df.columns = df.columns.str.lower()
        
        return profile_ohlcv(df, **kwargs)
        
    except Exception as e:
        return DataProfileResult(
            n_rows=0,
            date_range=(pd.NaT, pd.NaT),
            passed=False,
            errors=[f"Error reading CSV: {str(e)}"],
            quality_score=0.0
        )


def compare_data_periods(
    df: pd.DataFrame,
    split_date: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Compare statistics between two time periods.
    
    Useful for detecting regime changes or data quality issues
    between training and test periods.
    
    Args:
        df: OHLCV DataFrame
        split_date: Date to split on (default: midpoint)
        
    Returns:
        Dictionary with comparison metrics
    """
    if split_date is None:
        mid_idx = len(df) // 2
        split_date = df.index[mid_idx]
    
    period_1 = df[df.index < split_date]
    period_2 = df[df.index >= split_date]
    
    results = {
        'split_date': split_date,
        'period_1_rows': len(period_1),
        'period_2_rows': len(period_2)
    }
    
    if len(period_1) == 0 or len(period_2) == 0:
        results['error'] = "One period is empty"
        return results
    
    # Compare returns statistics
    returns_1 = _compute_returns(period_1['close'])
    returns_2 = _compute_returns(period_2['close'])
    
    results['period_1_mean_return'] = returns_1.mean()
    results['period_2_mean_return'] = returns_2.mean()
    results['period_1_volatility'] = returns_1.std()
    results['period_2_volatility'] = returns_2.std()
    results['mean_return_change'] = results['period_2_mean_return'] - results['period_1_mean_return']
    results['volatility_ratio'] = results['period_2_volatility'] / results['period_1_volatility'] if results['period_1_volatility'] > 0 else np.nan
    
    # Compare price ranges
    results['period_1_price_range'] = (period_1['close'].min(), period_1['close'].max())
    results['period_2_price_range'] = (period_2['close'].min(), period_2['close'].max())
    
    # Kolmogorov-Smirnov test for distribution shift
    try:
        from scipy import stats
        ks_stat, ks_pval = stats.ks_2samp(returns_1.dropna(), returns_2.dropna())
        results['ks_statistic'] = ks_stat
        results['ks_pvalue'] = ks_pval
        results['distribution_shift_detected'] = ks_pval < 0.05
    except ImportError:
        results['ks_statistic'] = None
        results['ks_pvalue'] = None
        results['distribution_shift_detected'] = None
    
    return results


if __name__ == "__main__":
    # Example usage
    data_dir = Path("data")
    
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            print("=" * 60)
            print("OHLCV Data Profiler")
            print("=" * 60)
            
            for csv_file in csv_files[:1]:  # Profile first CSV
                print(f"\nProfiling: {csv_file}")
                result = profile_from_csv(str(csv_file))
                print(result)
                
                # Period comparison
                if result.passed:
                    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                    df.columns = df.columns.str.lower()
                    comparison = compare_data_periods(df)
                    print("\nPeriod Comparison:")
                    for key, val in comparison.items():
                        print(f"  {key}: {val}")
        else:
            print("No CSV files found in data/")
    else:
        print("Data directory not found")


