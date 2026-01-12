# train.py
"""
Training script for regime detector and signal blender models.
Supports walk-forward training with 6-month training windows and 1-month validation.
"""

from __future__ import annotations
import argparse
import atexit
import faulthandler
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def _print_probability_distribution(proba: np.ndarray, label: str) -> None:
    """
    Print probability distribution statistics.
    
    Args:
        proba: Array of probabilities
        label: Label for the distribution
    """
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    counts, _ = np.histogram(proba, bins=bins)
    total = len(proba)
    
    print(f"  {label}:")
    print(f"    Mean: {proba.mean():.4f}, Median: {np.median(proba):.4f}")
    print(f"    Distribution by bin:")
    for i in range(len(bins) - 1):
        pct = counts[i] / total * 100 if total > 0 else 0
        print(f"      [{bins[i]:.1f}, {bins[i+1]:.1f}): {counts[i]:6d} ({pct:5.2f}%)")


def _fmt_optional(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return "n/a"
    except Exception:
        return "n/a"
    return f"{value:.{decimals}f}"


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _setup_logging(log_path: Path) -> Tuple[object, object, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)

    def _cleanup() -> None:
        sys.stdout = stdout
        sys.stderr = stderr
        log_file.close()
    atexit.register(_cleanup)
    return log_file, stdout, stderr


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_regime_labels(
    df: pd.DataFrame,
    features: pd.DataFrame,
    strategy: str
) -> pd.Series:
    if strategy == "rule":
        labels = rule_based_regime(features, df)
    elif strategy == "indicator":
        labels = make_regime_labels(df)
    else:
        raise ValueError(f"Unknown regime label strategy: {strategy}")
    return labels.reindex(features.index)


def compute_return_metrics(
    predictions: pd.Series,
    close: pd.Series,
    horizon_bars: int,
    eval_mode: str = "nonoverlap",
    transaction_cost_bps: float = 0.0
) -> Dict[str, object]:
    """
    Compute return metrics using either non-overlapping horizon trades or per-bar returns.

    Args:
        predictions: Series of signals (-1, 0, 1)
        close: Close price series
        horizon_bars: Forward horizon for non-overlap evaluation
        eval_mode: "nonoverlap" or "per-bar"
        transaction_cost_bps: Cost per side in basis points
    """
    def _drawdown_stats(returns: pd.Series) -> Tuple[float, int]:
        if returns.empty:
            return 0.0, 0
        equity = (1.0 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min())
        max_duration = 0
        current = 0
        for value in drawdown:
            if value == 0:
                if current > max_duration:
                    max_duration = current
                current = 0
            else:
                current += 1
        if current > max_duration:
            max_duration = current
        return max_drawdown, int(max_duration)

    def _trade_metrics(trade_returns: pd.Series) -> Dict[str, Optional[float]]:
        if trade_returns.empty:
            return {
                "win_rate": 0.0,
                "avg_trade_return": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0
            }
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        gross_profit = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(losses.sum()) if not losses.empty else 0.0
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        win_rate = float((trade_returns > 0).mean())
        expectancy = float(win_rate * avg_win + (1.0 - win_rate) * avg_loss)
        profit_factor = None
        if gross_loss < 0.0:
            profit_factor = float(gross_profit / abs(gross_loss))
        elif gross_profit == 0.0:
            profit_factor = 0.0
        return {
            "win_rate": win_rate,
            "avg_trade_return": float(trade_returns.mean()),
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }

    def _trade_returns_from_positions(
        preds: pd.Series,
        prices: pd.Series,
        cost_per_side: float
    ) -> pd.Series:
        trade_returns = []
        position = 0
        entry_price = None
        for pos, price in zip(preds, prices):
            pos = int(pos)
            if not np.isfinite(price):
                continue
            if position == 0:
                if pos != 0 and price != 0:
                    position = pos
                    entry_price = price
                continue
            if pos == position:
                continue
            if entry_price is not None and entry_price != 0 and price != 0:
                raw_return = (price / entry_price - 1.0) * position
                cost = 2.0 * cost_per_side if cost_per_side else 0.0
                trade_returns.append(raw_return - cost)
            position = pos
            if pos != 0 and price != 0:
                entry_price = price
            else:
                entry_price = None
        return pd.Series(trade_returns)
    if close.empty or predictions.empty:
        return {
            "eval_mode": eval_mode,
            "transaction_cost_bps": float(transaction_cost_bps),
            "total_cost": 0.0,
            "samples": 0,
            "trade_count": 0,
            "trade_rate": 0.0,
            "turnover": 0.0,
            "cumulative_return": 0.0,
            "gross_cumulative_return": 0.0,
            "gross_mean_return": 0.0,
            "gross_volatility": 0.0,
            "gross_sharpe_like": 0.0,
            "cost_impact": 0.0,
            "mean_return": 0.0,
            "volatility": 0.0,
            "sharpe_like": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0
        }

    preds = predictions.reindex(close.index).fillna(0).astype(int)
    close = close.reindex(preds.index)
    cost_per_side = max(0.0, float(transaction_cost_bps)) / 10000.0

    if eval_mode == "per-bar":
        bar_returns = close.pct_change().shift(-1)
        valid_mask = ~bar_returns.isna()
        preds = preds[valid_mask]
        bar_returns = bar_returns[valid_mask]
        close = close.reindex(preds.index)

        delta = preds.diff().fillna(preds.iloc[0])
        costs = delta.abs() * cost_per_side
        gross_signal_returns = preds * bar_returns
        signal_returns = gross_signal_returns - costs

        trade_returns = _trade_returns_from_positions(preds, close, cost_per_side)
        trade_count = int(len(trade_returns))
        trade_rate = float(trade_count / len(preds)) if len(preds) else 0.0
        turnover = float(delta.abs().sum() / len(preds)) if len(preds) else 0.0
        trade_stats = _trade_metrics(trade_returns)

        cumulative_return = float((1 + signal_returns).prod() - 1)
        gross_cumulative_return = float((1 + gross_signal_returns).prod() - 1)
        mean_return = float(signal_returns.mean())
        volatility = float(signal_returns.std())
        sharpe_like = float(mean_return / volatility) if volatility != 0 else 0.0
        gross_mean_return = float(gross_signal_returns.mean())
        gross_volatility = float(gross_signal_returns.std())
        gross_sharpe_like = float(gross_mean_return / gross_volatility) if gross_volatility != 0 else 0.0
        max_drawdown, max_duration = _drawdown_stats(signal_returns)

        return {
            "eval_mode": eval_mode,
            "transaction_cost_bps": float(transaction_cost_bps),
            "total_cost": float(costs.sum()),
            "samples": int(len(preds)),
            "trade_count": trade_count,
            "trade_rate": trade_rate,
            "turnover": turnover,
            "cumulative_return": cumulative_return,
            "gross_cumulative_return": gross_cumulative_return,
            "gross_mean_return": gross_mean_return,
            "gross_volatility": gross_volatility,
            "gross_sharpe_like": gross_sharpe_like,
            "cost_impact": float(gross_cumulative_return - cumulative_return),
            "mean_return": mean_return,
            "volatility": volatility,
            "sharpe_like": sharpe_like,
            "win_rate": trade_stats["win_rate"],
            "avg_trade_return": trade_stats["avg_trade_return"],
            "profit_factor": trade_stats["profit_factor"],
            "expectancy": trade_stats["expectancy"],
            "avg_win": trade_stats["avg_win"],
            "avg_loss": trade_stats["avg_loss"],
            "gross_profit": trade_stats["gross_profit"],
            "gross_loss": trade_stats["gross_loss"],
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_duration
        }

    if eval_mode != "nonoverlap":
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    close_vals = close.values
    n = len(preds)
    trade_returns = []
    gross_trade_returns = []
    i = 0
    while i + horizon_bars < n:
        signal = int(preds.iloc[i])
        if signal == 0:
            i += 1
            continue

        entry = close_vals[i]
        exit_price = close_vals[i + horizon_bars]
        if not np.isfinite(entry) or not np.isfinite(exit_price) or entry == 0:
            i += 1
            continue

        raw_return = (exit_price / entry - 1.0) * signal
        cost = 2.0 * cost_per_side if cost_per_side else 0.0
        gross_trade_returns.append(raw_return)
        trade_returns.append(raw_return - cost)
        i += horizon_bars

    if not trade_returns:
        return {
            "eval_mode": eval_mode,
            "transaction_cost_bps": float(transaction_cost_bps),
            "total_cost": 0.0,
            "samples": 0,
            "trade_count": 0,
            "trade_rate": 0.0,
            "turnover": 0.0,
            "cumulative_return": 0.0,
            "gross_cumulative_return": 0.0,
            "gross_mean_return": 0.0,
            "gross_volatility": 0.0,
            "gross_sharpe_like": 0.0,
            "cost_impact": 0.0,
            "mean_return": 0.0,
            "volatility": 0.0,
            "sharpe_like": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0
        }

    trade_returns = pd.Series(trade_returns)
    gross_trade_returns = pd.Series(gross_trade_returns)
    trade_stats = _trade_metrics(trade_returns)
    trade_count = int(len(trade_returns))
    trade_rate = float(trade_count / len(preds)) if len(preds) else 0.0
    turnover = float(2.0 * trade_count / len(preds)) if len(preds) else 0.0
    cumulative_return = float((1 + trade_returns).prod() - 1)
    gross_cumulative_return = float((1 + gross_trade_returns).prod() - 1)
    mean_return = float(trade_returns.mean())
    volatility = float(trade_returns.std())
    sharpe_like = float(mean_return / volatility) if volatility != 0 else 0.0
    gross_mean_return = float(gross_trade_returns.mean())
    gross_volatility = float(gross_trade_returns.std())
    gross_sharpe_like = float(gross_mean_return / gross_volatility) if gross_volatility != 0 else 0.0
    total_cost = float(2.0 * cost_per_side * trade_count)
    max_drawdown, max_duration = _drawdown_stats(trade_returns)

    return {
        "eval_mode": eval_mode,
        "transaction_cost_bps": float(transaction_cost_bps),
        "total_cost": total_cost,
        "samples": int(len(trade_returns)),
        "trade_count": trade_count,
        "trade_rate": trade_rate,
        "turnover": turnover,
        "cumulative_return": cumulative_return,
        "gross_cumulative_return": gross_cumulative_return,
        "gross_mean_return": gross_mean_return,
        "gross_volatility": gross_volatility,
        "gross_sharpe_like": gross_sharpe_like,
        "cost_impact": float(gross_cumulative_return - cumulative_return),
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe_like": sharpe_like,
        "win_rate": trade_stats["win_rate"],
        "avg_trade_return": trade_stats["avg_trade_return"],
        "profit_factor": trade_stats["profit_factor"],
        "expectancy": trade_stats["expectancy"],
        "avg_win": trade_stats["avg_win"],
        "avg_loss": trade_stats["avg_loss"],
        "gross_profit": trade_stats["gross_profit"],
        "gross_loss": trade_stats["gross_loss"],
        "max_drawdown": max_drawdown,
        "max_drawdown_duration": max_duration
    }

from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext
from core.feature_store import FeatureStore, get_gpu_memory_info
from core.regime_detector import RegimeDetector, wilder_atr
from core.signal_blender import SignalBlender, DirectionBlender
from core.labeling import make_direction_labels, make_regime_labels
from core.scalers import ZScoreScaler, MinMaxScaler, RobustScaler
from core.training_utils import (
    prune_features_by_importance,
    save_feature_importances,
    save_selected_features,
    plot_probability_histograms,
    create_training_diagnostics_report
)
from data.btc_data_loader import load_btc_csv


def rule_based_regime(features: pd.DataFrame, price_df: pd.DataFrame) -> pd.Series:
    """
    Generate regime labels using rule-based logic.
    This will be used as training labels for the ML model.
    
    Args:
        features: Feature dataframe
        price_df: Raw OHLCV dataframe
        
    Returns:
        Series of regime labels (trend_up, trend_down, chop)
    """
    regimes = pd.Series("chop", index=features.index, dtype=object)
    
    # Extract linreg features
    lr_slope_col = "linreg_lr_slope"
    lr_mid_col = "linreg_lr_mid"
    lr_width_col = "linreg_lr_width"
    
    if lr_slope_col not in features.columns or lr_mid_col not in features.columns:
        return regimes
    
    lr_slope = features[lr_slope_col]
    lr_mid = features[lr_mid_col]
    price = price_df["close"].reindex(features.index)
    
    # Rule-based labeling (3 regimes only: trend_up, trend_down, chop)
    threshold = 0.001
    
    # Trend regimes: slope direction + price position relative to midline
    trend_up_mask = (lr_slope > threshold) & (price > lr_mid)
    trend_dn_mask = (lr_slope < -threshold) & (price < lr_mid)
    
    regimes[trend_up_mask] = "trend_up"
    regimes[trend_dn_mask] = "trend_down"
    
    # Everything else remains "chop" (including high/low volatility periods)
    # Optionally refine chop detection using channel width
    if lr_width_col in features.columns:
        lr_width = features[lr_width_col]
        width_pct = (lr_width / price.replace(0, np.nan))
        # Very narrow channel with flat slope = definite chop
        narrow_mask = (width_pct < 0.02) & (lr_slope.abs() < threshold)
        narrow_mask = narrow_mask & (~trend_up_mask) & (~trend_dn_mask)
        regimes[narrow_mask] = "chop"
    
    return regimes


def future_return(close: pd.Series, horizon: int = 10) -> pd.Series:
    """
    Compute future return over specified horizon.
    
    Args:
        close: Close price series
        horizon: Number of bars forward to compute return
        
    Returns:
        Series of future returns
    """
    return (close.shift(-horizon) / close - 1.0)


def generate_splits(
    df: pd.DataFrame,
    train_months: int = 6,
    val_months: int = 1,
    slide_months: int = 1
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate walk-forward training/validation splits.
    
    Args:
        df: DataFrame with datetime index
        train_months: Training window size in months
        val_months: Validation window size in months
        slide_months: Slide forward by this many months
        
    Returns:
        List of tuples: (train_start, train_end, val_start, val_end)
    """
    splits = []
    start_date = df.index[0]
    end_date = df.index[-1]
    
    current_start = start_date
    
    while True:
        # Training window
        train_start = current_start
        train_end = train_start + pd.DateOffset(months=train_months)
        
        # Validation window (immediately after training)
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=val_months)
        
        # Check if we have enough data
        if val_end > end_date:
            break
        
        # Ensure dates are within dataframe range
        if train_start >= start_date and train_end <= end_date and val_start >= start_date and val_end <= end_date:
            splits.append((train_start, train_end, val_start, val_end))
        
        # Slide forward
        current_start = current_start + pd.DateOffset(months=slide_months)
        
        # Safety check to avoid infinite loop
        if current_start >= end_date:
            break
    
    return splits


def _infer_bar_delta(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if len(index) < 2:
        return None
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return None
    return diffs.median()


def _apply_purge_and_embargo(
    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    df: pd.DataFrame,
    purge_bars: int,
    embargo_days: int,
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]], pd.Timedelta, pd.Timedelta]:
    if purge_bars <= 0 and embargo_days <= 0:
        return splits, pd.Timedelta(0), pd.Timedelta(0)

    bar_delta = _infer_bar_delta(df.index)
    purge_delta = pd.Timedelta(0)
    if purge_bars > 0:
        if bar_delta is None or pd.isna(bar_delta):
            raise ValueError("Cannot infer bar duration for purge-bars; disable purge or use embargo-days.")
        purge_delta = bar_delta * purge_bars

    embargo_delta = pd.Timedelta(days=embargo_days) if embargo_days > 0 else pd.Timedelta(0)
    adjusted = []
    start_date = df.index[0]
    end_date = df.index[-1]

    for train_start, train_end, val_start, val_end in splits:
        adj_train_end = train_end - purge_delta
        adj_val_start = val_start + embargo_delta
        adj_val_end = val_end + embargo_delta

        if adj_train_end <= train_start:
            continue
        if adj_val_start >= adj_val_end:
            continue
        if adj_train_end >= adj_val_start:
            continue
        if adj_val_end > end_date:
            continue
        if adj_train_end <= start_date or adj_val_start <= start_date:
            continue

        adjusted.append((train_start, adj_train_end, adj_val_start, adj_val_end))

    return adjusted, purge_delta, embargo_delta


def _check_gpu_memory_safe_for_preload(features: pd.DataFrame, use_gpu: bool, max_usage_ratio: float = 0.5) -> bool:
    """
    Check if GPU has enough memory to safely pre-load the full feature DataFrame.
    
    Args:
        features: Feature DataFrame to estimate memory for
        use_gpu: Whether GPU is requested
        max_usage_ratio: Maximum fraction of GPU memory to use (default: 0.5 = 50%)
        
    Returns:
        True if safe to pre-load, False otherwise
    """
    if not use_gpu:
        return False
    
    try:
        # Estimate memory needed (rough estimate: 8 bytes per float64 value)
        num_rows = len(features)
        num_cols = len(features.columns)
        estimated_bytes = num_rows * num_cols * 8
        estimated_mb = estimated_bytes / (1024 ** 2)
        
        # Get GPU memory info
        gpu_info = get_gpu_memory_info()
        if gpu_info is None:
            print(f"[GPU] Cannot get GPU memory info, skipping pre-load (estimated: {estimated_mb:.0f}MB)")
            return False
        
        # Use device_free_gb as available memory
        available_gb = gpu_info.get('device_free_gb', 0)
        if available_gb <= 0:
            print(f"[GPU] No free GPU memory detected, skipping pre-load (estimated: {estimated_mb:.0f}MB)")
            return False
        
        available_mb = available_gb * 1024
        max_allowed_mb = available_mb * max_usage_ratio
        
        if estimated_mb > max_allowed_mb:
            print(f"[GPU] Feature frame too large for pre-load: {estimated_mb:.0f}MB > {max_allowed_mb:.0f}MB (using {max_usage_ratio*100:.0f}% of {available_gb:.1f}GB free)")
            return False
        
        print(f"[GPU] Memory check passed: {estimated_mb:.0f}MB < {max_allowed_mb:.0f}MB ({max_usage_ratio*100:.0f}% of {available_gb:.1f}GB free)")
        return True
    except Exception as e:
        print(f"[GPU] Error checking GPU memory: {e}, skipping pre-load")
        return False


def train_one_window(
    df_train: pd.DataFrame,
    signal_modules: List,
    context_modules: List,
    feats_train: pd.DataFrame,
    horizon_bars: int = 12,
    label_threshold: float = 0.0005,
    regime_label_strategy: str = "indicator",
    smoothing_window: int = 12,
    use_gpu: bool = False,
    random_state: Optional[int] = None,
    calibration_method: Optional[str] = None,
    sharpening_alpha: float = 2.0,
    disable_calibration: bool = False,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    calibration_source: str = "train"
) -> Tuple[RegimeDetector, SignalBlender, Optional[DirectionBlender], Dict, Dict]:
    """
    Train regime detector, signal blender, direction blender, and regime-specific models on a training window.
    
    Args:
        df_train: Training data window
        signal_modules: List of signal modules
        context_modules: List of context modules
        feats_train: Training features
        horizon_bars: Forward horizon in bars for direction labels
        label_threshold: Threshold for direction labels
        regime_label_strategy: Regime label strategy ("indicator" or "rule")
        smoothing_window: Smoothing window for direction labels
        use_gpu: Whether to use GPU acceleration for models
        random_state: Random seed for model initialization
        calibration_method: Calibration method for SignalBlender
        sharpening_alpha: Sharpening alpha for calibrated probabilities
        disable_calibration: Disable calibration for SignalBlender
        X_val: Optional validation features for calibration (used when calibration_source='val')
        y_val: Optional validation labels for calibration (used when calibration_source='val')
        calibration_source: Source of calibration data ('train' or 'val', default: 'train')
        
    Returns:
        Tuple of (regime_detector, signal_blender, direction_blender, training_stats, regime_models_dict)
    """
    # Enforce index integrity between raw data and cached features to avoid
    # silent misalignment after cleaning/warmups.
    assert feats_train.index.equals(df_train.index), "Feature/price index mismatch in training window"

    # Generate labels
    regime_y = build_regime_labels(df_train, feats_train, regime_label_strategy)
    # Use new direction labels
    df_labeled = make_direction_labels(
        df_train,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
        smoothing_window=smoothing_window,
        debug=False  # Suppress debug output in walk-forward windows
    )
    blend_y = df_labeled['blend_label'].reindex(feats_train.index).fillna(0).astype(int)
    
    # Remove invalid rows
    valid_mask = ~(regime_y.isna() | blend_y.isna())
    valid_mask = valid_mask & (regime_y.isin(['trend_up', 'trend_down', 'chop']))
    
    X_regime = feats_train[valid_mask]
    regime_y = regime_y[valid_mask]
    X_blend_base = feats_train[valid_mask]
    blend_y = blend_y[valid_mask]
    
    # Compute module signals for blender
    module_signals = {}
    for module in signal_modules:
        module_features = X_blend_base.filter(regex=f'^{module.name}_', axis=1).copy()
        if not module_features.empty:
            module_features.columns = [col.replace(f'{module.name}_', '') for col in module_features.columns]
        module_signals[module.name] = module.compute_signal(module_features)
    
    X_blend = X_blend_base.copy()
    for module_name, signal in module_signals.items():
        X_blend[f"{module_name}_signal"] = signal[valid_mask]
    X_blend["regime"] = regime_y.map({
        'trend_up': 0, 'trend_down': 1, 'chop': 2
    }).fillna(2)  # Default to chop if unknown
    
    # Train models
    reg = RegimeDetector(use_gpu=use_gpu, random_state=random_state)
    reg.fit(X_regime, regime_y)
    
    # Determine calibration data based on calibration_source
    X_cal = None
    y_cal = None
    if calibration_source == "val" and X_val is not None and y_val is not None:
        # Use validation window for calibration
        X_cal = X_val
        y_cal = y_val
    # Otherwise, calibration will use train data (default behavior in SignalBlender.fit)
    
    blender = SignalBlender(use_gpu=use_gpu, random_state=random_state)
    blender.fit(
        X_blend,
        blend_y,
        calibration_method=calibration_method,
        sharpening_alpha=sharpening_alpha,
        X_val=X_cal,
        y_val=y_cal,
        disable_calibration=disable_calibration
    )
    
    # Train DirectionBlender on trade samples only
    mask_trade = blend_y != 0
    X_blend_trade = X_blend[mask_trade]
    blend_y_trade = blend_y[mask_trade]
    
    direction_blender = None
    if len(X_blend_trade) >= 1000:  # Minimum samples threshold
        y_direction = pd.Series(
            np.where(blend_y_trade > 0, 1, -1),
            index=blend_y_trade.index,
            dtype=int,
            name='direction'
        )
        
        # Determine calibration data for DirectionBlender
        X_cal_dir = None
        y_cal_dir = None
        if calibration_source == "val" and X_val is not None and y_val is not None:
            # Filter validation to trade samples
            y_val_trade = y_val[y_val != 0]
            if len(y_val_trade) >= 200:  # Minimum validation samples
                X_cal_dir = X_val.loc[y_val_trade.index]
                y_cal_dir = pd.Series(
                    np.where(y_val_trade > 0, 1, -1),
                    index=y_val_trade.index,
                    dtype=int,
                    name='direction'
                )
        
        direction_blender = DirectionBlender(use_gpu=use_gpu, random_state=random_state)
        direction_blender.fit(
            X_blend_trade,
            y_direction,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            X_val=X_cal_dir,
            y_val=y_cal_dir,
            disable_calibration=disable_calibration
        )
    
    # Train regime-specific models
    regime_models = {}
    regime_classes = sorted(pd.unique(regime_y))
    
    for regime_name in regime_classes:
        mask_regime = (regime_y == regime_name)
        X_blend_regime = X_blend[mask_regime]
        blend_y_regime = blend_y[mask_regime]
        
        if len(X_blend_regime) < 5000:  # Skip if insufficient samples
            continue
        
        # Determine calibration data for regime-specific model
        X_cal_regime = None
        y_cal_regime = None
        if calibration_source == "val" and X_val is not None and y_val is not None:
            # Filter validation to this regime
            regime_pred_val = reg.predict(X_val.loc[y_val.index] if hasattr(X_val, 'loc') else X_val)
            mask_val_regime = (regime_pred_val == regime_name)
            if mask_val_regime.sum() >= 1000:  # Minimum validation samples
                X_cal_regime = X_val[mask_val_regime]
                y_cal_regime = y_val[mask_val_regime]
        
        model_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'n_bins': 128,
            'split_criterion': 0,
            'bootstrap': True,
            'max_samples': 0.8,
            'max_features': 0.9,
            'n_streams': 4
        }
        model_r = SignalBlender(model_params=model_params, use_gpu=use_gpu, random_state=random_state)
        model_r.fit(
            X_blend_regime,
            blend_y_regime,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            X_val=X_cal_regime,
            y_val=y_cal_regime,
            disable_calibration=disable_calibration
        )
        regime_models[regime_name] = model_r
    
    # Training statistics
    stats = {
        'train_samples': len(X_regime),
        'regime_dist': regime_y.value_counts().to_dict(),
        'blend_dist': blend_y.value_counts().to_dict(),
        'feature_count': len(feats_train.columns)
    }
    
    return reg, blender, direction_blender, stats, regime_models


def validate_one_window(
    df_val: pd.DataFrame,
    reg_model: RegimeDetector,
    blender_model: SignalBlender,
    signal_modules: List,
    context_modules: List,
    feats_val: pd.DataFrame,
    horizon_bars: int = 12,
    label_threshold: float = 0.0005,
    regime_label_strategy: str = "indicator",
    smoothing_window: int = 12,
    eval_mode: str = "nonoverlap",
    transaction_cost_bps: float = 0.0
) -> Dict:
    """
    Validate models on a validation window.
    
    Args:
        df_val: Validation data window
        reg_model: Trained regime detector
        blender_model: Trained signal blender
        signal_modules: List of signal modules
        context_modules: List of context modules
        horizon_bars: Forward horizon in bars for direction labels
        label_threshold: Threshold for direction labels
        regime_label_strategy: Regime label strategy ("indicator" or "rule")
        smoothing_window: Smoothing window for direction labels
        eval_mode: Return evaluation mode ("nonoverlap" or "per-bar")
        transaction_cost_bps: Cost per side in basis points
        
    Returns:
        Dictionary with validation metrics
    """
    # Enforce index integrity between raw data and cached features to avoid
    # silent misalignment after cleaning/warmups.
    assert feats_val.index.equals(df_val.index), "Feature/price index mismatch in validation window"

    # Generate true labels (use new direction labels for consistency)
    regime_y_true = build_regime_labels(df_val, feats_val, regime_label_strategy)
    df_labeled = make_direction_labels(
        df_val,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
        smoothing_window=smoothing_window,
        debug=False  # Suppress debug output in walk-forward windows
    )
    blend_y_true = df_labeled['blend_label'].reindex(feats_val.index).fillna(0).astype(int)
    
    # Remove invalid rows
    valid_mask = ~(regime_y_true.isna() | blend_y_true.isna())
    valid_mask = valid_mask & (regime_y_true.isin(['trend_up', 'trend_down', 'chop']))
    
    X_regime = feats_val[valid_mask]
    regime_y_true = regime_y_true[valid_mask]
    X_blend_base = feats_val[valid_mask]
    blend_y_true = blend_y_true[valid_mask]
    
    # Predict regime
    regime_y_pred = reg_model.predict(X_regime)
    
    # Compute module signals for blender
    module_signals = {}
    for module in signal_modules:
        # Extract module features (vectorized)
        module_features = X_blend_base.filter(regex=f'^{module.name}_', axis=1).copy()
        if not module_features.empty:
            # Remove prefix (vectorized)
            new_cols = pd.Index([col.replace(f'{module.name}_', '') for col in module_features.columns])
            module_features.columns = new_cols
        # Compute signal (returns Series)
        module_signals[module.name] = module.compute_signal(module_features)
    
    # Build X_blend using vectorized operations
    X_blend = X_blend_base.copy()
    for module_name, signal in module_signals.items():
        # Align signal to X_blend index
        aligned_signal = signal.reindex(X_blend.index).fillna(0)
        X_blend[f"{module_name}_signal"] = aligned_signal
    # Encode regime (vectorized)
    regime_map = pd.Series({
        'trend_up': 0, 'trend_down': 1, 'chop': 2
    })
    X_blend["regime"] = regime_y_pred.map(regime_map).fillna(2)  # Default to chop if unknown
    
    # Predict blend
    blend_y_pred = blender_model.predict(X_blend)
    
    # Calculate metrics
    regime_accuracy = (regime_y_pred == regime_y_true).mean()
    blend_accuracy = (blend_y_pred == blend_y_true).mean()
    
    return_metrics = compute_return_metrics(
        blend_y_pred,
        df_val['close'].reindex(blend_y_pred.index),
        horizon_bars=horizon_bars,
        eval_mode=eval_mode,
        transaction_cost_bps=transaction_cost_bps
    )

    metrics = {
        'val_samples': len(X_regime),
        'regime_accuracy': float(regime_accuracy),
        'blend_accuracy': float(blend_accuracy),
        'cumulative_return': float(return_metrics['cumulative_return']),
        'gross_cumulative_return': float(return_metrics['gross_cumulative_return']),
        'max_drawdown': float(return_metrics['max_drawdown']),
        'max_drawdown_duration': int(return_metrics['max_drawdown_duration']),
        'return_metrics': return_metrics,
        'regime_dist_true': regime_y_true.value_counts().to_dict(),
        'regime_dist_pred': regime_y_pred.value_counts().to_dict(),
        'blend_dist_true': blend_y_true.value_counts().to_dict(),
        'blend_dist_pred': blend_y_pred.value_counts().to_dict()
    }
    
    return metrics


def build_features_once(
    df: pd.DataFrame,
    signal_modules: List,
    context_modules: List,
    cache_path: Path,
    force_recompute: bool = False,
    use_gpu: bool = False,
    # Ablation flags
    disable_ml_features: bool = False,
    disable_regime_context: bool = False,
    disable_signal_dynamics: bool = False,
    disable_rolling_stats: bool = False,
    disable_modules: List[str] = None,
    include_modules: Optional[List[str]] = None,
    include_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
) -> Tuple[FeatureStore, pd.DataFrame]:
    """
    Compute all features exactly once, cache to Parquet, and reuse for slicing.

    This centralizes the expensive feature computation step and ensures the
    walk-forward loop only slices cached features instead of rebuilding per
    window.
    
    Ablation flags allow disabling specific feature groups for A/B testing.
    Feature filters allow include/exclude regex after feature generation.
    """
    feature_store = FeatureStore(
        use_gpu=use_gpu,
        disable_ml_features=disable_ml_features,
        disable_regime_context=disable_regime_context,
        disable_signal_dynamics=disable_signal_dynamics,
        disable_rolling_stats=disable_rolling_stats,
        disable_modules=disable_modules,
        include_modules=include_modules,
        include_features=include_features,
        exclude_features=exclude_features,
    )
    cached_features = feature_store.build_and_cache(
        df,
        signal_modules,
        context_modules,
        cache_path,
        force_recompute=force_recompute,
    )
    return feature_store, cached_features


def slice_window(
    df: pd.DataFrame,
    features: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice cached features and raw data for a time window, enforcing alignment.
    Supports both pandas and cuDF DataFrames for features.

    Labels always derive from the raw df slice; models see the cached features
    on the exact same index. Any divergence is a correctness bug and must fail
    fast.
    """
    # Detect if features is cuDF
    is_cudf = hasattr(features, '__class__') and 'cudf' in str(type(features))
    
    if is_cudf:
        # Slice cuDF DataFrame
        feats_window = features[(features.index >= start_ts) & (features.index < end_ts)]
        # Convert to pandas for compatibility with rest of pipeline
        feats_window = feats_window.to_pandas()
    else:
        # Slice pandas DataFrame (existing logic)
        feats_window = features[(features.index >= start_ts) & (features.index < end_ts)].copy()
    
    df_window = df.loc[feats_window.index]

    if not feats_window.index.equals(df_window.index):
        raise ValueError("Cached feature index diverged from raw data during slicing")

    return df_window, feats_window


def prepare_windows(
    df: pd.DataFrame,
    train_months: int,
    val_months: int,
    slide_months: int,
    embargo_days: int = 0,
    purge_bars: int = 0,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Lightweight wrapper to document walk-forward split generation."""
    splits = generate_splits(df, train_months, val_months, slide_months)
    adjusted, purge_delta, embargo_delta = _apply_purge_and_embargo(
        splits,
        df,
        purge_bars=purge_bars,
        embargo_days=embargo_days,
    )
    if purge_bars > 0 or embargo_days > 0:
        print(f"[WalkForward] Purge bars: {purge_bars} ({purge_delta})")
        print(f"[WalkForward] Embargo days: {embargo_days} ({embargo_delta})")
        dropped = len(splits) - len(adjusted)
        if dropped > 0:
            print(f"[WalkForward] Dropped {dropped} splits due to purge/embargo constraints")
    return adjusted


def train_models_for_window(
    df_window: pd.DataFrame,
    feats_window: pd.DataFrame,
    signal_modules: List,
    context_modules: List,
    horizon_bars: int,
    label_threshold: float,
    regime_label_strategy: str,
    smoothing_window: int,
    use_gpu: bool,
    random_state: Optional[int],
    calibration_method: Optional[str],
    sharpening_alpha: float,
    disable_calibration: bool,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    calibration_source: str = "train",
) -> Tuple[RegimeDetector, SignalBlender, Optional[DirectionBlender], Dict, Dict]:
    """Thin wrapper to make the training stage signature explicit."""
    return train_one_window(
        df_window,
        signal_modules,
        context_modules,
        feats_window,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
        regime_label_strategy=regime_label_strategy,
        smoothing_window=smoothing_window,
        use_gpu=use_gpu,
        random_state=random_state,
        calibration_method=calibration_method,
        sharpening_alpha=sharpening_alpha,
        disable_calibration=disable_calibration,
        X_val=X_val,
        y_val=y_val,
        calibration_source=calibration_source,
    )


def validate_models_for_window(
    df_window: pd.DataFrame,
    feats_window: pd.DataFrame,
    reg_model: RegimeDetector,
    blender_model: SignalBlender,
    signal_modules: List,
    context_modules: List,
    horizon_bars: int,
    label_threshold: float,
    regime_label_strategy: str,
    smoothing_window: int,
    eval_mode: str,
    transaction_cost_bps: float,
) -> Dict:
    """Thin wrapper to make the validation stage signature explicit."""
    return validate_one_window(
        df_window,
        reg_model,
        blender_model,
        signal_modules,
        context_modules,
        feats_window,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
        regime_label_strategy=regime_label_strategy,
        smoothing_window=smoothing_window,
        eval_mode=eval_mode,
        transaction_cost_bps=transaction_cost_bps,
    )


def combined_training(
    df: pd.DataFrame,
    feature_store: FeatureStore,
    signal_modules: List,
    context_modules: List,
    models_dir: Path,
    feature_cache_path: Optional[Path] = None,
    force_recompute_cache: bool = False,
    train_months: int = 6,
    val_months: int = 1,
    slide_months: int = 1,
    embargo_days: int = 0,
    purge_bars: int = 0,
    horizon_bars: int = 12,
    label_threshold: float = 0.0005,
    regime_label_strategy: str = "indicator",
    smoothing_window: int = 12,
    use_gpu: bool = False,
    random_state: Optional[int] = None,
    calibration_method: Optional[str] = None,
    sharpening_alpha: float = 2.0,
    disable_calibration: bool = False,
    calibration_source: str = "train",
    eval_mode: str = "nonoverlap",
    transaction_cost_bps: float = 0.0,
    # Ablation flags
    disable_ml_features: bool = False,
    disable_regime_context: bool = False,
    disable_signal_dynamics: bool = False,
    disable_rolling_stats: bool = False,
    disable_modules: Optional[List[str]] = None,
    include_modules: Optional[List[str]] = None,
    include_features: Optional[List[str]] = None,
    exclude_features: Optional[List[str]] = None,
) -> Tuple[RegimeDetector, SignalBlender, Optional[DirectionBlender], Dict]:
    """
    Perform walk-forward training across all windows.
    
    Args:
        df: Full dataset
        feature_store: Shared FeatureStore instance for feature caching/slicing
        signal_modules: List of signal modules
        context_modules: List of context modules
        models_dir: Directory to save models
        feature_cache_path: Optional path to feature cache
        force_recompute_cache: Whether to force recompute features
        train_months: Training window size in months
        val_months: Validation window size in months
        slide_months: Slide forward by this many months
        embargo_days: Gap between train end and validation start in days
        purge_bars: Bars to purge from end of training window (label leakage guard)
        horizon_bars: Forward horizon in bars for direction labels
        label_threshold: Threshold for direction labels
        regime_label_strategy: Regime label strategy ("indicator" or "rule")
        smoothing_window: Smoothing window for direction labels
        use_gpu: Whether to use GPU acceleration for models
        random_state: Random seed for model initialization
        calibration_method: Calibration method for SignalBlender
        sharpening_alpha: Sharpening alpha for calibrated probabilities
        disable_calibration: Disable calibration for SignalBlender
        calibration_source: Source of calibration data ('train' or 'val', default: 'train')
        eval_mode: Return evaluation mode ("nonoverlap" or "per-bar")
        transaction_cost_bps: Cost per side in basis points
        include_modules: Optional list of module names to keep
        include_features: Optional list of regex patterns to include
        exclude_features: Optional list of regex patterns to exclude
        
    Returns:
        Tuple of (final_regime_detector, final_signal_blender, final_direction_blender, final_regime_models_dict)
    """
    # Force a single feature build to avoid O(N x windows) recomputation costs.
    cache_path = feature_cache_path or (models_dir / "cached_features.parquet")
    feature_store, cached_features = build_features_once(
        df,
        signal_modules,
        context_modules,
        cache_path,
        force_recompute=force_recompute_cache,
        use_gpu=False,
        disable_ml_features=disable_ml_features,
        disable_regime_context=disable_regime_context,
        disable_signal_dynamics=disable_signal_dynamics,
        disable_rolling_stats=disable_rolling_stats,
        disable_modules=disable_modules,
        include_modules=include_modules,
        include_features=include_features,
        exclude_features=exclude_features,
    )

    # Pre-load features to GPU once if using GPU and memory check passes
    cached_features_gpu = None
    if use_gpu and _check_gpu_memory_safe_for_preload(cached_features, use_gpu=True):
        try:
            import cudf
            print(f"[GPU] Pre-loading {len(cached_features)} rows to GPU memory...")
            sys.stdout.flush()
            cached_features_gpu = cudf.from_pandas(cached_features)
            print(f"[GPU] Features loaded to GPU: {cached_features_gpu.shape}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[GPU] Warning: Could not pre-load to GPU: {e}")
            print(f"[GPU] Full traceback:\n{traceback.format_exc()}")
            print(f"[GPU] Will convert per-window instead")
            sys.stdout.flush()
            cached_features_gpu = None
    elif use_gpu:
        print(f"[GPU] Skipping pre-load (memory check failed or GPU unavailable), will convert per-window")
        sys.stdout.flush()

    # Generate splits
    splits = prepare_windows(
        df,
        train_months,
        val_months,
        slide_months,
        embargo_days=embargo_days,
        purge_bars=purge_bars,
    )
    
    if len(splits) == 0:
        raise ValueError("No valid splits generated. Check data range and window sizes.")
    
    print(f"\nGenerated {len(splits)} walk-forward splits")
    print(f"Training window: {train_months} months, Validation window: {val_months} months, Slide: {slide_months} months\n")
    sys.stdout.flush()
    
    # Store results
    all_results = []
    best_val_return = -np.inf
    best_reg = None
    best_blender = None
    best_direction_blender = None
    best_regime_models = {}
    
    # Track timing for progress
    start_time = time.time()
    window_times = []
    
    # Walk-forward training
    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, 1):
        window_start_time = time.time()
        
        # Progress bar
        progress_pct = i / len(splits) * 100
        bar_length = 40
        filled = int(bar_length * i / len(splits))
        bar = '=' * filled + '-' * (bar_length - filled)
        elapsed = time.time() - start_time
        avg_time_per_window = elapsed / i if i > 0 else 0
        remaining_windows = len(splits) - i
        eta_seconds = avg_time_per_window * remaining_windows
        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds > 0 else "calculating..."
        
        print(f"\n{'='*60}")
        print(f"Window {i}/{len(splits)} [{progress_pct:.1f}%] |{bar}| ETA: {eta_str}")
        print(f"Train: {train_start.date()} to {train_end.date()}")
        print(f"Val:   {val_start.date()} to {val_end.date()}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        # Extract windows using cached feature index to prevent any recomputation
        # Use GPU features if available, otherwise fallback to pandas
        try:
            features_to_slice = cached_features_gpu if cached_features_gpu is not None else cached_features
            df_train, feats_train = slice_window(df, features_to_slice, train_start, train_end)
            df_val, feats_val = slice_window(df, features_to_slice, val_start, val_end)
        except (ValueError, Exception) as e:
            print(f"Warning: Skipping window {i} due to error: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            sys.stdout.flush()
            continue

        if len(df_train) == 0 or len(df_val) == 0:
            print(f"Warning: Skipping window {i} - insufficient data")
            sys.stdout.flush()
            continue

        # Train on window using cached features to prevent repeated computation
        print("Training models...")
        sys.stdout.flush()
        try:
            # Prepare validation data for calibration if needed
            X_val_window = None
            y_val_window = None
            if calibration_source == "val":
                # Build validation features and labels for calibration
                regime_y_val = build_regime_labels(df_val, feats_val, regime_label_strategy)
                df_labeled_val = make_direction_labels(
                    df_val,
                    horizon_bars=horizon_bars,
                    label_threshold=label_threshold,
                    smoothing_window=smoothing_window,
                    debug=False
                )
                blend_y_val = df_labeled_val['blend_label'].reindex(feats_val.index).fillna(0).astype(int)
                
                valid_mask_val = ~(regime_y_val.isna() | blend_y_val.isna())
                valid_mask_val = valid_mask_val & (regime_y_val.isin(['trend_up', 'trend_down', 'chop']))
                
                X_blend_base_val = feats_val[valid_mask_val]
                blend_y_val_clean = blend_y_val[valid_mask_val]
                
                # Compute module signals for validation
                module_signals_val = {}
                for module in signal_modules:
                    module_features_val = X_blend_base_val.filter(regex=f'^{module.name}_', axis=1).copy()
                    if not module_features_val.empty:
                        module_features_val.columns = [col.replace(f'{module.name}_', '') for col in module_features_val.columns]
                    module_signals_val[module.name] = module.compute_signal(module_features_val)
                
                X_blend_val = X_blend_base_val.copy()
                for module_name, signal in module_signals_val.items():
                    X_blend_val[f"{module_name}_signal"] = signal[valid_mask_val]
                X_blend_val["regime"] = regime_y_val[valid_mask_val].map({
                    'trend_up': 0, 'trend_down': 1, 'chop': 2
                }).fillna(2)
                
                X_val_window = X_blend_val
                y_val_window = blend_y_val_clean
            
            reg, blender, direction_blender, train_stats, regime_models = train_models_for_window(
                df_train,
                feats_train,
                signal_modules,
                context_modules,
                horizon_bars,
                label_threshold,
                regime_label_strategy,
                smoothing_window,
                use_gpu,
                random_state,
                calibration_method,
                sharpening_alpha,
                disable_calibration,
                X_val=X_val_window,
                y_val=y_val_window,
                calibration_source=calibration_source,
            )
            print(f"  Training samples: {train_stats['train_samples']}")
            print(f"  Features: {train_stats['feature_count']}")
            sys.stdout.flush()
        except Exception as e:
            print(f"  Error during training: {e}")
            print(f"  Full traceback:\n{traceback.format_exc()}")
            sys.stdout.flush()
            continue

        # Validate on window
        print("Validating models...")
        sys.stdout.flush()
        try:
            val_metrics = validate_models_for_window(
                df_val,
                feats_val,
                reg,
                blender,
                signal_modules,
                context_modules,
                horizon_bars,
                label_threshold,
                regime_label_strategy,
                smoothing_window,
                eval_mode,
                transaction_cost_bps,
            )
            print(f"  Validation samples: {val_metrics['val_samples']}")
            print(f"  Regime accuracy: {val_metrics['regime_accuracy']:.4f}")
            print(f"  Blend accuracy: {val_metrics['blend_accuracy']:.4f}")
            print(f"  Net cumulative return: {val_metrics['cumulative_return']:.4f}")
            print(f"  Gross cumulative return: {val_metrics['gross_cumulative_return']:.4f}")

            # Track best model
            if val_metrics['cumulative_return'] > best_val_return:
                best_val_return = val_metrics['cumulative_return']
                best_reg = reg
                best_blender = blender
                best_direction_blender = direction_blender
                best_regime_models = regime_models
                print(f"  *** New best model (return: {best_val_return:.4f}) ***")
            
            # Window timing
            window_time = time.time() - window_start_time
            window_times.append(window_time)
            avg_window_time = np.mean(window_times[-10:]) if len(window_times) > 0 else window_time  # Last 10 windows
            print(f"  Window completed in {window_time:.1f}s (avg: {avg_window_time:.1f}s)")
            sys.stdout.flush()
        except Exception as e:
            print(f"  Error during validation: {e}")
            print(f"  Full traceback:\n{traceback.format_exc()}")
            sys.stdout.flush()
            continue
        
        # Store results
        result = {
            'window': i,
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end,
            **train_stats,
            **val_metrics
        }
        all_results.append(result)
        
        # Periodic status update every 10 windows
        if i % 10 == 0:
            elapsed_total = time.time() - start_time
            print(f"\n[Progress] Completed {i}/{len(splits)} windows in {elapsed_total/60:.1f} minutes")
            print(f"[Progress] Best return so far: {best_val_return:.4f}")
            sys.stdout.flush()
    
    # Summary
    print(f"\n{'='*60}")
    print("WALK-FORWARD TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if len(all_results) > 0:
        avg_regime_acc = np.mean([r['regime_accuracy'] for r in all_results])
        avg_blend_acc = np.mean([r['blend_accuracy'] for r in all_results])
        avg_return = np.mean([r['cumulative_return'] for r in all_results])
        returns = np.array([r['cumulative_return'] for r in all_results], dtype=float)
        median_return = float(np.median(returns))
        p5 = float(np.percentile(returns, 5))
        p95 = float(np.percentile(returns, 95))
        trimmed_returns = returns[(returns >= p5) & (returns <= p95)]
        trimmed_mean = float(trimmed_returns.mean()) if trimmed_returns.size else float(avg_return)
        positive_pct = float((returns > 0).mean())
        
        print(f"Windows processed: {len(all_results)}")
        print(f"Average regime accuracy: {avg_regime_acc:.4f}")
        print(f"Average blend accuracy: {avg_blend_acc:.4f}")
        print(f"Average cumulative return: {avg_return:.4f}")
        print(f"Median cumulative return: {median_return:.4f}")
        print(f"Trimmed mean return (5-95%): {trimmed_mean:.4f}")
        print(f"Positive-return windows: {positive_pct:.2%}")
        print(f"Best validation return: {best_val_return:.4f}")
        
        # Save results log
        results_df = pd.DataFrame(all_results)
        results_path = models_dir / "training_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
    else:
        print("No windows successfully processed!")
        raise ValueError("No valid training windows processed")
    
    # Use best model or last model if no best found
    if best_reg is None or best_blender is None:
        print("\nWarning: No best model found, using last trained model")
        if len(all_results) > 0:
            # Retrain on last window
            last_split = splits[-1]
            df_train, feats_last = slice_window(df, cached_features, last_split[0], last_split[1])
            best_reg, best_blender, best_direction_blender, _, best_regime_models = train_models_for_window(
                df_train,
                feats_last,
                signal_modules,
                context_modules,
                horizon_bars,
                label_threshold,
                regime_label_strategy,
                smoothing_window,
                use_gpu,
                random_state,
                calibration_method,
                sharpening_alpha,
                disable_calibration,
                calibration_source=calibration_source,
            )
        else:
            raise ValueError("Cannot create final models - no successful training")
    
    return best_reg, best_blender, best_direction_blender, best_regime_models


def downsample_validation_for_calibration(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_size: int = 50000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Downsample validation set for calibration to avoid GPU OOM.
    Uses stratified sampling based on labels.
    
    Args:
        X_val: Validation features
        y_val: Validation labels
        target_size: Target number of samples (default: 50000)
        random_state: Random seed for sampling
        
    Returns:
        Tuple of (X_val_cal, y_val_cal) - downsampled validation set
    """
    if len(X_val) <= target_size:
        return X_val, y_val
    
    from sklearn.model_selection import train_test_split
    
    # Use stratified sampling to preserve label distribution
    X_val_cal, _, y_val_cal, _ = train_test_split(
        X_val, y_val,
        train_size=target_size,
        random_state=random_state,
        stratify=y_val
    )
    
    return X_val_cal, y_val_cal


def train_full_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_class: type = SignalBlender,
    model_name: str = "SignalBlender",
    calibration_method: str = 'isotonic',
    sharpening_alpha: float = 2.0,
    prune_bottom_percentile: float = 20.0,
    models_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
    use_gpu: bool = False,
    random_state: int = 42,
) -> Tuple:
    """
    Full training pipeline with feature pruning, importance analysis, and diagnostics.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features (if None, auto-splits 20% stratified)
        y_val: Optional validation labels
        model_class: Model class (SignalBlender or DirectionBlender)
        model_name: Name for saving artifacts
        calibration_method: Calibration method ('isotonic' or 'platt')
        sharpening_alpha: Sharpening parameter
        prune_bottom_percentile: Bottom percentile to prune (e.g., 20.0 = bottom 20%)
        models_dir: Directory to save models
        results_dir: Directory to save results
        random_state: Random seed for internal splits
        
    Returns:
        Tuple of (trained_model, diagnostics_dict)
    """
    from sklearn.model_selection import train_test_split
    
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name} - FULL PIPELINE")
    print(f"{'='*60}")
    
    # 1. Auto-split validation if not provided
    if X_val is None or y_val is None:
        print(f"\n[1/7] Auto-splitting 20% for validation (stratified)...")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        X_train = X_train_split
        X_val = X_val_split
        y_train = y_train_split
        y_val = y_val_split
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
    else:
        print(f"\n[1/7] Using provided validation set")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
    
    n_features_before = len(X_train.columns)
    
    # 1.5. Downsample validation set for calibration (to avoid GPU OOM)
    X_val_cal, y_val_cal = downsample_validation_for_calibration(
        X_val,
        y_val,
        target_size=50000,
        random_state=random_state
    )
    print(f"[{model_name}] Using {len(X_val_cal)} samples for calibration (downsampled from {len(X_val)}).")
    
    # 2. Initial training
    print(f"\n[2/7] Initial training...")
    model = model_class(use_gpu=use_gpu, random_state=random_state)
    model.fit(
        X_train, y_train,
        calibration_method=calibration_method,
        sharpening_alpha=sharpening_alpha,
        X_val=X_val_cal,  # Use downsampled set for calibration
        y_val=y_val_cal
    )
    
    # 3. Store and save feature importances
    print(f"\n[3/7] Storing feature importances...")
    model._store_feature_importances()
    
    # Get importances for pruning
    feature_importances = getattr(model, "feature_importances_", None)
    
    # Check if importances are available and non-zero
    skip_pruning = False
    if feature_importances is None:
        print(f"[{model_name}] Feature importances not available. Skipping pruning.")
        skip_pruning = True
    elif np.all(feature_importances == 0):
        print(f"[{model_name}] GPU mode: skipping feature pruning since importances unavailable.")
        skip_pruning = True
    
    if not skip_pruning:
        importance_stats = {
            'mean': float(np.mean(feature_importances)),
            'median': float(np.median(feature_importances)),
            'std': float(np.std(feature_importances)),
            'max': float(np.max(feature_importances)),
            'min': float(np.min(feature_importances))
        }
    else:
        importance_stats = {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0
        }
    
    # 4. Feature pruning
    if skip_pruning:
        print(f"\n[4/7] Skipping feature pruning (importances unavailable)...")
        kept_features = model.feature_names
        n_features_after = len(kept_features)
    else:
        print(f"\n[4/7] Pruning features (bottom {prune_bottom_percentile}%)...")
        kept_features = prune_features_by_importance(
            feature_importances,
            model.feature_names,
            bottom_percentile=prune_bottom_percentile
        )
        n_features_after = len(kept_features)
    
    if n_features_after < n_features_before:
        # Retrain with pruned features
        print(f"  Retraining with {n_features_after} features...")
        X_train_pruned = X_train[kept_features]
        X_val_pruned = X_val[kept_features]
        
        # Downsample validation for calibration (pruned features)
        X_val_pruned_cal, y_val_pruned_cal = downsample_validation_for_calibration(
            X_val_pruned,
            y_val,
            target_size=50000,
            random_state=random_state
        )
        print(f"[{model_name}] Using {len(X_val_pruned_cal)} samples for calibration (downsampled from {len(X_val_pruned)}).")
        
        model_pruned = model_class(use_gpu=use_gpu, random_state=random_state)
        model_pruned.fit(
            X_train_pruned, y_train,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            X_val=X_val_pruned_cal,  # Use downsampled set for calibration
            y_val=y_val_pruned_cal
        )
        # Recompute feature importances after retraining
        model_pruned._store_feature_importances()
        model = model_pruned
    
    # Save selected features
    models_dir.mkdir(parents=True, exist_ok=True)
    features_path = models_dir / f"selected_features_{model_name.lower()}.txt"
    save_selected_features(model.feature_names, features_path)
    
    # 5. Probability histogram
    print(f"\n[5/7] Generating probability histograms...")
    proba_val = model.predict_proba(X_val)
    
    if model_name == "SignalBlender":
        class_names = ['short', 'flat', 'long']  # -1, 0, 1
        # Map to actual column names
        proba_cols = []
        for name in class_names:
            if name == 'short' and -1 in proba_val.columns:
                proba_cols.append(-1)
            elif name == 'flat' and 0 in proba_val.columns:
                proba_cols.append(0)
            elif name == 'long' and 1 in proba_val.columns:
                proba_cols.append(1)
    else:
        class_names = ['short', 'long']  # -1, 1
        proba_cols = []
        for name in class_names:
            if name == 'short' and -1 in proba_val.columns:
                proba_cols.append(-1)
            elif name == 'long' and 1 in proba_val.columns:
                proba_cols.append(1)
    
    if not proba_cols:
        proba_cols = proba_val.columns.tolist()
    
    hist_path = results_dir / f"prob_hist_{model_name.lower()}.png"
    plot_probability_histograms(proba_val, proba_cols, hist_path, title=f"{model_name} Probability Distribution")
    
    # Compute histogram statistics
    histogram_stats = {}
    for col in proba_cols:
        probs = proba_val[col].values
        probs = probs[~np.isnan(probs)]
        if len(probs) > 0:
            histogram_stats[f"{col}_mean"] = float(np.mean(probs))
            histogram_stats[f"{col}_std"] = float(np.std(probs))
            histogram_stats[f"{col}_p5"] = float(np.percentile(probs, 5))
            histogram_stats[f"{col}_p95"] = float(np.percentile(probs, 95))
    
    # 6. Calibration statistics
    calibration_stats = {
        'method': calibration_method,
        'sharpening_alpha': sharpening_alpha,
        'n_calibrators': len(model.calibrators) if hasattr(model, 'calibrators') else 0
    }
    
    # 7. Create diagnostics report
    print(f"\n[6/7] Creating training diagnostics report...")
    diagnostics_path = results_dir / "training_diagnostics.txt"
    create_training_diagnostics_report(
        model_name=model_name,
        n_features_before=n_features_before,
        n_features_after=n_features_after,
        feature_importance_stats=importance_stats,
        calibration_stats=calibration_stats,
        histogram_stats=histogram_stats,
        output_path=diagnostics_path
    )
    
    # 8. Save model
    print(f"\n[7/7] Saving model...")
    model_path = models_dir / f"{model_name.lower()}_calibrated.pkl"
    model.save(str(model_path))
    print(f"  Model saved to {model_path}")
    
    diagnostics = {
        'n_features_before': n_features_before,
        'n_features_after': n_features_after,
        'importance_stats': importance_stats,
        'calibration_stats': calibration_stats,
        'histogram_stats': histogram_stats
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name} TRAINING COMPLETE")
    print(f"{'='*60}")
    
    return model, diagnostics


def main():
    """
    Main training function.
    """
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Train regime detector and signal blender models')
    parser.add_argument('--calibration-method', type=str, default=None, choices=['isotonic', 'platt'],
                        help='Probability calibration method: isotonic or platt (default: isotonic, always enabled)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Post-hoc sharpening parameter: prob = prob ** alpha (default: 2.0, range: 1.0-3.0)')
    parser.add_argument('--disable-calibration', action='store_true',
                        help='Disable probability calibration entirely')
    parser.add_argument('--use-full-pipeline', action='store_true',
                        help='Use full training pipeline with feature pruning and diagnostics')
    parser.add_argument('--horizon-bars', type=int, default=12,
                        help='Forward horizon in bars for label generation (default: 12 = 1 hour for 5-min data)')
    parser.add_argument('--label-threshold', type=float, default=0.0005,
                        help='Threshold for direction labels (default: 0.0005 = 0.05%%)')
    parser.add_argument('--min-trade-proba', type=float, default=0.0,
                        help='Minimum SignalBlender max proba to take a trade; below -> flat (default: 0.0)')
    parser.add_argument('--fee-bps', type=float, default=1.0,
                        help='Transaction cost per side in bps; use 0 to disable (default: 1.0)')
    parser.add_argument('--eval-mode', type=str, default='nonoverlap', choices=['nonoverlap', 'per-bar'],
                        help='Return evaluation mode: nonoverlap or per-bar (default: nonoverlap)')
    
    # Feature ablation flags (for A/B testing feature groups)
    parser.add_argument('--disable-ml-features', action='store_true',
                        help='Disable ML features (returns, volatility, RSI, SMA distances, candle metrics)')
    parser.add_argument('--disable-regime-context', action='store_true',
                        help='Disable regime context features (ATR ratios, volatility ratios, trend slopes)')
    parser.add_argument('--disable-signal-dynamics', action='store_true',
                        help='Disable signal dynamics features (streaks, time-since-signal, derivatives)')
    parser.add_argument('--disable-rolling-stats', action='store_true',
                        help='Disable rolling statistics (rolling mean/std/max/min/zscore)')
    parser.add_argument('--disable-modules', type=str, default=None,
                        help='Comma-separated list of modules to disable (e.g., "superma,trendmagic,pvt")')
    parser.add_argument('--include-modules', type=str, default=None,
                        help='Comma-separated list of modules to keep (others disabled)')
    parser.add_argument('--include-features', type=str, default=None,
                        help='Comma-separated regex patterns of features to keep (applied after generation)')
    parser.add_argument('--exclude-features', type=str, default=None,
                        help='Comma-separated regex patterns of features to drop (applied after generation)')
    
    parser.add_argument('--runpod', action='store_true',
                        help='Use RunPod workspace layout (/workspace/Hybridyzer) for data, models, and results')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU even if cuML is installed')
    parser.add_argument('--walkforward', action='store_true',
                        help='Use walk-forward training instead of the single 80/20 split')
    parser.add_argument('--train-months', type=int, default=6,
                        help='Walk-forward training window size in months (default: 6)')
    parser.add_argument('--val-months', type=int, default=1,
                        help='Walk-forward validation window size in months (default: 1)')
    parser.add_argument('--slide-months', type=int, default=1,
                        help='Walk-forward slide step in months (default: 1)')
    parser.add_argument('--embargo-days', type=int, default=0,
                        help='Embargo gap in days between training end and validation start (default: 0)')
    parser.add_argument('--purge-bars', type=int, default=0,
                        help='Bars to purge from end of training window to reduce label leakage (default: 0)')
    parser.add_argument('--purge-horizon', action='store_true',
                        help='Purge horizon_bars from end of training window (overrides --purge-bars)')
    parser.add_argument('--smoothing-window', type=int, default=12,
                        help='Smoothing window for direction labels (default: 12)')
    parser.add_argument('--regime-labels', type=str, default='indicator', choices=['indicator', 'rule'],
                        help='Regime label strategy: indicator (make_regime_labels) or rule (rule_based_regime)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--calibration-source', type=str, default='train', choices=['train', 'val'],
                        help='Calibration data source for walk-forward: train (use training window) or val (use validation window) (default: train)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file (default: results/train.log)')
    args = parser.parse_args()

    # Configuration
    base_dir = Path("/workspace/Hybridyzer") if args.runpod else Path.cwd()
    base_dir.mkdir(parents=True, exist_ok=True)
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else (results_dir / "train.log")
    log_file, _, _ = _setup_logging(log_path)
    print(f"[Logging] Writing output to {log_path}")

    # Enable faulthandler for crash debugging (dumps traceback on SIGSEGV, SIGFPE, etc.)
    try:
        faulthandler.enable(file=log_file)
        print("[Debug] faulthandler enabled for crash diagnostics")
    except Exception as exc:
        faulthandler.enable(file=sys.__stderr__)
        print(f"[Debug] faulthandler fallback to sys.__stderr__: {exc}")

    set_global_seed(args.seed)
    print(f"[Seed] Using random seed: {args.seed}")
    regime_label_strategy = args.regime_labels
    print(f"[Labels] Regime label strategy: {regime_label_strategy}")
    
    # GPU detection: check for cuML availability
    try:
        import cudf  # noqa: F401
        import cuml  # noqa: F401
        gpu_available = True
    except ImportError:
        gpu_available = False
    
    if args.cpu_only:
        use_cuml = False
        print("[GPU] cpu_only flag set: forcing CPU backend")
    else:
        use_cuml = gpu_available
        print(f"[GPU] cuML available: {use_cuml}")
    
    # Calibration settings - defaults to always calibrate
    if args.disable_calibration:
        calibration_method = None
        sharpening_alpha = 1.0
    else:
        calibration_method = args.calibration_method if args.calibration_method else 'isotonic'
        sharpening_alpha = args.alpha if args.alpha is not None else 2.0
        sharpening_alpha = max(1.0, min(3.0, sharpening_alpha))  # Clamp to [1.0, 3.0]
    
    print(f"\nCalibration settings:")
    print(f"  Method: {calibration_method if calibration_method else 'DISABLED'}")
    print(f"  Sharpening alpha: {sharpening_alpha}")
    print(f"[Evaluation] Mode: {args.eval_mode}, Fee bps: {args.fee_bps:.2f}")

    purge_bars = args.horizon_bars if args.purge_horizon else args.purge_bars
    
    # New label system configuration
    HORIZON_BARS = args.horizon_bars  # Forward horizon in bars (default: 12 = 1 hour for 5-min data)
    LABEL_THRESHOLD = args.label_threshold  # Threshold for direction labels (default: 0.0005)
    SMOOTHING_WINDOW = args.smoothing_window  # Rolling window for smoothing returns
    
    # 1. Load BTC data with load_btc_csv
    print("Loading BTC CSV data...")
    
    # For walk-forward training, use split datasets (train/val/test)
    if args.walkforward:
        train_path = data_dir / 'btcusd_5min_train_2017_2022.csv'
        val_path = data_dir / 'btcusd_5min_val_2023.csv'
        test_path = data_dir / 'btcusd_5min_test_2024.csv'
        
        if train_path.exists() and val_path.exists():
            print(f"Loading split datasets for walk-forward training...")
            df_train_split = load_btc_csv(str(train_path))
            df_val_split = load_btc_csv(str(val_path))
            # Combine train and val for walk-forward (test is separate)
            df = pd.concat([df_train_split, df_val_split]).sort_index()
            data_path = f"{train_path.name} + {val_path.name}"
            print(f"Loaded train: {len(df_train_split)} bars from {train_path}")
            print(f"Loaded val: {len(df_val_split)} bars from {val_path}")
            if test_path.exists():
                print(f"Note: Test set available at {test_path.name} (not used in walk-forward)")
        else:
            print(f"Warning: Split datasets not found, falling back to single file...")
            # Fall back to single file
            csv_paths = [
                data_dir / 'btcusd_5min.csv',
                data_dir / 'btcusd_4H.csv',
                data_dir / 'btcusd_1min.csv',
                data_dir / 'btc_1m.csv',
                data_dir / 'btcusd_1min_volspikes.csv',
                data_dir / 'btc_data.csv',
            ]
            df = None
            data_path = None
            for path in csv_paths:
                try:
                    df = load_btc_csv(str(path))
                    data_path = path
                    print(f"Loaded from: {path}")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    df = load_btc_csv(str(csv_files[0]))
                    data_path = csv_files[0]
                    print(f"Loaded from: {csv_files[0]}")
                else:
                    raise FileNotFoundError("No CSV files found in data/ directory")
    else:
        # For static training, prefer split datasets but combine train+val
        train_path = data_dir / 'btcusd_5min_train_2017_2022.csv'
        val_path = data_dir / 'btcusd_5min_val_2023.csv'
        
        if train_path.exists() and val_path.exists():
            print(f"Loading split datasets for static training...")
            df_train_split = load_btc_csv(str(train_path))
            df_val_split = load_btc_csv(str(val_path))
            # Combine train and val for static training
            df = pd.concat([df_train_split, df_val_split]).sort_index()
            data_path = f"{train_path.name} + {val_path.name}"
            print(f"Loaded train: {len(df_train_split)} bars from {train_path}")
            print(f"Loaded val: {len(df_val_split)} bars from {val_path}")
        else:
            # Fall back to single file
            csv_paths = [
                data_dir / 'btcusd_5min.csv',
                data_dir / 'btcusd_4H.csv',
                data_dir / 'btcusd_1min.csv',
                data_dir / 'btc_1m.csv',
                data_dir / 'btcusd_1min_volspikes.csv',
                data_dir / 'btc_data.csv',
            ]
            df = None
            data_path = None
            for path in csv_paths:
                try:
                    df = load_btc_csv(str(path))
                    data_path = path
                    print(f"Loaded from: {path}")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    df = load_btc_csv(str(csv_files[0]))
                    data_path = csv_files[0]
                    print(f"Loaded from: {csv_files[0]}")
                else:
                    raise FileNotFoundError("No CSV files found in data/ directory")
    
    if df.empty:
        print("Error: No data loaded.")
        return
    
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Parse ablation flags
    disable_modules_list = None
    if args.disable_modules:
        disable_modules_list = [m.strip() for m in args.disable_modules.split(',') if m.strip()]

    include_modules_list = None
    if args.include_modules:
        include_modules_list = [m.strip() for m in args.include_modules.split(',') if m.strip()]

    include_features_list = None
    if args.include_features:
        include_features_list = [f.strip() for f in args.include_features.split(',') if f.strip()]

    exclude_features_list = None
    if args.exclude_features:
        exclude_features_list = [f.strip() for f in args.exclude_features.split(',') if f.strip()]
    
    # Log ablation settings
    ablation_settings = {
        'disable_ml_features': args.disable_ml_features,
        'disable_regime_context': args.disable_regime_context,
        'disable_signal_dynamics': args.disable_signal_dynamics,
        'disable_rolling_stats': args.disable_rolling_stats,
        'disable_modules': disable_modules_list,
        'include_modules': include_modules_list,
        'include_features': include_features_list,
        'exclude_features': exclude_features_list,
    }
    if any(v for v in ablation_settings.values() if v):
        print(f"\n[ABLATION] Feature ablation enabled:")
        for k, v in ablation_settings.items():
            if v:
                print(f"  {k}: {v}")
    
    # 2. Build or load cached features once to avoid recomputation across runs
    cache_path = models_dir / "cached_features.parquet"
    feature_store = FeatureStore(
        disable_ml_features=args.disable_ml_features,
        disable_regime_context=args.disable_regime_context,
        disable_signal_dynamics=args.disable_signal_dynamics,
        disable_rolling_stats=args.disable_rolling_stats,
        disable_modules=disable_modules_list,
        include_modules=include_modules_list,
        include_features=include_features_list,
        exclude_features=exclude_features_list,
    )
    feature_store, feats = build_features_once(
        df,
        signal_modules=feature_store.signal_modules,
        context_modules=feature_store.context_modules,
        cache_path=cache_path,
        use_gpu=False,
        disable_ml_features=args.disable_ml_features,
        disable_regime_context=args.disable_regime_context,
        disable_signal_dynamics=args.disable_signal_dynamics,
        disable_rolling_stats=args.disable_rolling_stats,
        disable_modules=disable_modules_list,
        include_modules=include_modules_list,
        include_features=include_features_list,
        exclude_features=exclude_features_list,
    )
    print(f"[Features] Loaded cached features: {feats.shape}")
    
    # Print final feature columns for verification
    print("\nFINAL FEATURE COLUMNS:", feats.columns.tolist())

    if args.walkforward:
        print("\n" + "=" * 60)
        print("WALK-FORWARD TRAINING")
        print("=" * 60)
        print(f"[Calibration] Using {args.calibration_source} data for calibration")
        best_reg, best_blender, best_direction_blender, best_regime_models = combined_training(
            df=df,
            feature_store=feature_store,
            signal_modules=feature_store.signal_modules,
            context_modules=feature_store.context_modules,
            models_dir=models_dir,
            feature_cache_path=cache_path,
            force_recompute_cache=False,
            train_months=args.train_months,
            val_months=args.val_months,
            slide_months=args.slide_months,
            embargo_days=args.embargo_days,
            purge_bars=purge_bars,
            horizon_bars=HORIZON_BARS,
            label_threshold=LABEL_THRESHOLD,
            regime_label_strategy=regime_label_strategy,
            smoothing_window=SMOOTHING_WINDOW,
            use_gpu=use_cuml,
            random_state=args.seed,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            disable_calibration=args.disable_calibration,
            calibration_source=args.calibration_source,
            eval_mode=args.eval_mode,
            transaction_cost_bps=args.fee_bps,
            # Ablation flags
            disable_ml_features=args.disable_ml_features,
            disable_regime_context=args.disable_regime_context,
            disable_signal_dynamics=args.disable_signal_dynamics,
            disable_rolling_stats=args.disable_rolling_stats,
            disable_modules=disable_modules_list,
            include_modules=include_modules_list,
            include_features=include_features_list,
            exclude_features=exclude_features_list,
        )

        # Save all models
        best_reg.save(str(models_dir / "regime_model.pkl"))
        best_blender.save(str(models_dir / "blender_model.pkl"))
        print("\nWalk-forward models saved:")
        print(f"  - {models_dir / 'regime_model.pkl'}")
        print(f"  - {models_dir / 'blender_model.pkl'}")
        
        if best_direction_blender is not None:
            best_direction_blender.save(str(models_dir / "blender_direction_model.pkl"))
            print(f"  - {models_dir / 'blender_direction_model.pkl'}")
        
        # Save regime-specific models
        regime_models_saved = []
        for regime_name, model_r in best_regime_models.items():
            slug = regime_name.lower().replace(" ", "_")
            out_path = models_dir / f"blender_{slug}.pkl"
            model_r.save(str(out_path))
            regime_models_saved.append(f"blender_{slug}.pkl")
            print(f"  - {out_path}")

        if best_reg.feature_names:
            save_selected_features(best_reg.feature_names, models_dir / "feature_columns.txt")
        if best_blender.feature_names:
            save_selected_features(best_blender.feature_names, models_dir / "blend_feature_columns.txt")

        manifest = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_type": "walkforward",
            "seed": args.seed,
            "use_gpu": use_cuml,
            "data_path": data_path,
            "feature_cache_path": cache_path,
            "log_path": log_path,
            "calibration": {
                "method": calibration_method,
                "sharpening_alpha": sharpening_alpha,
                "disabled": args.disable_calibration,
                "source": args.calibration_source
            },
            "evaluation": {
                "eval_mode": args.eval_mode,
                "transaction_cost_bps": args.fee_bps
            },
            "ablation": {
                "disable_ml_features": args.disable_ml_features,
                "disable_regime_context": args.disable_regime_context,
                "disable_signal_dynamics": args.disable_signal_dynamics,
                "disable_rolling_stats": args.disable_rolling_stats,
                "disable_modules": disable_modules_list,
                "include_modules": include_modules_list,
                "include_features": include_features_list,
                "exclude_features": exclude_features_list,
            },
            "label_config": {
                "horizon_bars": HORIZON_BARS,
                "label_threshold": LABEL_THRESHOLD,
                "smoothing_window": SMOOTHING_WINDOW
            },
            "regime_label_strategy": regime_label_strategy,
            "walkforward": {
                "train_months": args.train_months,
                "val_months": args.val_months,
                "slide_months": args.slide_months,
                "embargo_days": args.embargo_days,
                "purge_bars": purge_bars
            },
            "modules": {
                "signal_modules": [m.name for m in feature_store.signal_modules],
                "context_modules": [m.name for m in feature_store.context_modules]
            },
            "feature_columns_path": models_dir / "feature_columns.txt",
            "blend_feature_columns_path": models_dir / "blend_feature_columns.txt",
            "model_paths": {
                "regime_model": models_dir / "regime_model.pkl",
                "signal_blender": models_dir / "blender_model.pkl",
                "direction_blender": models_dir / "blender_direction_model.pkl" if best_direction_blender is not None else None,
                "regime_specific_signal": [models_dir / name for name in regime_models_saved]
            },
            "results_paths": {
                "training_results": models_dir / "training_results.csv"
            }
        }
        manifest_path = models_dir / "training_manifest.json"
        _write_json(manifest_path, manifest)
        print(f"\nTraining manifest saved to {manifest_path}")
        return
    
    # 3. Merge raw price into features for labeling
    print("\nMerging raw price data into features...")
    feats_with_price = feats.copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            feats_with_price[col] = df[col].reindex(feats.index)
    
    # 4. Generate direction labels for SignalBlender (NEW LABEL SYSTEM)
    print("\nGenerating direction labels (NEW SYSTEM)...")
    print(f"  Forward horizon: {HORIZON_BARS} bars")
    print(f"  Label threshold: {LABEL_THRESHOLD}")
    print(f"  Smoothing window: {SMOOTHING_WINDOW} bars")
    
    df_labeled = make_direction_labels(
        df,
        horizon_bars=HORIZON_BARS,
        label_threshold=LABEL_THRESHOLD,
        smoothing_window=SMOOTHING_WINDOW,
        debug=True
    )
    
    # Extract labels
    # Use raw OHLCV data for regime labeling (not features)
    y_regime = build_regime_labels(df, feats, regime_label_strategy)
    y_blend = df_labeled['blend_label'].reindex(feats.index).fillna(0).astype(int)
    future_returns = df_labeled['future_return'].reindex(feats.index)
    smoothed_returns = df_labeled['smoothed_return'].reindex(feats.index)
    
    # Print label statistics (regime labels)
    print(f"\nRegime label distribution:\n{y_regime.value_counts()}")
    
    # Note: Detailed direction label stats are printed by make_direction_labels() with debug=True
    # Print first 5 rows for debugging
    print(f"\nFirst 5 rows (debug):")
    debug_df = pd.DataFrame({
        'future_return': future_returns.iloc[:5],
        'smoothed_return': smoothed_returns.iloc[:5],
        'label': y_blend.iloc[:5]
    })
    print(debug_df)
    
    # Validation sanity checks for direction labels
    print(f"\nDirection label validation checks:")
    unique_blend_labels = y_blend.nunique()
    print(f"  Unique direction labels: {unique_blend_labels}")
    assert unique_blend_labels == 3, f"Direction label collapse detected: only {unique_blend_labels} unique labels found (expected 3)"
    
    max_blend_class_pct = y_blend.value_counts(normalize=True).max()
    print(f"  Max direction class percentage: {max_blend_class_pct:.2%}")
    assert max_blend_class_pct < 0.8, f"Direction class imbalance too extreme: {max_blend_class_pct:.2%} in single class (max allowed: 80%)"
    
    print("  ✓ Direction label validation passed")
    
    # Validation sanity checks for regime labels
    print(f"\nRegime label validation checks:")
    unique_regime_labels = y_regime.nunique()
    print(f"  Unique regime labels: {unique_regime_labels}")
    assert unique_regime_labels == 3, f"Regime label collapse detected: only {unique_regime_labels} unique labels found (expected 3: trend_up, trend_down, chop)"
    
    max_regime_class_pct = y_regime.value_counts(normalize=True).max()
    print(f"  Max regime class percentage: {max_regime_class_pct:.2%}")
    assert max_regime_class_pct < 0.8, f"Regime class imbalance too extreme: {max_regime_class_pct:.2%} in single class (max allowed: 80%)"
    
    # Check for expected regime distribution
    regime_counts = y_regime.value_counts()
    print(f"\n  Regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(y_regime) * 100
        print(f"    {regime}: {count:6d} ({pct:5.2f}%)")
    
    print("  ✓ Regime label validation passed")
    
    # 5. Create X_regime, y_regime, X_blend, y_blend
    # Remove rows with NaN or invalid labels
    valid_mask = ~(y_regime.isna() | y_blend.isna())
    valid_mask = valid_mask & (y_regime.isin(['trend_up', 'trend_down', 'chop']))
    
    X_regime = feats[valid_mask]
    y_regime = y_regime[valid_mask]
    y_blend = y_blend[valid_mask]
    
    # For X_blend, we need features + module signals + regime
    # Compute module signals
    print("\nComputing module signals for blender...")
    module_signals = {}
    for module in feature_store.signal_modules:
        try:
            # Recompute module-specific base features directly from raw df to avoid
            # any columns dropped by safe_mode (e.g., high-NaN warmup features).
            # This ensures all internal features expected by compute_signal()
            # (such as 'botvecMA') are available.
            module_base_feats = module.compute_features(df)
            module_base_feats = module_base_feats.reindex(X_regime.index)
            # Compute signals using module's native feature set
            module_signals[module.name] = module.compute_signal(module_base_feats)
        except Exception as e:
            print(f"Warning: Failed to compute module signals for {module.name}: {e}")
            continue
    
    # Build X_blend with features + signals + regime
    X_blend = X_regime.copy()
    for module_name, signal in module_signals.items():
        aligned_signal = signal.reindex(X_blend.index).fillna(0)
        X_blend[f"{module_name}_signal"] = aligned_signal
    
    # Encode regime as numeric
    regime_map = {
        'trend_up': 0, 'trend_down': 1, 'chop': 2
    }
    X_blend["regime"] = y_regime.map(regime_map).fillna(2)  # Default to chop (2)

    feature_columns_path = models_dir / "feature_columns.txt"
    blend_feature_columns_path = models_dir / "blend_feature_columns.txt"
    save_selected_features(X_regime.columns.tolist(), feature_columns_path)
    save_selected_features(X_blend.columns.tolist(), blend_feature_columns_path)
    
    print(f"\nValid samples: {len(X_regime)}")
    print(f"X_regime shape: {X_regime.shape}")
    print(f"X_blend shape: {X_blend.shape}")
    
    # 6. Time-series train/validation split (last 20% for validation)
    # Do NOT shuffle - preserve temporal order
    split_idx = int(len(X_regime) * 0.8)
    
    # RegimeDetector split
    X_regime_train = X_regime.iloc[:split_idx]
    X_regime_val = X_regime.iloc[split_idx:]
    y_regime_train = y_regime.iloc[:split_idx]
    y_regime_val = y_regime.iloc[split_idx:]
    
    # SignalBlender split (must align with X_blend)
    X_blend_train = X_blend.iloc[:split_idx]
    X_blend_val = X_blend.iloc[split_idx:]
    y_blend_train = y_blend.iloc[:split_idx]
    y_blend_val = y_blend.iloc[split_idx:]
    
    print(f"\n{'='*60}")
    print("TIME-SERIES SPLIT")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_regime_train)} ({len(X_regime_train)/len(X_regime)*100:.1f}%)")
    print(f"Validation samples: {len(X_regime_val)} ({len(X_regime_val)/len(X_regime)*100:.1f}%)")
    
    # 6.5. Initialize and fit feature scaler on training data only
    print("\n" + "="*60)
    print("FEATURE SCALING")
    print("="*60)
    # Use ZScoreScaler by default (can be changed to MinMaxScaler or RobustScaler)
    scaler = ZScoreScaler(use_gpu=use_cuml)
    print(f"Initialized {scaler.__class__.__name__}")
    
    print("Fitting scaler on training data (X_regime_train)...")
    # Fit on X_regime_train (use this as the reference feature set)
    scaler.fit(X_regime_train)
    
    # Transform both train and validation sets for regime detector
    X_regime_train_scaled = scaler.transform(X_regime_train)
    X_regime_val_scaled = scaler.transform(X_regime_val)
    
    # For X_blend, scale only the feature columns (same as X_regime)
    # Keep signal and regime columns unscaled
    feature_cols = X_regime_train.columns
    signal_regime_cols = [col for col in X_blend_train.columns if col not in feature_cols]
    
    # Scale feature columns only
    X_blend_train_feat_scaled = scaler.transform(X_blend_train[feature_cols])
    X_blend_val_feat_scaled = scaler.transform(X_blend_val[feature_cols])
    
    # Reconstruct X_blend with scaled features + unscaled signals/regime
    X_blend_train_scaled = X_blend_train_feat_scaled.copy()
    X_blend_val_scaled = X_blend_val_feat_scaled.copy()
    for col in signal_regime_cols:
        X_blend_train_scaled[col] = X_blend_train[col]
        X_blend_val_scaled[col] = X_blend_val[col]
    
    print("Feature scaling completed")
    
    # Save the scaler
    scaler_path = models_dir / "feature_scaler.pkl"
    scaler.save(str(scaler_path))
    print(f"Scaler saved to {scaler_path}")
    
    # 7. Train RegimeDetector on training set only (using scaled features)
    # GPU: Model will automatically convert pandas → cuDF for GPU training if cuML available
    print("\n" + "="*60)
    print("TRAINING REGIME DETECTOR")
    print("="*60)
    print(f"Training on {len(X_regime_train)} samples...")
    regime_detector = RegimeDetector(use_gpu=use_cuml, random_state=args.seed)  # GPU-accelerated training
    regime_detector.fit(X_regime_train_scaled, y_regime_train)  # Train on scaled training set only
    
    # Validate on validation set
    print(f"\nValidating on {len(X_regime_val)} samples...")
    regime_pred_val = regime_detector.predict(X_regime_val_scaled)
    regime_val_accuracy = accuracy_score(y_regime_val, regime_pred_val)
    
    print(f"\n[RegimeDetector] Validation accuracy: {regime_val_accuracy:.4f}")
    print(f"\n[RegimeDetector] Validation confusion matrix:")
    regime_cm = confusion_matrix(y_regime_val, regime_pred_val, labels=regime_detector.regime_classes)
    regime_cm_df = pd.DataFrame(
        regime_cm,
        index=regime_detector.regime_classes,
        columns=regime_detector.regime_classes
    )
    print(regime_cm_df)

    regime_cm_path = results_dir / "confusion_regime.csv"
    regime_cm_df.to_csv(regime_cm_path)
    
    print(f"\n[RegimeDetector] Validation label distribution (true):")
    for label, count in y_regime_val.value_counts().items():
        pct = count / len(y_regime_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\n[RegimeDetector] Validation label distribution (predicted):")
    for label, count in regime_pred_val.value_counts().items():
        pct = count / len(regime_pred_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    regime_detector.save(str(models_dir / "regime_model.pkl"))
    print(f"\nRegimeDetector saved to {models_dir / 'regime_model.pkl'}")
    
    # 8. Train SignalBlender on training set only
    # GPU: Model will automatically convert pandas → cuDF for GPU training if cuML available
    print("\n" + "="*60)
    print("TRAINING SIGNAL BLENDER")
    print("="*60)
    
    if args.use_full_pipeline:
        # Use full pipeline with pruning and diagnostics
        signal_blender, signal_diagnostics = train_full_pipeline(
            X_train=X_blend_train_scaled,
            y_train=y_blend_train,
            X_val=X_blend_val_scaled,
            y_val=y_blend_val,
            model_class=SignalBlender,
            model_name="SignalBlender",
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            models_dir=models_dir,
            results_dir=results_dir,
            use_gpu=use_cuml,
            random_state=args.seed
        )
    else:
        # Legacy training path
        print(f"Training on {len(X_blend_train)} samples...")
        signal_blender = SignalBlender(use_gpu=use_cuml, random_state=args.seed)  # GPU-accelerated training
        signal_blender.fit(
            X_blend_train_scaled, 
            y_blend_train,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            X_val=X_blend_val_scaled,
            y_val=y_blend_val,
            disable_calibration=args.disable_calibration
        )
    
    # Analyze probability distribution before calibration
    if calibration_method:
        print(f"\nAnalyzing probability distribution (before calibration)...")
        try:
            # Get raw probabilities directly from model (before calibration)
            if signal_blender.use_gpu:
                try:
                    import cudf
                    X_gpu = cudf.from_pandas(X_blend_val_scaled)
                    proba_raw = signal_blender.model.predict_proba(X_gpu).to_pandas().values
                except ImportError:
                    proba_raw = signal_blender.model.predict_proba(X_blend_val_scaled)
            else:
                proba_raw = signal_blender.model.predict_proba(X_blend_val_scaled)
            max_proba_raw = proba_raw.max(axis=1)
            _print_probability_distribution(max_proba_raw, "Raw probabilities")
        except Exception as e:
            print(f"  Could not analyze raw probabilities: {e}")
    
    # Validate on validation set
    print(f"\nValidating on {len(X_blend_val)} samples...")
    blend_proba_val = signal_blender.predict_proba(X_blend_val_scaled)
    blend_pred_val = blend_proba_val.idxmax(axis=1).astype(int)
    filtered_pct = 0.0
    if args.min_trade_proba > 0:
        max_proba = blend_proba_val.max(axis=1)
        low_conf_mask = max_proba < args.min_trade_proba
        blend_pred_val = blend_pred_val.where(~low_conf_mask, 0)
        filtered_pct = float(low_conf_mask.mean()) if len(low_conf_mask) else 0.0
        print(f"[SignalBlender] Confidence filter: min_proba={args.min_trade_proba:.2f}, filtered={filtered_pct:.2%}")
    blend_val_accuracy = accuracy_score(y_blend_val, blend_pred_val)
    
    # Analyze probability distribution after calibration
    if calibration_method:
        print(f"\nAnalyzing probability distribution (after calibration)...")
        max_proba_cal = blend_proba_val.max(axis=1)
        _print_probability_distribution(max_proba_cal, "Calibrated probabilities")
    
    print(f"\n[SignalBlender] Validation accuracy: {blend_val_accuracy:.4f}")
    print(f"\n[SignalBlender] Validation confusion matrix:")
    blend_cm = confusion_matrix(y_blend_val, blend_pred_val, labels=signal_blender.signal_classes)
    blend_cm_df = pd.DataFrame(
        blend_cm,
        index=signal_blender.signal_classes,
        columns=signal_blender.signal_classes
    )
    print(blend_cm_df)

    blend_cm_path = results_dir / "confusion_signal.csv"
    blend_cm_df.to_csv(blend_cm_path)
    
    print(f"\n[SignalBlender] Validation label distribution (true):")
    for label, count in y_blend_val.value_counts().items():
        pct = count / len(y_blend_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\n[SignalBlender] Validation label distribution (predicted):")
    for label, count in blend_pred_val.value_counts().items():
        pct = count / len(blend_pred_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")

    signal_return_metrics = compute_return_metrics(
        blend_pred_val,
        df['close'].reindex(blend_pred_val.index),
        horizon_bars=HORIZON_BARS,
        eval_mode=args.eval_mode,
        transaction_cost_bps=args.fee_bps
    )
    print(f"\n[SignalBlender] Return metrics (validation):")
    print(f"  Eval mode: {signal_return_metrics['eval_mode']}")
    print(f"  Fee (bps): {signal_return_metrics['transaction_cost_bps']:.2f}")
    print(f"  Total cost: {signal_return_metrics['total_cost']:.6f}")
    print(f"  Net cumulative return: {signal_return_metrics['cumulative_return']:.4f}")
    print(f"  Gross cumulative return: {signal_return_metrics['gross_cumulative_return']:.4f}")
    print(f"  Mean return: {signal_return_metrics['mean_return']:.6f}")
    print(f"  Volatility: {signal_return_metrics['volatility']:.6f}")
    print(f"  Sharpe-like: {signal_return_metrics['sharpe_like']:.4f}")
    print(f"  Trade rate: {signal_return_metrics['trade_rate']:.2%}")
    print(f"  Win rate: {signal_return_metrics['win_rate']:.2%}")
    print(f"  Turnover: {signal_return_metrics['turnover']:.4f}")
    print(f"  Profit factor: {_fmt_optional(signal_return_metrics['profit_factor'])}")
    print(f"  Expectancy: {_fmt_optional(signal_return_metrics['expectancy'], 6)}")
    print(f"  Max drawdown: {signal_return_metrics['max_drawdown']:.4f}")
    print(f"  Max DD duration: {signal_return_metrics['max_drawdown_duration']}")
    
    # Save both regular and calibrated models
    signal_blender.save(str(models_dir / "blender_model.pkl"))
    print(f"\nSignalBlender saved to {models_dir / 'blender_model.pkl'}")
    
    # Always save calibrated version (calibration is enabled by default)
    if calibration_method:
        signal_blender.save(str(models_dir / "blender_calibrated.pkl"))
        print(f"SignalBlender (calibrated) saved to {models_dir / 'blender_calibrated.pkl'}")
    
    # 8.5. Train DirectionBlender (binary direction classifier on trade samples only)
    print("\n" + "="*60)
    print("TRAINING DIRECTION BLENDER")
    print("="*60)
    
    # Filter to only samples where y_blend != 0 (trade samples)
    mask_trade_train = y_blend_train != 0
    mask_trade_val = y_blend_val != 0
    
    X_blend_trade_train = X_blend_train_scaled[mask_trade_train]
    X_blend_trade_val = X_blend_val_scaled[mask_trade_val]
    y_blend_trade_train = y_blend_train[mask_trade_train]
    y_blend_trade_val = y_blend_val[mask_trade_val]
    
    # Create binary direction labels: 1 for long, -1 for short
    y_direction_train = pd.Series(
        np.where(y_blend_trade_train > 0, 1, -1),
        index=y_blend_trade_train.index,
        dtype=int,
        name='direction'
    )
    y_direction_val = pd.Series(
        np.where(y_blend_trade_val > 0, 1, -1),
        index=y_blend_trade_val.index,
        dtype=int,
        name='direction'
    )
    
    print(f"Trade samples (training): {len(X_blend_trade_train)} ({len(X_blend_trade_train)/len(X_blend_train)*100:.1f}% of training set)")
    print(f"Trade samples (validation): {len(X_blend_trade_val)} ({len(X_blend_trade_val)/len(X_blend_val)*100:.1f}% of validation set)")
    print(f"Total direction samples: {len(y_direction_train) + len(y_direction_val)}")
    
    # Direction label distribution
    print(f"\nDirection label distribution (training):")
    for label, count in y_direction_train.value_counts().items():
        pct = count / len(y_direction_train) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\nDirection label distribution (validation):")
    for label, count in y_direction_val.value_counts().items():
        pct = count / len(y_direction_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    # Train DirectionBlender
    if args.use_full_pipeline:
        # Use full pipeline with pruning and diagnostics
        direction_blender, direction_diagnostics = train_full_pipeline(
            X_train=X_blend_trade_train,
            y_train=y_direction_train,
            X_val=X_blend_trade_val,
            y_val=y_direction_val,
            model_class=DirectionBlender,
            model_name="DirectionBlender",
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            models_dir=models_dir,
            results_dir=results_dir,
            use_gpu=use_cuml,
            random_state=args.seed
        )
    else:
        # Legacy training path
        print(f"\nTraining DirectionBlender on {len(X_blend_trade_train)} trade samples...")
        direction_blender = DirectionBlender(use_gpu=use_cuml, random_state=args.seed)  # GPU-accelerated training
        direction_blender.fit(
            X_blend_trade_train, 
            y_direction_train,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            X_val=X_blend_trade_val,
            y_val=y_direction_val,
            disable_calibration=args.disable_calibration
        )
    
    # Analyze probability distribution before calibration
    if calibration_method:
        print(f"\nAnalyzing probability distribution (before calibration)...")
        try:
            # Get raw probabilities directly from model (before calibration)
            if direction_blender.use_gpu:
                try:
                    import cudf
                    X_gpu = cudf.from_pandas(X_blend_trade_val)
                    proba_raw = direction_blender.model.predict_proba(X_gpu).to_pandas().values
                except ImportError:
                    proba_raw = direction_blender.model.predict_proba(X_blend_trade_val)
            else:
                proba_raw = direction_blender.model.predict_proba(X_blend_trade_val)
            max_proba_raw = proba_raw.max(axis=1)
            _print_probability_distribution(max_proba_raw, "Raw probabilities")
        except Exception as e:
            print(f"  Could not analyze raw probabilities: {e}")
    
    # Validate on validation set
    print(f"\nValidating on {len(X_blend_trade_val)} trade samples...")
    direction_pred_val = direction_blender.predict(X_blend_trade_val)
    direction_val_accuracy = accuracy_score(y_direction_val, direction_pred_val)
    
    # Analyze probability distribution after calibration
    if calibration_method:
        print(f"\nAnalyzing probability distribution (after calibration)...")
        proba_cal = direction_blender.predict_proba(X_blend_trade_val)
        max_proba_cal = proba_cal.max(axis=1)
        _print_probability_distribution(max_proba_cal, "Calibrated probabilities")
    
    print(f"\n[DirectionBlender] Validation accuracy: {direction_val_accuracy:.4f}")
    print(f"\n[DirectionBlender] Validation confusion matrix:")
    direction_cm = confusion_matrix(y_direction_val, direction_pred_val, labels=direction_blender.direction_classes)
    direction_cm_df = pd.DataFrame(
        direction_cm,
        index=direction_blender.direction_classes,
        columns=direction_blender.direction_classes
    )
    print(direction_cm_df)

    direction_cm_path = results_dir / "confusion_direction.csv"
    direction_cm_df.to_csv(direction_cm_path)
    
    print(f"\n[DirectionBlender] Validation label distribution (true):")
    for label, count in y_direction_val.value_counts().items():
        pct = count / len(y_direction_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\n[DirectionBlender] Validation label distribution (predicted):")
    for label, count in direction_pred_val.value_counts().items():
        pct = count / len(direction_pred_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")

    direction_return_metrics = compute_return_metrics(
        direction_pred_val,
        df['close'].reindex(direction_pred_val.index),
        horizon_bars=HORIZON_BARS,
        eval_mode=args.eval_mode,
        transaction_cost_bps=args.fee_bps
    )
    print(f"\n[DirectionBlender] Return metrics (validation):")
    print(f"  Eval mode: {direction_return_metrics['eval_mode']}")
    print(f"  Fee (bps): {direction_return_metrics['transaction_cost_bps']:.2f}")
    print(f"  Total cost: {direction_return_metrics['total_cost']:.6f}")
    print(f"  Net cumulative return: {direction_return_metrics['cumulative_return']:.4f}")
    print(f"  Gross cumulative return: {direction_return_metrics['gross_cumulative_return']:.4f}")
    print(f"  Mean return: {direction_return_metrics['mean_return']:.6f}")
    print(f"  Volatility: {direction_return_metrics['volatility']:.6f}")
    print(f"  Sharpe-like: {direction_return_metrics['sharpe_like']:.4f}")
    print(f"  Trade rate: {direction_return_metrics['trade_rate']:.2%}")
    print(f"  Win rate: {direction_return_metrics['win_rate']:.2%}")
    print(f"  Turnover: {direction_return_metrics['turnover']:.4f}")
    print(f"  Profit factor: {_fmt_optional(direction_return_metrics['profit_factor'])}")
    print(f"  Expectancy: {_fmt_optional(direction_return_metrics['expectancy'], 6)}")
    print(f"  Max drawdown: {direction_return_metrics['max_drawdown']:.4f}")
    print(f"  Max DD duration: {direction_return_metrics['max_drawdown_duration']}")

    metrics_payload = {
        "regime_accuracy": float(regime_val_accuracy),
        "signal_accuracy": float(blend_val_accuracy),
        "direction_accuracy": float(direction_val_accuracy),
        "signal_confidence_threshold": float(args.min_trade_proba),
        "signal_confidence_filtered_pct": float(filtered_pct),
        "signal_return_metrics": signal_return_metrics,
        "direction_return_metrics": direction_return_metrics,
        "confusion_matrices": {
            "regime": regime_cm_path,
            "signal": blend_cm_path,
            "direction": direction_cm_path
        }
    }
    metrics_path = results_dir / "validation_metrics.json"
    _write_json(metrics_path, metrics_payload)
    print(f"\nValidation metrics saved to {metrics_path}")
    
    # Save both regular and calibrated models
    direction_blender.save(str(models_dir / "blender_direction_model.pkl"))
    print(f"\nDirectionBlender saved to {models_dir / 'blender_direction_model.pkl'}")
    
    # Always save calibrated version (calibration is enabled by default)
    if calibration_method:
        direction_blender.save(str(models_dir / "blender_direction_calibrated.pkl"))
        print(f"DirectionBlender (calibrated) saved to {models_dir / 'blender_direction_calibrated.pkl'}")
    
    # 8.6. Train Regime-Specific Signal Blenders
    print("\n" + "="*60)
    print("TRAINING REGIME-SPECIFIC SIGNAL BLENDERS")
    print("="*60)
    
    def _regime_to_slug(name: str) -> str:
        """Convert regime name to slug for filenames."""
        return name.lower().replace(" ", "_")
    
    regime_classes = sorted(pd.unique(y_regime_train))
    print(f"Regime classes detected: {regime_classes}")
    
    for regime_name in regime_classes:
        slug = _regime_to_slug(regime_name)
        print("\n" + "-"*60)
        print(f"Training regime-specific SignalBlender for '{regime_name}'")
        print("-"*60)
        
        mask_train = (y_regime_train == regime_name)
        mask_val = (y_regime_val == regime_name)
        
        train_count = int(mask_train.sum())
        val_count = int(mask_val.sum())
        
        print(f"Samples → Train: {train_count}, Val: {val_count}")
        
        if train_count < 5000 or val_count < 1000:
            print(f"❌ Skipping regime '{regime_name}' (not enough samples)")
            continue
        
        Xb_train_r = X_blend_train_scaled[mask_train]
        yb_train_r = y_blend_train[mask_train]
        
        Xb_val_r = X_blend_val_scaled[mask_val]
        yb_val_r = y_blend_val[mask_val]
        
        # Create model with custom parameters for regime-specific training
        model_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'n_bins': 128,
            'split_criterion': 0,
            'bootstrap': True,
            'max_samples': 0.8,
            'max_features': 0.9,
            'n_streams': 4
        }
        model_r = SignalBlender(model_params=model_params, use_gpu=use_cuml, random_state=args.seed)
        
        print(f"Training SignalBlender[{regime_name}]...")
        model_r.fit(
            Xb_train_r, 
            yb_train_r,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            X_val=Xb_val_r,
            y_val=yb_val_r,
            disable_calibration=args.disable_calibration
        )
        
        preds = model_r.predict(Xb_val_r)
        acc = accuracy_score(yb_val_r, preds)
        
        print(f"[SignalBlender:{regime_name}] Validation Accuracy: {acc:.4f}")
        print(f"[SignalBlender:{regime_name}] Validation confusion matrix:")
        regime_cm = confusion_matrix(yb_val_r, preds, labels=model_r.signal_classes)
        regime_cm_df = pd.DataFrame(
            regime_cm,
            index=model_r.signal_classes,
            columns=model_r.signal_classes
        )
        print(regime_cm_df)
        
        out_path = models_dir / f"blender_{slug}.pkl"
        model_r.save(str(out_path))
        print(f"Saved: {out_path}")
        
        if calibration_method:
            out_path_cal = models_dir / f"blender_{slug}_calibrated.pkl"
            model_r.save(str(out_path_cal))
            print(f"Saved (calibrated): {out_path_cal}")
    
    # 8.7. Train Regime-Specific Direction Blenders
    print("\n" + "="*60)
    print("TRAINING REGIME-SPECIFIC DIRECTION BLENDERS")
    print("="*60)
    
    trade_train = (y_blend_train != 0)
    trade_val = (y_blend_val != 0)
    
    for regime_name in regime_classes:
        slug = _regime_to_slug(regime_name)
        print("\n" + "-"*60)
        print(f"Training regime-specific DirectionBlender for '{regime_name}'")
        print("-"*60)
        
        mask_train = trade_train & (y_regime_train == regime_name)
        mask_val = trade_val & (y_regime_val == regime_name)
        
        train_count = int(mask_train.sum())
        val_count = int(mask_val.sum())
        
        print(f"Trade Samples → Train: {train_count}, Val: {val_count}")
        
        if train_count < 3000 or val_count < 800:
            print(f"❌ Skipping DirectionBlender for '{regime_name}' (too few trades)")
            continue
        
        Xd_train_r = X_blend_train_scaled[mask_train]
        Xd_val_r = X_blend_val_scaled[mask_val]
        
        y_dir_train = pd.Series(
            np.where(y_blend_train[mask_train] > 0, 1, -1),
            index=y_blend_train[mask_train].index,
            dtype=int,
            name='direction'
        )
        y_dir_val = pd.Series(
            np.where(y_blend_val[mask_val] > 0, 1, -1),
            index=y_blend_val[mask_val].index,
            dtype=int,
            name='direction'
        )
        
        # Create model with custom parameters for regime-specific training
        model_params = {
            'n_estimators': 300,
            'max_depth': 10,
            'n_bins': 128,
            'split_criterion': 0,
            'bootstrap': True,
            'max_samples': 0.8,
            'max_features': 0.9,
            'n_streams': 4
        }
        model_r = DirectionBlender(model_params=model_params, use_gpu=use_cuml, random_state=args.seed)
        
        print(f"Training DirectionBlender[{regime_name}]...")
        model_r.fit(
            Xd_train_r, 
            y_dir_train,
            calibration_method=calibration_method,
            sharpening_alpha=sharpening_alpha,
            X_val=Xd_val_r,
            y_val=y_dir_val,
            disable_calibration=args.disable_calibration
        )
        
        preds = model_r.predict(Xd_val_r)
        acc = accuracy_score(y_dir_val, preds)
        
        print(f"[DirectionBlender:{regime_name}] Validation Accuracy: {acc:.4f}")
        print(f"[DirectionBlender:{regime_name}] Validation confusion matrix:")
        direction_cm = confusion_matrix(y_dir_val, preds, labels=model_r.direction_classes)
        direction_cm_df = pd.DataFrame(
            direction_cm,
            index=model_r.direction_classes,
            columns=model_r.direction_classes
        )
        print(direction_cm_df)
        
        out_path = models_dir / f"blender_direction_{slug}.pkl"
        model_r.save(str(out_path))
        print(f"Saved: {out_path}")
        
        if calibration_method:
            out_path_cal = models_dir / f"blender_direction_{slug}_calibrated.pkl"
            model_r.save(str(out_path_cal))
            print(f"Saved (calibrated): {out_path_cal}")
    
    # 9. Print final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\nRegime Detector:")
    print(f"  Validation Accuracy: {regime_val_accuracy:.4f}")
    print(f"  Training samples: {len(X_regime_train)}")
    print(f"  Validation samples: {len(X_regime_val)}")
    
    print(f"\nSignal Blender:")
    print(f"  Validation Accuracy: {blend_val_accuracy:.4f}")
    print(f"  Training samples: {len(X_blend_train)}")
    print(f"  Validation samples: {len(X_blend_val)}")
    
    print(f"\nDirection Blender:")
    print(f"  Validation Accuracy: {direction_val_accuracy:.4f}")
    print(f"  Training samples: {len(X_blend_trade_train)}")
    print(f"  Validation samples: {len(X_blend_trade_val)}")
    
    print("\nTraining complete!")
    print(f"Models saved to {models_dir}/")
    print(f"  - regime_model.pkl")
    print(f"  - blender_model.pkl")
    print(f"  - blender_direction_model.pkl")
    
    # List regime-specific models that were created
    regime_models_signal = []
    regime_models_direction = []
    for regime_name in regime_classes:
        slug = _regime_to_slug(regime_name)
        if (models_dir / f"blender_{slug}.pkl").exists():
            regime_models_signal.append(f"blender_{slug}.pkl")
        if (models_dir / f"blender_direction_{slug}.pkl").exists():
            regime_models_direction.append(f"blender_direction_{slug}.pkl")
    
    if regime_models_signal:
        print(f"\nRegime-specific SignalBlenders ({len(regime_models_signal)}):")
        for model in regime_models_signal:
            print(f"  - {model}")
    
    if regime_models_direction:
        print(f"\nRegime-specific DirectionBlenders ({len(regime_models_direction)}):")
        for model in regime_models_direction:
            print(f"  - {model}")

    # Save training manifest (always, regardless of regime-specific models)
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_type": "static",
        "seed": args.seed,
        "use_gpu": use_cuml,
        "data_path": data_path,
        "feature_cache_path": cache_path,
        "log_path": log_path,
        "calibration": {
            "method": calibration_method,
            "sharpening_alpha": sharpening_alpha,
            "disabled": args.disable_calibration
        },
        "evaluation": {
            "eval_mode": args.eval_mode,
            "transaction_cost_bps": args.fee_bps
        },
        "ablation": {
            "disable_ml_features": args.disable_ml_features,
            "disable_regime_context": args.disable_regime_context,
            "disable_signal_dynamics": args.disable_signal_dynamics,
            "disable_rolling_stats": args.disable_rolling_stats,
            "disable_modules": disable_modules_list,
            "include_modules": include_modules_list,
            "include_features": include_features_list,
            "exclude_features": exclude_features_list,
        },
        "label_config": {
            "horizon_bars": HORIZON_BARS,
            "label_threshold": LABEL_THRESHOLD,
            "smoothing_window": SMOOTHING_WINDOW
        },
        "regime_label_strategy": regime_label_strategy,
        "modules": {
            "signal_modules": [m.name for m in feature_store.signal_modules],
            "context_modules": [m.name for m in feature_store.context_modules]
        },
        "paths": {
            "feature_columns": feature_columns_path,
            "blend_feature_columns": blend_feature_columns_path,
            "scaler": scaler_path,
            "metrics": metrics_path,
            "confusion_regime": regime_cm_path,
            "confusion_signal": blend_cm_path,
            "confusion_direction": direction_cm_path
        },
        "model_paths": {
            "regime_model": models_dir / "regime_model.pkl",
            "signal_blender": models_dir / "blender_model.pkl",
            "signal_blender_calibrated": (models_dir / "blender_calibrated.pkl") if calibration_method else None,
            "direction_blender": models_dir / "blender_direction_model.pkl",
            "direction_blender_calibrated": (models_dir / "blender_direction_calibrated.pkl") if calibration_method else None,
            "regime_specific_signal": [models_dir / name for name in regime_models_signal],
            "regime_specific_direction": [models_dir / name for name in regime_models_direction]
        }
    }
    manifest_path = models_dir / "training_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"\nTraining manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
