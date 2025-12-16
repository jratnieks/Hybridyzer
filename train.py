# train.py
"""
Training script for regime detector and signal blender models.
Supports walk-forward training with 6-month training windows and 1-month validation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import timedelta
import argparse
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
    print(f"    Mean: {proba.mean():.4f}, Median: {proba.median():.4f}")
    print(f"    Distribution by bin:")
    for i in range(len(bins) - 1):
        pct = counts[i] / total * 100 if total > 0 else 0
        print(f"      [{bins[i]:.1f}, {bins[i+1]:.1f}): {counts[i]:6d} ({pct:5.2f}%)")
from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext
from core.feature_store import FeatureStore
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


def train_one_window(
    df_train: pd.DataFrame,
    signal_modules: List,
    context_modules: List,
    feats_train: pd.DataFrame,
    horizon_bars: int = 48,
    label_threshold: float = 0.0005
) -> Tuple[RegimeDetector, SignalBlender, Dict]:
    """
    Train regime detector and signal blender on a training window.
    
    Args:
        df_train: Training data window
        signal_modules: List of signal modules
        context_modules: List of context modules
        horizon_bars: Forward horizon in bars for direction labels
        label_threshold: Threshold for direction labels
        
    Returns:
        Tuple of (regime_detector, signal_blender, training_stats)
    """
    # Enforce index integrity between raw data and cached features to avoid
    # silent misalignment after cleaning/warmups.
    assert feats_train.index.equals(df_train.index), "Feature/price index mismatch in training window"

    # Generate labels
    regime_y = rule_based_regime(feats_train, df_train)
    # Use new direction labels
    df_labeled = make_direction_labels(
        df_train,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
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
    reg = RegimeDetector()
    reg.fit(X_regime, regime_y)
    
    blender = SignalBlender(return_threshold=0.001)
    blender.fit(X_blend, blend_y)
    
    # Training statistics
    stats = {
        'train_samples': len(X_regime),
        'regime_dist': regime_y.value_counts().to_dict(),
        'blend_dist': blend_y.value_counts().to_dict(),
        'feature_count': len(feats_train.columns)
    }
    
    return reg, blender, stats


def validate_one_window(
    df_val: pd.DataFrame,
    reg_model: RegimeDetector,
    blender_model: SignalBlender,
    signal_modules: List,
    context_modules: List,
    feats_val: pd.DataFrame,
    horizon_bars: int = 48,
    label_threshold: float = 0.0005
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
        
    Returns:
        Dictionary with validation metrics
    """
    # Enforce index integrity between raw data and cached features to avoid
    # silent misalignment after cleaning/warmups.
    assert feats_val.index.equals(df_val.index), "Feature/price index mismatch in validation window"

    # Generate true labels (use new direction labels for consistency)
    regime_y_true = rule_based_regime(feats_val, df_val)
    df_labeled = make_direction_labels(
        df_val,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
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
    
    # Calculate returns for signal accuracy
    actual_returns = future_return(df_val['close'], horizon=horizon_bars).reindex(blend_y_true.index)
    signal_returns = blend_y_pred * actual_returns
    cumulative_return = (1 + signal_returns).prod() - 1
    
    metrics = {
        'val_samples': len(X_regime),
        'regime_accuracy': float(regime_accuracy),
        'blend_accuracy': float(blend_accuracy),
        'cumulative_return': float(cumulative_return),
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
) -> Tuple[FeatureStore, pd.DataFrame]:
    """
    Compute all features exactly once, cache to Parquet, and reuse for slicing.

    This centralizes the expensive feature computation step and ensures the
    walk-forward loop only slices cached features instead of rebuilding per
    window.
    """
    feature_store = FeatureStore(use_gpu=use_gpu)
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

    Labels always derive from the raw df slice; models see the cached features
    on the exact same index. Any divergence is a correctness bug and must fail
    fast.
    """
    feats_window = features[(features.index >= start_ts) & (features.index < end_ts)]
    df_window = df.loc[feats_window.index]

    if not feats_window.index.equals(df_window.index):
        raise ValueError("Cached feature index diverged from raw data during slicing")

    return df_window, feats_window


def prepare_windows(
    df: pd.DataFrame,
    train_months: int,
    val_months: int,
    slide_months: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Lightweight wrapper to document walk-forward split generation."""
    return generate_splits(df, train_months, val_months, slide_months)


def train_models_for_window(
    df_window: pd.DataFrame,
    feats_window: pd.DataFrame,
    signal_modules: List,
    context_modules: List,
    horizon_bars: int,
    label_threshold: float,
) -> Tuple[RegimeDetector, SignalBlender, Dict]:
    """Thin wrapper to make the training stage signature explicit."""
    return train_one_window(
        df_window,
        signal_modules,
        context_modules,
        feats_window,
        horizon_bars=horizon_bars,
        label_threshold=label_threshold,
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
    horizon_bars: int = 48,
    label_threshold: float = 0.0005
) -> Tuple[RegimeDetector, SignalBlender]:
    """
    Perform walk-forward training across all windows.
    
    Args:
        df: Full dataset
        feature_store: Shared FeatureStore instance for feature caching/slicing
        signal_modules: List of signal modules
        context_modules: List of context modules
        models_dir: Directory to save models
        train_months: Training window size in months
        val_months: Validation window size in months
        slide_months: Slide forward by this many months
        horizon_bars: Forward horizon in bars for direction labels
        label_threshold: Threshold for direction labels
        
    Returns:
        Tuple of (final_regime_detector, final_signal_blender)
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
    )

    # Generate splits
    splits = prepare_windows(df, train_months, val_months, slide_months)
    
    if len(splits) == 0:
        raise ValueError("No valid splits generated. Check data range and window sizes.")
    
    print(f"\nGenerated {len(splits)} walk-forward splits")
    print(f"Training window: {train_months} months, Validation window: {val_months} months, Slide: {slide_months} months\n")
    
    # Store results
    all_results = []
    best_val_return = -np.inf
    best_reg = None
    best_blender = None
    
    # Walk-forward training
    for i, (train_start, train_end, val_start, val_end) in enumerate(splits, 1):
        print(f"\n{'='*60}")
        print(f"Window {i}/{len(splits)}")
        print(f"Train: {train_start.date()} to {train_end.date()}")
        print(f"Val:   {val_start.date()} to {val_end.date()}")
        print(f"{'='*60}")
        
        # Extract windows using cached feature index to prevent any recomputation
        try:
            df_train, feats_train = slice_window(df, cached_features, train_start, train_end)
            df_val, feats_val = slice_window(df, cached_features, val_start, val_end)
        except ValueError as e:
            print(f"Warning: Skipping window {i} due to alignment error: {e}")
            continue

        if len(df_train) == 0 or len(df_val) == 0:
            print(f"Warning: Skipping window {i} - insufficient data")
            continue

        # Train on window using cached features to prevent repeated computation
        print("Training models...")
        try:
            reg, blender, train_stats = train_models_for_window(
                df_train,
                feats_train,
                signal_modules,
                context_modules,
                horizon_bars,
                label_threshold,
            )
            print(f"  Training samples: {train_stats['train_samples']}")
            print(f"  Features: {train_stats['feature_count']}")
        except Exception as e:
            print(f"  Error during training: {e}")
            continue

        # Validate on window
        print("Validating models...")
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
            )
            print(f"  Validation samples: {val_metrics['val_samples']}")
            print(f"  Regime accuracy: {val_metrics['regime_accuracy']:.4f}")
            print(f"  Blend accuracy: {val_metrics['blend_accuracy']:.4f}")
            print(f"  Cumulative return: {val_metrics['cumulative_return']:.4f}")

            # Track best model
            if val_metrics['cumulative_return'] > best_val_return:
                best_val_return = val_metrics['cumulative_return']
                best_reg = reg
                best_blender = blender
                print(f"  *** New best model (return: {best_val_return:.4f}) ***")
        except Exception as e:
            print(f"  Error during validation: {e}")
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
    
    # Summary
    print(f"\n{'='*60}")
    print("WALK-FORWARD TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if len(all_results) > 0:
        avg_regime_acc = np.mean([r['regime_accuracy'] for r in all_results])
        avg_blend_acc = np.mean([r['blend_accuracy'] for r in all_results])
        avg_return = np.mean([r['cumulative_return'] for r in all_results])
        
        print(f"Windows processed: {len(all_results)}")
        print(f"Average regime accuracy: {avg_regime_acc:.4f}")
        print(f"Average blend accuracy: {avg_blend_acc:.4f}")
        print(f"Average cumulative return: {avg_return:.4f}")
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
            best_reg, best_blender, _ = train_models_for_window(
                df_train,
                feats_last,
                signal_modules,
                context_modules,
                horizon_bars,
                label_threshold,
            )
        else:
            raise ValueError("Cannot create final models - no successful training")
    
    return best_reg, best_blender


def downsample_validation_for_calibration(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_size: int = 50000
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Downsample validation set for calibration to avoid GPU OOM.
    Uses stratified sampling based on labels.
    
    Args:
        X_val: Validation features
        y_val: Validation labels
        target_size: Target number of samples (default: 50000)
        
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
        random_state=42,
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
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
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
    X_val_cal, y_val_cal = downsample_validation_for_calibration(X_val, y_val, target_size=50000)
    print(f"[{model_name}] Using {len(X_val_cal)} samples for calibration (downsampled from {len(X_val)}).")
    
    # 2. Initial training
    print(f"\n[2/7] Initial training...")
    model = model_class(use_gpu=use_gpu)
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
            X_val_pruned, y_val, target_size=50000
        )
        print(f"[{model_name}] Using {len(X_val_pruned_cal)} samples for calibration (downsampled from {len(X_val_pruned)}).")
        
        model_pruned = model_class(use_gpu=use_gpu)
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
    parser.add_argument('--horizon-bars', type=int, default=48,
                        help='Forward horizon in bars for label generation (default: 48 = 4 hours for 5-min data)')
    parser.add_argument('--label-threshold', type=float, default=0.0005,
                        help='Threshold for direction labels (default: 0.0005 = 0.05%%)')
    parser.add_argument('--runpod', action='store_true',
                        help='Use RunPod workspace layout (/workspace/Hybridyzer) for data, models, and results')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU even if cuML is installed')
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
    
    # New label system configuration
    HORIZON_BARS = args.horizon_bars  # Forward horizon in bars (default: 48)
    LABEL_THRESHOLD = args.label_threshold  # Threshold for direction labels (default: 0.0005)
    SMOOTHING_WINDOW = 12  # Rolling window for smoothing returns
    
    # 1. Load BTC data with load_btc_csv
    print("Loading BTC CSV data...")
    # Prefer 4H data for faster experimentation, then fall back to 1m
    csv_paths = [
        data_dir / 'btcusd_4H.csv',           # 4-hour data (small, fast)
        data_dir / 'btcusd_1min.csv',         # 1-minute data (large)
        data_dir / 'btc_1m.csv',
        data_dir / 'btcusd_1min_volspikes.csv',
        data_dir / 'btc_data.csv',
    ]
    df = None
    for path in csv_paths:
        try:
            df = load_btc_csv(str(path))
            print(f"Loaded from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        # Try to find any CSV in data directory
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            df = load_btc_csv(str(csv_files[0]))
            print(f"Loaded from: {csv_files[0]}")
        else:
            raise FileNotFoundError("No CSV files found in data/ directory")
    
    if df.empty:
        print("Error: No data loaded.")
        return
    
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # 2. Build or load cached features once to avoid recomputation across runs
    cache_path = models_dir / "cached_features.parquet"
    feature_store = FeatureStore()
    feature_store, feats = build_features_once(
        df,
        signal_modules=feature_store.signal_modules,
        context_modules=feature_store.context_modules,
        cache_path=cache_path,
        use_gpu=False,
    )
    print(f"[Features] Loaded cached features: {feats.shape}")
    
    # Print final feature columns for verification
    print("\nFINAL FEATURE COLUMNS:", feats.columns.tolist())
    
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
    y_regime = make_regime_labels(df)
    y_regime = y_regime.reindex(feats.index)
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
    regime_detector = RegimeDetector(use_gpu=use_cuml)  # GPU-accelerated training
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
                use_gpu=use_cuml
            )
    else:
        # Legacy training path
        print(f"Training on {len(X_blend_train)} samples...")
        signal_blender = SignalBlender(use_gpu=use_cuml)  # GPU-accelerated training
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
    blend_pred_val = signal_blender.predict(X_blend_val_scaled)
    blend_val_accuracy = accuracy_score(y_blend_val, blend_pred_val)
    
    # Analyze probability distribution after calibration
    if calibration_method:
        print(f"\nAnalyzing probability distribution (after calibration)...")
        proba_cal = signal_blender.predict_proba(X_blend_val_scaled)
        max_proba_cal = proba_cal.max(axis=1)
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
    
    print(f"\n[SignalBlender] Validation label distribution (true):")
    for label, count in y_blend_val.value_counts().items():
        pct = count / len(y_blend_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\n[SignalBlender] Validation label distribution (predicted):")
    for label, count in blend_pred_val.value_counts().items():
        pct = count / len(blend_pred_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
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
            use_gpu=use_cuml
        )
    else:
        # Legacy training path
        print(f"\nTraining DirectionBlender on {len(X_blend_trade_train)} trade samples...")
        direction_blender = DirectionBlender(use_gpu=use_cuml)  # GPU-accelerated training
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
    
    print(f"\n[DirectionBlender] Validation label distribution (true):")
    for label, count in y_direction_val.value_counts().items():
        pct = count / len(y_direction_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\n[DirectionBlender] Validation label distribution (predicted):")
    for label, count in direction_pred_val.value_counts().items():
        pct = count / len(direction_pred_val) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
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
            'n_streams': 4,
            'random_state': 42
        }
        model_r = SignalBlender(model_params=model_params, use_gpu=use_cuml)
        
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
            'n_streams': 4,
            'random_state': 42
        }
        model_r = DirectionBlender(model_params=model_params, use_gpu=use_cuml)
        
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


if __name__ == "__main__":
    main()

