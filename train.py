# train.py
"""
Training script for regime detector and signal blender models.
Supports walk-forward training with 6-month training windows and 1-month validation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import timedelta
from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext
from core.feature_store import FeatureStore
from core.regime_detector import RegimeDetector, wilder_atr
from core.signal_blender import SignalBlender
from data.btc_data_loader import BTCDataLoader


def rule_based_regime(features: pd.DataFrame, price_df: pd.DataFrame) -> pd.Series:
    """
    Generate regime labels using rule-based logic.
    This will be used as training labels for the ML model.
    
    Args:
        features: Feature dataframe
        price_df: Raw OHLCV dataframe
        
    Returns:
        Series of regime labels (trend_up, trend_down, chop, high_vol, low_vol)
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
    
    # Compute ATR z-score
    atr = wilder_atr(price_df, 14)
    atr_mean = atr.rolling(100, min_periods=14).mean()
    atr_std = atr.rolling(100, min_periods=14).std()
    atr_zscore = ((atr - atr_mean) / atr_std.replace(0, np.nan)).reindex(features.index)
    
    # Rule-based labeling
    threshold = 0.001
    
    # Volatility regimes
    high_vol_mask = atr_zscore > 1.5
    low_vol_mask = atr_zscore < -1.0
    regimes[high_vol_mask] = "high_vol"
    regimes[low_vol_mask] = "low_vol"
    
    # Trend regimes
    trend_up_mask = (lr_slope > threshold) & (price > lr_mid)
    trend_dn_mask = (lr_slope < -threshold) & (price < lr_mid)
    
    trend_up_mask = trend_up_mask & (~high_vol_mask) & (~low_vol_mask)
    trend_dn_mask = trend_dn_mask & (~high_vol_mask) & (~low_vol_mask)
    
    regimes[trend_up_mask] = "trend_up"
    regimes[trend_dn_mask] = "trend_down"
    
    # Chop regime
    if lr_width_col in features.columns:
        lr_width = features[lr_width_col]
        width_pct = (lr_width / price.replace(0, np.nan))
        narrow_mask = (width_pct < 0.02) & (lr_slope.abs() < threshold)
        narrow_mask = narrow_mask & (~high_vol_mask) & (~low_vol_mask) & (~trend_up_mask) & (~trend_dn_mask)
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
    feature_store: FeatureStore,
    horizon: int = 10
) -> Tuple[RegimeDetector, SignalBlender, Dict]:
    """
    Train regime detector and signal blender on a training window.
    
    Args:
        df_train: Training data window
        signal_modules: List of signal modules
        context_modules: List of context modules
        feature_store: FeatureStore instance
        horizon: Future return horizon for blend labels
        
    Returns:
        Tuple of (regime_detector, signal_blender, training_stats)
    """
    # Build features
    feats = feature_store.build_features(df_train, signal_modules, context_modules)
    feats = feature_store.clean_features(feats)
    
    # Generate labels
    regime_y = rule_based_regime(feats, df_train)
    future_ret = future_return(df_train['close'], horizon=horizon)
    blend_y = np.sign(future_ret)
    blend_y = blend_y.reindex(feats.index).fillna(0).astype(int)
    
    # Remove invalid rows
    valid_mask = ~(regime_y.isna() | blend_y.isna())
    valid_mask = valid_mask & (regime_y.isin(['trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol']))
    
    X_regime = feats[valid_mask]
    regime_y = regime_y[valid_mask]
    X_blend_base = feats[valid_mask]
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
        'trend_up': 0, 'trend_down': 1, 'chop': 2,
        'high_vol': 3, 'low_vol': 4
    }).fillna(5)
    
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
        'feature_count': len(feats.columns)
    }
    
    return reg, blender, stats


def validate_one_window(
    df_val: pd.DataFrame,
    reg_model: RegimeDetector,
    blender_model: SignalBlender,
    signal_modules: List,
    context_modules: List,
    feature_store: FeatureStore,
    horizon: int = 10
) -> Dict:
    """
    Validate models on a validation window.
    
    Args:
        df_val: Validation data window
        reg_model: Trained regime detector
        blender_model: Trained signal blender
        signal_modules: List of signal modules
        context_modules: List of context modules
        feature_store: FeatureStore instance
        horizon: Future return horizon for evaluation
        
    Returns:
        Dictionary with validation metrics
    """
    # Build features
    feats = feature_store.build_features(df_val, signal_modules, context_modules)
    feats = feature_store.clean_features(feats)
    
    # Generate true labels
    regime_y_true = rule_based_regime(feats, df_val)
    future_ret = future_return(df_val['close'], horizon=horizon)
    blend_y_true = np.sign(future_ret)
    blend_y_true = blend_y_true.reindex(feats.index).fillna(0).astype(int)
    
    # Remove invalid rows
    valid_mask = ~(regime_y_true.isna() | blend_y_true.isna())
    valid_mask = valid_mask & (regime_y_true.isin(['trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol']))
    
    X_regime = feats[valid_mask]
    regime_y_true = regime_y_true[valid_mask]
    X_blend_base = feats[valid_mask]
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
        'trend_up': 0, 'trend_down': 1, 'chop': 2,
        'high_vol': 3, 'low_vol': 4
    })
    X_blend["regime"] = regime_y_pred.map(regime_map).fillna(5)
    
    # Predict blend
    blend_y_pred = blender_model.predict(X_blend)
    
    # Calculate metrics
    regime_accuracy = (regime_y_pred == regime_y_true).mean()
    blend_accuracy = (blend_y_pred == blend_y_true).mean()
    
    # Calculate returns for signal accuracy
    actual_returns = future_return(df_val['close'], horizon=horizon).reindex(blend_y_true.index)
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


def combined_training(
    df: pd.DataFrame,
    signal_modules: List,
    context_modules: List,
    models_dir: Path,
    train_months: int = 6,
    val_months: int = 1,
    slide_months: int = 1,
    horizon: int = 10
) -> Tuple[RegimeDetector, SignalBlender]:
    """
    Perform walk-forward training across all windows.
    
    Args:
        df: Full dataset
        signal_modules: List of signal modules
        context_modules: List of context modules
        models_dir: Directory to save models
        train_months: Training window size in months
        val_months: Validation window size in months
        slide_months: Slide forward by this many months
        horizon: Future return horizon
        
    Returns:
        Tuple of (final_regime_detector, final_signal_blender)
    """
    feature_store = FeatureStore()
    
    # Generate splits
    splits = generate_splits(df, train_months, val_months, slide_months)
    
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
        
        # Extract windows
        df_train = df[(df.index >= train_start) & (df.index < train_end)]
        df_val = df[(df.index >= val_start) & (df.index < val_end)]
        
        if len(df_train) == 0 or len(df_val) == 0:
            print(f"Warning: Skipping window {i} - insufficient data")
            continue
        
        # Train on window
        print("Training models...")
        try:
            reg, blender, train_stats = train_one_window(
                df_train, signal_modules, context_modules, feature_store, horizon
            )
            print(f"  Training samples: {train_stats['train_samples']}")
            print(f"  Features: {train_stats['feature_count']}")
        except Exception as e:
            print(f"  Error during training: {e}")
            continue
        
        # Validate on window
        print("Validating models...")
        try:
            val_metrics = validate_one_window(
                df_val, reg, blender, signal_modules, context_modules, feature_store, horizon
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
            df_train = df[(df.index >= last_split[0]) & (df.index < last_split[1])]
            best_reg, best_blender, _ = train_one_window(
                df_train, signal_modules, context_modules, feature_store, horizon
            )
        else:
            raise ValueError("Cannot create final models - no successful training")
    
    return best_reg, best_blender


def main():
    """
    Main training function.
    """
    # Configuration
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    horizon = 10
    train_split = 0.8  # 80% train, 20% validation
    
    # 1. Load BTC CSV
    print("Loading BTC CSV data...")
    from data.btc_data_loader import load_btc_csv
    df = load_btc_csv("data/btcusd_1min_volspikes.csv")
    
    if df.empty:
        print("Error: No data loaded.")
        return
    
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # 2. Build the feature store
    print("\nBuilding feature store...")
    feature_store = FeatureStore()
    feats = feature_store.build(df)
    print(f"Built {len(feats.columns)} features")
    
    # 3. Create regime labels (rule-based)
    print("\nCreating regime labels...")
    regime_y = rule_based_regime(feats, df)
    print(f"Regime distribution:\n{regime_y.value_counts()}")
    
    # 4. Create blender labels: future_return(close, horizon=10)
    print(f"\nCreating blender labels (horizon={horizon})...")
    blend_y = future_return(df['close'], horizon=horizon)
    blend_y = blend_y.reindex(feats.index)
    print(f"Blend labels stats: min={blend_y.min():.4f}, max={blend_y.max():.4f}, mean={blend_y.mean():.4f}")
    
    # Remove rows with NaN or invalid labels
    valid_mask = ~(regime_y.isna() | blend_y.isna())
    valid_mask = valid_mask & (regime_y.isin(['trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol']))
    
    feats = feats[valid_mask]
    regime_y = regime_y[valid_mask]
    blend_y = blend_y[valid_mask]
    df = df[valid_mask]
    
    print(f"\nValid samples: {len(feats)}")
    
    # 5. Split into train/validation
    print(f"\nSplitting into train/validation ({train_split:.0%}/{1-train_split:.0%})...")
    split_idx = int(len(feats) * train_split)
    
    X_train = feats.iloc[:split_idx]
    X_val = feats.iloc[split_idx:]
    regime_y_train = regime_y.iloc[:split_idx]
    regime_y_val = regime_y.iloc[split_idx:]
    blend_y_train = blend_y.iloc[:split_idx]
    blend_y_val = blend_y.iloc[split_idx:]
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # For signal blender, we need features + module signals + regime
    # Compute module signals for training
    print("\nComputing module signals for blender...")
    module_signals_train = {}
    module_signals_val = {}
    
    for module in feature_store.signal_modules:
        # Extract module-specific features
        module_features_train = X_train.filter(regex=f'^{module.name}_', axis=1).copy()
        module_features_val = X_val.filter(regex=f'^{module.name}_', axis=1).copy()
        
        if not module_features_train.empty:
            # Remove prefix
            new_cols = pd.Index([col.replace(f'{module.name}_', '') for col in module_features_train.columns])
            module_features_train.columns = new_cols
            module_features_val.columns = new_cols
            
            # Compute signals
            module_signals_train[module.name] = module.compute_signal(module_features_train)
            module_signals_val[module.name] = module.compute_signal(module_features_val)
    
    # Build X_blend with features + signals + regime
    X_blend_train = X_train.copy()
    X_blend_val = X_val.copy()
    
    for module_name, signal in module_signals_train.items():
        aligned_signal = signal.reindex(X_blend_train.index).fillna(0)
        X_blend_train[f"{module_name}_signal"] = aligned_signal
    
    for module_name, signal in module_signals_val.items():
        aligned_signal = signal.reindex(X_blend_val.index).fillna(0)
        X_blend_val[f"{module_name}_signal"] = aligned_signal
    
    # Encode regime
    regime_map = pd.Series({
        'trend_up': 0, 'trend_down': 1, 'chop': 2,
        'high_vol': 3, 'low_vol': 4
    })
    X_blend_train["regime"] = regime_y_train.map(regime_map).fillna(5)
    X_blend_val["regime"] = regime_y_val.map(regime_map).fillna(5)
    
    # 6. Fit regime_detector
    print("\nTraining regime detector...")
    regime_detector = RegimeDetector()
    regime_detector.fit(X_train, regime_y_train)
    
    # Validate regime detector
    regime_pred_val = regime_detector.predict(X_val)
    regime_accuracy = (regime_pred_val == regime_y_val).mean()
    print(f"Regime validation accuracy: {regime_accuracy:.4f}")
    
    # 7. Fit signal_blender
    print("\nTraining signal blender...")
    signal_blender = SignalBlender(return_threshold=0.001)
    signal_blender.fit(X_blend_train, blend_y_train)
    
    # Validate signal blender
    blend_pred_val = signal_blender.predict(X_blend_val)
    blend_accuracy = (np.sign(blend_pred_val) == np.sign(blend_y_val)).mean()
    print(f"Blend validation accuracy: {blend_accuracy:.4f}")
    
    # 8. Save .pkl files
    print("\nSaving models...")
    regime_detector.save(str(models_dir / "regime_detector.pkl"))
    signal_blender.save(str(models_dir / "signal_blender.pkl"))
    
    print("\nTraining complete!")
    print(f"Models saved to {models_dir}/")
    print(f"  - regime_detector.pkl")
    print(f"  - signal_blender.pkl")


if __name__ == "__main__":
    main()

