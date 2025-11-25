# inference.py
"""
Inference script for regime detection and signal generation.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext
from core.feature_store import FeatureStore
from core.regime_detector import RegimeDetector
from core.signal_blender import SignalBlender


def infer(
    df: pd.DataFrame,
    regime_model_path: str = "models/regime_detector.pkl",
    blender_model_path: str = "models/signal_blender.pkl"
) -> dict:
    """
    Run inference on incoming bar data.
    
    Args:
        df: Incoming bar data (OHLCV dataframe)
        regime_model_path: Path to regime detector model
        blender_model_path: Path to signal blender model
        
    Returns:
        Dictionary with keys: regime, signal, confidence
    """
    # Initialize modules
    signal_modules = [
        SuperMA4hr(),
        TrendMagicV2(),
        PVTEliminator(),
    ]
    
    context_modules = [
        PivotRSIContext(),
        LinRegChannelContext(),
    ]
    
    # Build features
    feature_store = FeatureStore()
    features = feature_store.build_features(df, signal_modules, context_modules)
    features = feature_store.clean_features(features)
    
    # Load models
    regime_detector = RegimeDetector()
    regime_detector.load(regime_model_path)
    
    signal_blender = SignalBlender()
    signal_blender.load(blender_model_path)
    
    # Predict regime
    regime = regime_detector.predict(features)
    regime_proba = regime_detector.predict_proba(features)
    
    # Compute module signals
    module_signals = {}
    for module in signal_modules:
        # Extract module-specific features (prefixed with module name)
        module_features = features.filter(regex=f'^{module.name}_', axis=1).copy()
        # Remove prefix from column names for module consumption
        if not module_features.empty:
            module_features.columns = [col.replace(f'{module.name}_', '') for col in module_features.columns]
        module_signals[module.name] = module.compute_signal(module_features)
    
    # Prepare features for blender (features + signals + regime)
    X_blend = features.copy()
    for module_name, signal in module_signals.items():
        X_blend[f"{module_name}_signal"] = signal
    X_blend["regime"] = regime
    
    # Predict signal
    signal = signal_blender.predict(X_blend)
    signal_proba = signal_blender.predict_proba(X_blend)
    
    # Compute confidence (max probability from regime detector)
    confidence = regime_proba.max(axis=1)
    
    # Return results for the last bar
    last_idx = df.index[-1]
    
    return {
        "regime": regime.loc[last_idx],
        "signal": int(signal.loc[last_idx]),
        "confidence": float(confidence.loc[last_idx])
    }


def main():
    """
    Main inference function.
    """
    # TODO: Load incoming bar data
    # This should be called with real-time or batch data
    print("Loading data...")
    
    # Placeholder - replace with actual data loading
    df = pd.DataFrame()
    
    if df.empty:
        print("No data provided. Exiting.")
        return
    
    # Run inference
    result = infer(df)
    
    print("Inference results:")
    print(f"  Regime: {result['regime']}")
    print(f"  Signal: {result['signal']}")
    print(f"  Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()

