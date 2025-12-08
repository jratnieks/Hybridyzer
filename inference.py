# inference.py
"""
Inference script for regime detection and signal generation.
Uses FinalSignalGenerator to combine RegimeDetector, DirectionBlender, and SignalBlender.
Supports regime-specific models for improved accuracy.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from core.final_signal import FinalSignalGenerator


def infer(
    df: pd.DataFrame,
    probability_threshold: float = 0.60,
    require_blender_agreement: bool = False,
    regime_model_path: str = "models/regime_model.pkl",
    direction_model_path: str = "models/blender_direction_model.pkl",
    blender_model_path: str = "models/blender_model.pkl",
    scaler_path: str = "models/feature_scaler.pkl",
    use_gpu: bool = True
) -> dict:
    """
    Run inference on incoming bar data using FinalSignalGenerator.
    
    Args:
        df: Incoming bar data (OHLCV dataframe)
        probability_threshold: Minimum probability for DirectionBlender to take a trade (default: 0.60)
        require_blender_agreement: If True, require 3-class SignalBlender to agree (default: False)
        regime_model_path: Path to regime detector model
        direction_model_path: Path to direction blender model
        blender_model_path: Path to signal blender (3-class) model
        scaler_path: Path to feature scaler
        use_gpu: Whether to use GPU acceleration (default: True)
        
    Returns:
        Dictionary with keys: regime, signal, confidence, direction_proba
        For single-row input, returns scalar values.
        For multi-row input, returns Series.
    """
    # Initialize FinalSignalGenerator
    signal_generator = FinalSignalGenerator(
        probability_threshold=probability_threshold,
        require_blender_agreement=require_blender_agreement,
        use_gpu=use_gpu
    )
    
    # Load all models
    signal_generator.load_models(
        regime_model_path=regime_model_path,
        direction_model_path=direction_model_path,
        blender_model_path=blender_model_path,
        scaler_path=scaler_path
    )
    
    # Generate final signal
    results = signal_generator.generate_signal(df)
    
    # For single-row input, return scalar values for convenience
    # For multi-row input, return Series
    if len(df) == 1:
        last_idx = df.index[-1]
        return {
            "regime": results['regime'].loc[last_idx],
            "signal": int(results['signal'].loc[last_idx]),
            "confidence": float(results['confidence'].loc[last_idx]),
            "direction_proba": float(results['direction_proba'].loc[last_idx])
        }
    else:
        return results


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
    
    # Run inference with configurable parameters
    result = infer(
        df,
        probability_threshold=0.60,  # Minimum confidence for taking a trade
        require_blender_agreement=False  # Set to True to require 3-class SignalBlender agreement
    )
    
    print("\nInference results:")
    print(f"  Regime: {result['regime']}")
    print(f"  Signal: {result['signal']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Direction Probability: {result['direction_proba']:.4f}")


if __name__ == "__main__":
    main()

