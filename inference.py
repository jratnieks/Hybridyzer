# inference.py
"""
Inference script for regime detection and signal generation.
Uses FinalSignalGenerator to combine RegimeDetector, DirectionBlender, and SignalBlender.
Supports regime-specific models for improved accuracy.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from core.final_signal import FinalSignalGenerator
from core.pattern_memory import PatternMemory, MemoryGate


def infer(
    df: pd.DataFrame,
    probability_threshold: float = 0.60,
    require_blender_agreement: bool = False,
    regime_model_path: str = "models/regime_model.pkl",
    direction_model_path: str = "models/blender_direction_model.pkl",
    blender_model_path: str = "models/blender_model.pkl",
    scaler_path: str = "models/feature_scaler.pkl",
    use_gpu: bool = True,
    enable_pattern_memory: bool = False,
    enable_memory_gate: bool = False,
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
    
    # Build features
    features = signal_generator._build_features(df)
    
    # Add PatternMemory features if enabled
    pattern_memory = None
    if enable_pattern_memory:
        models_dir = Path(regime_model_path).parent
        pattern_memory_path = models_dir / "pattern_memory"
        if pattern_memory_path.parent.exists():
            pattern_memory = PatternMemory()
            try:
                pattern_memory.load(pattern_memory_path)
                # Add mem_* features
                mem_features = pattern_memory.transform(features, df)
                features = pd.concat([features, mem_features], axis=1)
            except Exception as e:
                print(f"[PatternMemory] Warning: Could not load memory: {e}")
                pattern_memory = None
    
    # Generate final signal
    signals, ev_series = signal_generator.predict(features=features, df=df)
    
    # Get regime predictions
    regime = signal_generator.regime_detector.predict(features)
    
    # Apply MemoryGate if enabled
    if enable_memory_gate and pattern_memory is not None:
        memory_gate = MemoryGate()
        # Get direction confidence for soft penalty
        direction_confidence = signal_generator.direction_confidence.reindex(signals.index).fillna(0.0)
        signals = memory_gate.apply(signals, features, ml_proba=direction_confidence)
    
    # Build results dict (matching generate_signal format)
    results = {
        'signal': signals,
        'regime': regime.reindex(signals.index),
        'confidence': signal_generator.direction_confidence.reindex(signals.index).fillna(0.0),
        'direction_proba': signal_generator.direction_confidence.reindex(signals.index).fillna(0.0),
    }
    
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
    Main inference function - runs on test data.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument("--data", type=str, default="data/btcusd_5min_test_2025.csv",
                        help="Path to data file (default: 2025 test data)")
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="Probability threshold for trades (default: 0.60)")
    parser.add_argument("--require-agreement", action="store_true",
                        help="Require SignalBlender agreement")
    parser.add_argument("--tail", type=int, default=20,
                        help="Show last N signals (default: 20)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU (use CPU only)")
    parser.add_argument('--pattern-memory', action='store_true',
                        help='Enable PatternMemory: load memory table and add mem_* features (default: off)')
    parser.add_argument('--memory-gate', action='store_true',
                        help='Enable MemoryGate: apply veto/penalty to signals (default: off)')
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    
    try:
        df = pd.read_csv(args.data, parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"Error: Data file not found: {args.data}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if df.empty:
        print("No data provided. Exiting.")
        return
    
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"Running inference with threshold={args.threshold}...")
    
    # Run inference
    results = infer(
        df,
        probability_threshold=args.threshold,
        require_blender_agreement=args.require_agreement,
        use_gpu=not args.no_gpu,
        enable_pattern_memory=args.pattern_memory,
        enable_memory_gate=args.memory_gate,
    )
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'regime': results['regime'],
        'signal': results['signal'],
        'confidence': results['confidence'],
        'direction_proba': results['direction_proba']
    })
    
    # Signal distribution
    print("\n=== Signal Distribution ===")
    signal_counts = summary['signal'].value_counts().sort_index()
    signal_names = {-1: "SHORT", 0: "FLAT", 1: "LONG"}
    for sig, count in signal_counts.items():
        pct = 100 * count / len(summary)
        print(f"  {signal_names.get(sig, sig):>5}: {count:>6} ({pct:.1f}%)")
    
    # Regime distribution
    print("\n=== Regime Distribution ===")
    regime_counts = summary['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(summary)
        print(f"  {regime:>12}: {count:>6} ({pct:.1f}%)")
    
    # Show last N signals
    print(f"\n=== Last {args.tail} Signals ===")
    tail_df = summary.tail(args.tail).copy()
    tail_df['signal_name'] = tail_df['signal'].map(signal_names)
    print(tail_df[['regime', 'signal_name', 'confidence', 'direction_proba']].to_string())


if __name__ == "__main__":
    main()

