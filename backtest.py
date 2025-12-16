# backtest.py
"""
Simple long/short backtest using trained regime detector and signal blender models.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from core.feature_store import FeatureStore
from core.regime_detector import RegimeDetector
from core.signal_blender import SignalBlender, DirectionBlender
from core.final_signal import FinalSignalGenerator
from core.labeling import make_direction_labels
from core.profiles import get_profile, list_profiles
from data.btc_data_loader import load_btc_csv


def compute_sharpe(returns: pd.Series, periods_per_year: int = 2190) -> float:
    """
    Compute Sharpe ratio (annualized) for a returns series.
    
    Args:
        returns: Series of period returns
        periods_per_year: Number of periods per year (default: 2190 for 4-hour bars)
        
    Returns:
        Sharpe ratio (annualized)
    """
    if len(returns) > 1:
        returns_clean = returns.dropna()
        if len(returns_clean) > 1 and returns_clean.std() > 0:
            return np.sqrt(periods_per_year) * returns_clean.mean() / returns_clean.std()
    return 0.0


def assign_quantile_prob_bins(trades_df: pd.DataFrame, prob_col: str = "direction_confidence", n_bins: int = 5) -> pd.DataFrame:
    """
    Assign quantile-based probability bins to trades_df['prob'].
    
    Adds:
      - 'prob_bin' (categorical interval label)
      - 'prob_min', 'prob_max' (float boundaries for each bin row)
    
    Args:
        trades_df: DataFrame with probability column
        prob_col: Column name for probabilities
        n_bins: Number of quantile bins to create
        
    Returns:
        DataFrame with added prob_bin, prob_min, prob_max columns
    """
    df = trades_df.copy()
    
    # Guard: if all probs are the same, just put everything in one bin
    prob = df[prob_col].astype(float)
    if prob.nunique() <= 1:
        prob_min = prob.min()
        prob_max = prob.max()
        # Create a single interval
        df['prob_bin'] = pd.Interval(prob_min, prob_max, closed='both')
        df['prob_min'] = prob_min
        df['prob_max'] = prob_max
        return df
    
    # Quantile-based bins
    # Example: 5 bins => [0–20%, 20–40%, 40–60%, 60–80%, 80–100%]
    try:
        df['prob_bin'] = pd.qcut(prob, q=n_bins, duplicates='drop')
    except ValueError:
        # Fallback: fewer bins if qcut complains
        unique = prob.nunique()
        q = min(n_bins, max(1, unique))
        df['prob_bin'] = pd.qcut(prob, q=q, duplicates='drop')
    
    # Extract numeric edges per bin
    df['prob_min'] = df['prob_bin'].map(lambda iv: float(iv.left))
    df['prob_max'] = df['prob_bin'].map(lambda iv: float(iv.right))
    
    return df


def generate_calibration_report(
    df: pd.DataFrame,
    prob_col: str = "direction_confidence",
    signal_col: str = "final_signal",
    return_col: str = "future_return",
    regime_col: str = "regime"
) -> dict:
    """
    Generate probability calibration report for DirectionBlender.
    Uses quantile-based probability bins with numeric edges.
    
    Args:
        df: DataFrame with columns: prob_col, signal_col, return_col, regime_col
        prob_col: Column name for direction confidence probabilities
        signal_col: Column name for final signals (-1, 0, 1)
        return_col: Column name for future returns
        regime_col: Column name for regime labels
        
    Returns:
        Dictionary with keys: 'overall', 'per_side', 'per_regime_side'
        Each value is a DataFrame with calibration metrics including prob_min/prob_max
    """
    # Filter to only trades (non-zero signals and non-NaN returns)
    trades_df = df[(df[signal_col] != 0) & ~df[return_col].isna()].copy()
    
    if len(trades_df) == 0:
        print("No trades found for calibration report")
        return {
            'overall': pd.DataFrame(),
            'per_side': pd.DataFrame(),
            'per_regime_side': pd.DataFrame()
        }
    
    # Add side column
    trades_df['side'] = np.sign(trades_df[signal_col])
    trades_df['side_str'] = trades_df['side'].map({1: 'long', -1: 'short', 0: 'flat'})
    
    # Assign quantile-based probability bins with numeric edges
    trades_df = assign_quantile_prob_bins(trades_df, prob_col=prob_col, n_bins=5)
    
    # Compute hit rate: sign(future_return) == sign(final_signal) for non-zero signals
    trades_df['hit'] = (np.sign(trades_df[return_col]) == np.sign(trades_df[signal_col])).astype(int)
    trades_df['is_win'] = trades_df['hit']  # Alias for clarity
    trades_df['gross_return'] = trades_df[return_col]  # Alias for clarity
    
    def compute_group_metrics(group):
        """Compute calibration metrics for a group."""
        return pd.Series({
            'count': len(group),
            'hit_rate': group['is_win'].mean() if len(group) > 0 else 0.0,
            'avg_gross_return': group['gross_return'].mean() if len(group) > 0 else 0.0,
            'median_gross_return': group['gross_return'].median() if len(group) > 0 else 0.0,
            'avg_abs_return': group['gross_return'].abs().mean() if len(group) > 0 else 0.0,
        })
    
    # Overall calibration (by prob_bin only) - preserve prob_min/prob_max
    overall = (
        trades_df
        .groupby('prob_bin', observed=False)
        .apply(compute_group_metrics)
        .reset_index()
    )
    
    # Merge numeric edges from the original df
    edges = (
        trades_df
        .groupby('prob_bin', observed=False)[['prob_min', 'prob_max']]
        .agg({'prob_min': 'min', 'prob_max': 'max'})
        .reset_index()
    )
    overall = overall.merge(edges, on='prob_bin', how='left')
    
    # Per-side calibration
    per_side = (
        trades_df
        .groupby(['prob_bin', 'side_str'], observed=False)
        .apply(compute_group_metrics)
        .reset_index()
    )
    
    edges_side = (
        trades_df
        .groupby(['prob_bin', 'side_str'], observed=False)[['prob_min', 'prob_max']]
        .agg({'prob_min': 'min', 'prob_max': 'max'})
        .reset_index()
    )
    per_side = per_side.merge(edges_side, on=['prob_bin', 'side_str'], how='left')
    
    # Per-regime+side calibration
    per_regime_side = (
        trades_df
        .groupby(['prob_bin', regime_col, 'side_str'], observed=False)
        .apply(compute_group_metrics)
        .reset_index()
    )
    
    edges_regime_side = (
        trades_df
        .groupby(['prob_bin', regime_col, 'side_str'], observed=False)[['prob_min', 'prob_max']]
        .agg({'prob_min': 'min', 'prob_max': 'max'})
        .reset_index()
    )
    per_regime_side = per_regime_side.merge(
        edges_regime_side,
        on=['prob_bin', regime_col, 'side_str'],
        how='left'
    )
    
    return {
        'overall': overall,
        'per_side': per_side,
        'per_regime_side': per_regime_side
    }


def compute_metrics(returns: pd.Series, equity: pd.Series, positions: pd.Series) -> dict:
    """
    Compute backtest performance metrics.
    
    Args:
        returns: Series of period returns (PnL)
        equity: Series of cumulative equity curve
        positions: Series of positions (1=long, -1=short, 0=flat)
        
    Returns:
        Dictionary of metrics (total_return, cagr, sharpe, max_drawdown, hit_rate, long_accuracy, short_accuracy)
    """
    metrics = {}
    
    # Total return and CAGR
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    n_periods = len(returns)
    # Approximate periods per year (assuming 4-hour bars: 6 per day * 365 = 2190 per year)
    # Adjust based on actual data frequency
    periods_per_year = 2190  # For 4-hour bars
    if n_periods > 0:
        years = n_periods / periods_per_year
        if years > 0:
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
        else:
            cagr = total_return
    else:
        cagr = 0.0
    metrics['total_return'] = total_return
    metrics['cagr'] = cagr
    
    # Sharpe ratio (annualized)
    if len(returns) > 1:
        returns_clean = returns.dropna()
        if len(returns_clean) > 1 and returns_clean.std() > 0:
            sharpe = np.sqrt(periods_per_year) * returns_clean.mean() / returns_clean.std()
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    metrics['sharpe'] = sharpe
    
    # Max drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    metrics['max_drawdown'] = max_drawdown
    
    # Hit rate (percentage of profitable trades)
    profitable_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    if total_trades > 0:
        hit_rate = profitable_trades / total_trades
    else:
        hit_rate = 0.0
    metrics['hit_rate'] = hit_rate
    
    # Long accuracy / Short accuracy
    # Filter returns based on position direction (positive = long, negative = short)
    # Note: returns = position * future_return, so:
    # - If position > 0 (long) and future_return > 0 → returns > 0 (profitable long)
    # - If position < 0 (short) and future_return < 0 → returns > 0 (profitable short)
    # Align positions with returns index
    positions_aligned = positions.reindex(returns.index)
    long_mask = positions_aligned > 0
    short_mask = positions_aligned < 0
    
    long_returns = returns[long_mask]
    short_returns = returns[short_mask]
    
    if len(long_returns) > 0:
        long_accuracy = (long_returns > 0).sum() / len(long_returns)
    else:
        long_accuracy = 0.0
    
    if len(short_returns) > 0:
        short_accuracy = (short_returns > 0).sum() / len(short_returns)
    else:
        short_accuracy = 0.0
    
    metrics['long_accuracy'] = long_accuracy
    metrics['short_accuracy'] = short_accuracy
    
    return metrics


def main():
    """Run backtest on trained models."""
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Backtest trained models')
    parser.add_argument('--recent', type=int, default=None, help='Only use last N bars for backtest')
    parser.add_argument('--p', type=float, default=None,
                        help='Probability threshold for DirectionBlender trades (overrides default, applies to both long and short)')
    parser.add_argument('--p-long', '--p_long', type=float, default=None, dest='p_long',
                        help='Probability threshold for long trades (overrides --p for longs)')
    parser.add_argument('--p-short', '--p_short', type=float, default=None, dest='p_short',
                        help='Probability threshold for short trades (overrides --p for shorts)')
    parser.add_argument('--profile', type=str, default=None,
                        help=f'Use named profile configuration. Available: {", ".join(list_profiles())}')
    parser.add_argument('--fee-bps-per-side', type=float, default=2.0,
                        help='Fee in basis points per side (default: 2.0)')
    parser.add_argument('--slippage-bps-per-side', type=float, default=1.0,
                        help='Slippage in basis points per side (default: 1.0)')
    parser.add_argument('--disable-shorts', action='store_true',
                        help='Force all short signals to be rejected before backtest or cost processing')
    parser.add_argument('--min-gross-return', type=float, default=0.0,
                        help='Minimum required gross return per trade (fraction). Trades below this are zeroed out.')
    parser.add_argument('--min-ev', type=float, default=None,
                        help='Minimum expected value threshold. Only allow trades with EV >= min_ev (requires --calibration-csv)')
    parser.add_argument('--calibration-csv', type=str, default=None,
                        help='Path to calibration CSV file (e.g., results/calibration_overall_*.csv) for EV filtering')
    parser.add_argument('--horizon-bars', type=int, default=48,
                        help='Forward horizon in bars for label generation (default: 48 = 4 hours for 5-min data)')
    parser.add_argument('--label-threshold', type=float, default=0.0005,
                        help='Threshold for direction labels (default: 0.0005 = 0.05%)')
    parser.add_argument('--dump-ev-analysis', action='store_true',
                        help='Dump EV diagnostics, plots, and tables to results/')
    parser.add_argument('--use-cuml', action='store_true',
                        help='Enable cuML/cuDF GPU models (requires RAPIDS; fails if unavailable)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='Chunk size for GPU processing (default: 100000, reduce to 50000 if OOM)')
    args = parser.parse_args()
    
    # Load profile if specified (STRICT: exit if not found)
    profile = None
    profile_name = "<none>"
    if args.profile:
        profile = get_profile(args.profile)
        if profile is None:
            available = list_profiles()
            print(f"Error: Profile '{args.profile}' not found. Available profiles: {available}")
            import sys
            sys.exit(1)
        profile_name = args.profile
    
    # Extract probability thresholds with safer defaults
    # Default p = 0.6 if not provided
    default_p = 0.6
    if profile and profile.get('probability_threshold') is not None and args.p is None:
        prob_threshold = profile['probability_threshold']
    else:
        prob_threshold = args.p if args.p is not None else default_p
    
    # Handle separate long/short thresholds (default to p if not set)
    p_long = args.p_long if args.p_long is not None else prob_threshold
    p_short = args.p_short if args.p_short is not None else prob_threshold
    
    # Transaction costs
    fee_bps = args.fee_bps_per_side
    slippage_bps = args.slippage_bps_per_side
    total_cost_bps = 2 * (fee_bps + slippage_bps)  # Round trip cost
    
    # Label configuration
    HORIZON_BARS = args.horizon_bars
    LABEL_THRESHOLD = args.label_threshold
    SMOOTHING_WINDOW = 12
    
    # Print configuration block
    print("\n" + "="*60)
    print("BACKTEST CONFIGURATION")
    print("="*60)
    print(f"Active profile: {profile_name}")
    if profile:
        print(f"  disable_regime: {profile.get('disable_regime', 'N/A')}")
        print(f"  regime_policy: {profile.get('regime_policy', 'N/A')}")
    print(f"Label generation (NEW SYSTEM):")
    print(f"  horizon_bars: {HORIZON_BARS}")
    print(f"  label_threshold: {LABEL_THRESHOLD}")
    print(f"  smoothing_window: {SMOOTHING_WINDOW}")
    print(f"Probability thresholds:")
    print(f"  p (general): {prob_threshold}")
    print(f"  p_long: {p_long}")
    print(f"  p_short: {p_short}")
    print(f"Transaction costs:")
    print(f"  fee: {fee_bps} bps/side")
    print(f"  slippage: {slippage_bps} bps/side")
    print(f"  total round-trip: {total_cost_bps} bps ({total_cost_bps/100:.2f}%)")
    print("="*60 + "\n")
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw BTC CSV
    print("Loading BTC data...")
    csv_paths = [
        data_dir / "btcusd_4H.csv",
        data_dir / "btcusd_1min.csv",
        data_dir / "btc_1m.csv",
    ]
    
    df_full = None
    for path in csv_paths:
        if Path(path).exists():
            df_full = load_btc_csv(str(path))
            print(f"Loaded {len(df_full)} bars from {path}")
            break

    if df_full is None:
        # Try to find any CSV in data directory
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            df_full = load_btc_csv(str(csv_files[0]))
            print(f"Loaded {len(df_full)} bars from {csv_files[0]}")
        else:
            raise FileNotFoundError("No BTC CSV file found in data/ directory")
    
    # 2. Build features with FeatureStore (same settings as train.py)
    use_cuml = args.use_cuml
    if use_cuml:
        print("[GPU] cuML requested via --use-cuml")
    else:
        print("[CPU] sklearn backend active (no --use-cuml)")

    feature_store_use_gpu = False
    print(f"\nBuilding feature store... (GPU: {feature_store_use_gpu}, chunk_size: {args.chunk_size})")
    feature_store = FeatureStore(
        safe_mode=True,
        use_gpu=feature_store_use_gpu,
        chunk_size=args.chunk_size
    )
    
    # Use the FeatureStore's own module definitions to match training exactly
    signal_modules = feature_store.signal_modules
    context_modules = feature_store.context_modules
    
    features_full = feature_store.build(df_full)
    print(f"Built {features_full.shape[1]} features for {len(features_full)} bars")
    # Print final feature columns to verify parity with training
    print("\nFINAL FEATURE COLUMNS:", features_full.columns.tolist())
    
    # 3. Load trained models
    regime_model_path = models_dir / "regime_model.pkl"
    blender_model_path = models_dir / "blender_model.pkl"
    direction_model_path = models_dir / "blender_direction_model.pkl"
    
    if not regime_model_path.exists():
        raise FileNotFoundError(f"Regime model not found: {regime_model_path}")
    if not blender_model_path.exists():
        raise FileNotFoundError(f"Blender model not found: {blender_model_path}")
    if not direction_model_path.exists():
        raise FileNotFoundError(f"Direction model not found: {direction_model_path}")
    
    print("\nLoading models...")
    regime_detector = RegimeDetector(use_gpu=use_cuml)
    regime_detector.load(str(regime_model_path))
    print(f"Loaded RegimeDetector (GPU: {use_cuml})")
    
    # Try to load calibrated models first, fall back to regular models
    blender_calibrated_path = models_dir / "blender_calibrated.pkl"
    if blender_calibrated_path.exists():
        signal_blender = SignalBlender(use_gpu=use_cuml)
        signal_blender.load(str(blender_calibrated_path))
        print(f"Loaded SignalBlender (calibrated) (GPU: {use_cuml})")
    else:
        signal_blender = SignalBlender(use_gpu=use_cuml)
        signal_blender.load(str(blender_model_path))
        print(f"Loaded SignalBlender (GPU: {use_cuml})")
    
    direction_calibrated_path = models_dir / "blender_direction_calibrated.pkl"
    if direction_calibrated_path.exists():
        direction_blender = DirectionBlender(use_gpu=use_cuml)
        direction_blender.load(str(direction_calibrated_path))
        print(f"Loaded DirectionBlender (calibrated) (GPU: {use_cuml})")
    else:
        direction_blender = DirectionBlender(use_gpu=use_cuml)
        direction_blender.load(str(direction_model_path))
        print(f"Loaded DirectionBlender (GPU: {use_cuml})")
    
    # Instantiate FinalSignalGenerator with profile settings
    # MERGE ORDER: defaults → CLI → profile (profile HARD overrides)
    
    # Step 1: Load all defaults
    final_signal_gen_kwargs = {
        'regime_detector': regime_detector,
        'signal_blender': signal_blender,
        'direction_blender': direction_blender,
        'use_gpu': use_cuml,
        'require_blender_agreement': True,
        'probability_threshold': prob_threshold,
        'p_long': p_long,
        'p_short': p_short,
    }
    
    # Step 2: CLI flags already applied above (p_long, p_short, prob_threshold)
    
    # Step 3: Profile HARD overrides defaults AND CLI options
    if profile:
        # Profile disable_regime HARD overrides
        if 'disable_regime' in profile:
            final_signal_gen_kwargs['disable_regime'] = profile['disable_regime']
        
        # Profile regime_policy HARD overrides (note: using 'regime_policy' key)
        if 'regime_policy' in profile:
            final_signal_gen_kwargs['regime_side_policy'] = profile['regime_policy']
        
        # Profile probability threshold override (if provided)
        if profile.get('probability_threshold') is not None:
            final_signal_gen_kwargs['probability_threshold'] = profile['probability_threshold']
            # Recompute p_long and p_short if they weren't explicitly set via CLI
            if args.p_long is None:
                final_signal_gen_kwargs['p_long'] = profile['probability_threshold']
            if args.p_short is None:
                final_signal_gen_kwargs['p_short'] = profile['probability_threshold']
        
        # Profile disable_shorts override
        if 'disable_shorts' in profile:
            args.disable_shorts = profile['disable_shorts']
    
    # Add disable_shorts to kwargs
    final_signal_gen_kwargs['disable_shorts'] = args.disable_shorts
    
    # Add EV filtering parameters
    final_signal_gen_kwargs['min_ev'] = args.min_ev
    
    # Auto-find calibration CSV if not specified
    calibration_csv = args.calibration_csv
    if calibration_csv is None:
        # Try to find a calibration file in results directory
        if results_dir.exists():
            # Look for overall calibration files
            cal_files = sorted(results_dir.glob("calibration_overall*.csv"), reverse=True)
            if cal_files:
                calibration_csv = str(cal_files[0])
                print(f"[Auto-detected calibration CSV: {calibration_csv}]")
    
    final_signal_gen_kwargs['calibration_csv_path'] = calibration_csv
    
    final_signal_gen = FinalSignalGenerator(**final_signal_gen_kwargs)
    
    # 4. Slice data if --recent is specified
    if args.recent is not None and args.recent > 0:
        df = df_full.iloc[-args.recent:]
        features = features_full.loc[df.index]
        print(f"\nUsing last {args.recent} bars for backtest")
    else:
        df = df_full
        features = features_full
    
    # 5. Generate final signals with regime-aware ensemble
    print("\nGenerating final signals with regime-aware ensemble...")
    signals, ev_series = final_signal_gen.predict(features=features, df=df)
    
    # Capture direction confidence for calibration
    direction_confidence = final_signal_gen.direction_confidence.reindex(signals.index).fillna(0.0)
    
    # Apply min-gross-return guardrail BEFORE cost processing
    if args.min_gross_return > 0:
        print(f"\nApplying min-gross-return guardrail: {args.min_gross_return:.6f}")
        df_with_returns_temp = make_direction_labels(df, horizon_bars=1, label_threshold=LABEL_THRESHOLD,
                                                      smoothing_window=1, debug=False)
        future_returns_temp = df_with_returns_temp['future_return'].reindex(signals.index)
        # Only filter trades (non-zero signals) that don't meet minimum gross return
        trade_mask_temp = signals != 0
        gross_returns_temp = signals * future_returns_temp
        # Filter out trades where absolute gross return is below threshold
        below_threshold = trade_mask_temp & (np.abs(gross_returns_temp) < args.min_gross_return)
        signals.loc[below_threshold] = 0
        filtered_count = below_threshold.sum()
        print(f"Filtered {filtered_count} trades under min gross return threshold")
    
    # Print final trade config summary
    print("\n" + "="*60)
    print("FINAL TRADE CONFIG SUMMARY")
    print("="*60)
    print(f"disable_shorts: {args.disable_shorts}")
    print(f"p_long: {final_signal_gen.p_long}, p_short: {final_signal_gen.p_short}")
    print(f"min_gross_return: {args.min_gross_return}")
    # EV configuration output
    if args.calibration_csv is not None:
        print(f"calibration_csv: {args.calibration_csv}")
    if args.min_ev is not None:
        print(f"min_ev: {args.min_ev}")
        print(f"EV-filtered trades: {final_signal_gen.ev_filtered_count}")
    
    # Compute EV statistics (always, if data available)
    trade_mask_summary = signals != 0
    if trade_mask_summary.any():
        trade_evs_summary = ev_series[trade_mask_summary].dropna()
        if len(trade_evs_summary) > 0:
            print(f"avg_ev: {trade_evs_summary.mean():.6f}")
            print(f"median_ev: {trade_evs_summary.median():.6f}")
            if args.min_ev is not None:
                print(f"min_ev_threshold: {args.min_ev}")
                print(f"num_ev_filtered: {final_signal_gen.ev_filtered_count}")
        else:
            print(f"EV data: Not available (no calibration CSV or lookup failed)")
    print("="*60)
    
    print("\nSignal distribution:")
    print(signals.value_counts().sort_index())
    
    # Get regime predictions for logging (optional, for results CSV)
    regime_pred = regime_detector.predict(features)
    
    # 6. Compute future returns for PnL calculation using new label system
    # Use horizon=1 for actual trade PnL (next bar return), but also compute
    # labels with the configured horizon for consistency
    print("\nComputing future returns...")
    # For PnL calculation: use 1-bar forward return
    df_with_returns = make_direction_labels(df, horizon_bars=1, label_threshold=LABEL_THRESHOLD, 
                                            smoothing_window=1, debug=False)
    future_returns = df_with_returns['future_return'].reindex(signals.index)
    
    # 7. Trading logic: signal -> position
    # signal = 1 → long (position = 1)
    # signal = -1 → short (position = -1)
    # signal = 0 → flat (position = 0)
    positions = signals.copy()
    positions.name = 'position'
    
    # 8. Compute raw strategy returns (before cost)
    # Ensure all Series are aligned on signals.index
    strategy_returns = positions * future_returns
    strategy_returns.name = 'strategy_returns'
    
    # 9. Apply transaction costs per-trade (in same units as returns)
    # Compute round-trip cost in fractional form
    fee_frac = fee_bps / 10000.0
    slip_frac = slippage_bps / 10000.0
    round_trip_cost = 2.0 * (fee_frac + slip_frac)  # Entry + exit
    
    # Apply cost on every bar where we have a trade (signal != 0)
    # This treats each bar with non-zero signal as a full round-trip
    trade_mask = positions != 0
    cost_per_trade = round_trip_cost
    
    # Apply cost: subtract cost from returns on bars where we have a trade
    strategy_returns_net = strategy_returns.copy()
    strategy_returns_net[trade_mask] = strategy_returns_net[trade_mask] - cost_per_trade
    strategy_returns_net.name = 'strategy_returns_net'
    
    # Use net returns for PnL calculation
    pnl = strategy_returns_net
    pnl.name = 'pnl'
    
    # Keep raw returns for sanity checks
    pnl_raw = strategy_returns.copy()
    pnl_raw.name = 'pnl_raw'
    
    # Remove NaN rows (where future_return is NaN due to being at the end)
    valid_mask = ~(pnl.isna() | positions.isna() | future_returns.isna())
    positions = positions[valid_mask]
    pnl = pnl[valid_mask]
    pnl_raw = pnl_raw[valid_mask]
    future_returns = future_returns[valid_mask]
    trade_mask = trade_mask[valid_mask]
    
    # Compute total costs for informational purposes only (after filtering)
    num_trades = trade_mask.sum()
    total_transaction_costs = num_trades * cost_per_trade
    
    # 10. Compute cumulative equity curve from net returns
    # Start with initial capital of 1.0, then compound returns
    equity = (1 + pnl).cumprod()
    equity.name = 'equity'
    
    # 11. Compute metrics
    print("\nComputing backtest metrics...")
    metrics = compute_metrics(pnl, equity, positions)
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"Long Accuracy: {metrics['long_accuracy']:.2%}")
    print(f"Short Accuracy: {metrics['short_accuracy']:.2%}")
    print(f"\nNet of costs: fee = {fee_bps} bps/side, slippage = {slippage_bps} bps/side")
    print(f"Total round-trip cost: {total_cost_bps} bps ({total_cost_bps/100:.2f}%)")
    print(f"Total transaction costs (approx): {total_transaction_costs:.6f}")
    print(f"\nTotal Trades: {num_trades}")
    print(f"Long Trades: {(positions > 0).sum()}")
    print(f"Short Trades: {(positions < 0).sum()}")
    print(f"Flat Periods: {(positions == 0).sum()}")
    
    # 11.5. Build detailed breakdown DataFrame
    print("\nBuilding detailed breakdown...")
    # Align all data to the same index (signals.index after valid_mask)
    bt = pd.DataFrame(index=positions.index)
    bt['regime'] = regime_pred.reindex(positions.index)
    bt['signal'] = positions
    bt['future_return'] = future_returns
    bt['trade_pnl'] = pnl
    bt['trade_pnl_raw'] = pnl_raw  # Gross returns before costs
    bt['final_signal'] = signals.reindex(positions.index).fillna(0)
    bt['direction_confidence'] = direction_confidence.reindex(positions.index).fillna(0.0)
    bt['ev'] = ev_series.reindex(positions.index)
    # Only non-zero where a trade is taken
    bt.loc[bt['signal'] == 0, 'trade_pnl'] = 0.0
    bt.loc[bt['signal'] == 0, 'trade_pnl_raw'] = 0.0
    
    # Compute per-side stats
    print("\n" + "="*60)
    print("SIDE-BY-SIDE STATS")
    print("="*60)
    
    long_mask = bt['signal'] == 1
    short_mask = bt['signal'] == -1
    
    for side_name, mask in [('LONG', long_mask), ('SHORT', short_mask)]:
        if mask.sum() == 0:
            print(f"\n{side_name}:")
            print(f"  No trades")
            continue
        
        side_pnl = bt.loc[mask, 'trade_pnl']
        side_equity = (1 + side_pnl).cumprod()
        
        total_return = side_equity.iloc[-1] / side_equity.iloc[0] - 1.0 if len(side_equity) > 0 else 0.0
        sharpe = compute_sharpe(side_pnl)
        
        # Max drawdown
        running_max = side_equity.expanding().max()
        drawdown = (side_equity - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
        
        hit_rate = (side_pnl > 0).mean() if len(side_pnl) > 0 else 0.0
        trade_count = mask.sum()
        
        print(f"\n{side_name}:")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Hit Rate: {hit_rate:.2%}")
        print(f"  Number of Trades: {trade_count}")
    
    # Compute per-regime stats
    print("\n" + "="*60)
    print("PER-REGIME STATS")
    print("="*60)
    
    regime_stats = []
    for regime_name in ['trend_up', 'trend_down', 'chop']:
        regime_mask = (bt['regime'] == regime_name)
        bars_in_regime = regime_mask.sum()
        
        if bars_in_regime == 0:
            continue
        
        trades_in_regime = ((bt['signal'] != 0) & regime_mask).sum()
        long_trades = ((bt['signal'] == 1) & regime_mask).sum()
        short_trades = ((bt['signal'] == -1) & regime_mask).sum()
        
        # PnL of trades in this regime
        regime_trade_pnl = bt.loc[regime_mask & (bt['signal'] != 0), 'trade_pnl']
        total_pnl = regime_trade_pnl.sum() if len(regime_trade_pnl) > 0 else 0.0
        hit_rate = (regime_trade_pnl > 0).mean() if len(regime_trade_pnl) > 0 else 0.0
        
        # Return of trades in this regime (cumulative)
        if len(regime_trade_pnl) > 0:
            regime_equity = (1 + regime_trade_pnl).cumprod()
            regime_return = regime_equity.iloc[-1] / regime_equity.iloc[0] - 1.0 if len(regime_equity) > 0 else 0.0
        else:
            regime_return = 0.0
        
        regime_stats.append({
            'regime': regime_name,
            'bars': bars_in_regime,
            'trades': trades_in_regime,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'total_pnl': total_pnl,
            'hit_rate': hit_rate
        })
        
        print(f"\n{regime_name.upper()}:")
        print(f"  Bars in regime: {bars_in_regime}")
        print(f"  Trades taken: {trades_in_regime}")
        print(f"  Long trades: {long_trades}")
        print(f"  Short trades: {short_trades}")
        print(f"  Return: {regime_return:.2%}")
        print(f"  Total PnL: {total_pnl:.6f}")
        print(f"  Hit Rate: {hit_rate:.2%}")
    
    # Compute per-regime + side stats
    print("\n" + "="*60)
    print("PER-REGIME + SIDE STATS")
    print("="*60)
    
    # Filter to only trade bars (signal != 0)
    trade_bt = bt[bt['signal'] != 0].copy()
    
    if len(trade_bt) > 0:
        # Group by regime and signal
        grouped = trade_bt.groupby(['regime', 'signal'])
        
        breakdown_stats = []
        for (regime_name, signal_val), group in grouped:
            signal_name = 'long' if signal_val == 1 else 'short'
            trade_count = len(group)
            hit_rate = (group['trade_pnl'] > 0).mean() if trade_count > 0 else 0.0
            avg_trade_return = group['trade_pnl'].mean() if trade_count > 0 else 0.0
            total_pnl = group['trade_pnl'].sum()
            
            breakdown_stats.append({
                'regime': regime_name,
                'side': signal_name,
                'trade_count': trade_count,
                'hit_rate': hit_rate,
                'avg_trade_return': avg_trade_return,
                'total_pnl': total_pnl
            })
            
            print(f"\n({regime_name}, {signal_name}):")
            print(f"  Trade Count: {trade_count}")
            print(f"  Hit Rate: {hit_rate:.2%}")
            print(f"  Avg Trade Return: {avg_trade_return:.6f}")
            print(f"  Total PnL: {total_pnl:.6f}")
        
        # Create summary DataFrame
        breakdown_df = pd.DataFrame(breakdown_stats)
    else:
        breakdown_df = pd.DataFrame()
        print("\nNo trades to analyze")
    
    # 12. Transaction-cost-aware diagnostics
    print("\n" + "="*60)
    print("TRANSACTION-COST-AWARE DIAGNOSTICS")
    print("="*60)
    
    # Filter to only trade bars for gross return analysis
    trade_bt_gross = bt[bt['signal'] != 0].copy()
    
    if len(trade_bt_gross) > 0:
        # Overall gross return statistics (before costs)
        gross_returns = trade_bt_gross['trade_pnl_raw']
        avg_gross = gross_returns.mean()
        median_gross = gross_returns.median()
        percentiles = gross_returns.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        
        print("\n[OVERALL] Gross Return Statistics (before costs):")
        print(f"  Number of trades: {len(gross_returns)}")
        print(f"  Average gross return per trade: {avg_gross:.6f} ({avg_gross*100:.4f}%)")
        print(f"  Median gross return per trade: {median_gross:.6f} ({median_gross*100:.4f}%)")
        print(f"  Distribution percentiles:")
        print(f"    5th:  {percentiles[0.05]:.6f} ({percentiles[0.05]*100:.4f}%)")
        print(f"    25th: {percentiles[0.25]:.6f} ({percentiles[0.25]*100:.4f}%)")
        print(f"    50th: {percentiles[0.50]:.6f} ({percentiles[0.50]*100:.4f}%)")
        print(f"    75th: {percentiles[0.75]:.6f} ({percentiles[0.75]*100:.4f}%)")
        print(f"    95th: {percentiles[0.95]:.6f} ({percentiles[0.95]*100:.4f}%)")
        
        # Cost comparison
        print(f"\n[COST ANALYSIS] Round-trip cost: {cost_per_trade:.6f} ({cost_per_trade*100:.4f}%)")
        if avg_gross != 0:
            cost_avg_ratio = cost_per_trade / abs(avg_gross)
            print(f"  cost / avg_gross = {cost_avg_ratio:.2f}x")
            if cost_avg_ratio > 1.0:
                print(f"  ⚠️  WARNING: Cost exceeds average gross return!")
            elif cost_avg_ratio > 0.5:
                print(f"  ⚠️  CAUTION: Cost is >50% of average gross return")
            else:
                print(f"  ✓ Cost is manageable relative to average gross return")
        else:
            print(f"  cost / avg_gross = N/A (avg_gross is zero)")
        
        if median_gross != 0:
            cost_median_ratio = cost_per_trade / abs(median_gross)
            print(f"  cost / median_gross = {cost_median_ratio:.2f}x")
            if cost_median_ratio > 1.0:
                print(f"  ⚠️  WARNING: Cost exceeds median gross return!")
            elif cost_median_ratio > 0.5:
                print(f"  ⚠️  CAUTION: Cost is >50% of median gross return")
            else:
                print(f"  ✓ Cost is manageable relative to median gross return")
        else:
            print(f"  cost / median_gross = N/A (median_gross is zero)")
        
        # Per-side gross return statistics
        print("\n[PER-SIDE] Gross Return Statistics (before costs):")
        for side_name, side_val in [('LONG', 1), ('SHORT', -1)]:
            side_mask = trade_bt_gross['signal'] == side_val
            if side_mask.sum() == 0:
                print(f"\n  {side_name}: No trades")
                continue
            
            side_gross = trade_bt_gross.loc[side_mask, 'trade_pnl_raw']
            side_avg = side_gross.mean()
            side_median = side_gross.median()
            side_percentiles = side_gross.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            
            print(f"\n  {side_name}:")
            print(f"    Trades: {len(side_gross)}")
            print(f"    Avg gross: {side_avg:.6f} ({side_avg*100:.4f}%)")
            print(f"    Median gross: {side_median:.6f} ({side_median*100:.4f}%)")
            print(f"    Percentiles: 5th={side_percentiles[0.05]:.6f}, 25th={side_percentiles[0.25]:.6f}, "
                  f"50th={side_percentiles[0.50]:.6f}, 75th={side_percentiles[0.75]:.6f}, 95th={side_percentiles[0.95]:.6f}")
            
            if side_avg != 0:
                side_cost_avg_ratio = cost_per_trade / abs(side_avg)
                print(f"    cost / avg_gross = {side_cost_avg_ratio:.2f}x", end="")
                if side_cost_avg_ratio > 1.0:
                    print(" ⚠️")
                elif side_cost_avg_ratio > 0.5:
                    print(" ⚠️")
                else:
                    print(" ✓")
        
        # Per-regime + side gross return statistics
        print("\n[PER-REGIME+SIDE] Gross Return Statistics (before costs):")
        for regime_name in ['trend_up', 'trend_down', 'chop']:
            regime_mask = trade_bt_gross['regime'] == regime_name
            if not regime_mask.any():
                continue
            
            for side_name, side_val in [('long', 1), ('short', -1)]:
                combo_mask = regime_mask & (trade_bt_gross['signal'] == side_val)
                if not combo_mask.any():
                    continue
                
                combo_gross = trade_bt_gross.loc[combo_mask, 'trade_pnl_raw']
                combo_avg = combo_gross.mean()
                combo_median = combo_gross.median()
                combo_percentiles = combo_gross.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
                
                print(f"\n  ({regime_name}, {side_name}):")
                print(f"    Trades: {len(combo_gross)}")
                print(f"    Avg gross: {combo_avg:.6f} ({combo_avg*100:.4f}%)")
                print(f"    Median gross: {combo_median:.6f} ({combo_median*100:.4f}%)")
                print(f"    Percentiles: 5th={combo_percentiles[0.05]:.6f}, 25th={combo_percentiles[0.25]:.6f}, "
                      f"50th={combo_percentiles[0.50]:.6f}, 75th={combo_percentiles[0.75]:.6f}, 95th={combo_percentiles[0.95]:.6f}")
                
                if combo_avg != 0:
                    combo_cost_avg_ratio = cost_per_trade / abs(combo_avg)
                    print(f"    cost / avg_gross = {combo_cost_avg_ratio:.2f}x", end="")
                    if combo_cost_avg_ratio > 1.0:
                        print(" ⚠️  WARNING: Cost exceeds average gross return!")
                    elif combo_cost_avg_ratio > 0.5:
                        print(" ⚠️  CAUTION: Cost is >50% of average gross return")
                    else:
                        print(" ✓ Cost is manageable")
    else:
        print("\nNo trades to analyze for cost diagnostics")
    
    print("="*60)
    
    # 12.5. Probability calibration report
    print("\n" + "="*60)
    print("PROBABILITY CALIBRATION REPORT")
    print("="*60)
    
    # Prepare calibration data
    calibration_df = bt[['direction_confidence', 'final_signal', 'future_return', 'regime']].copy()
    
    # Generate calibration reports
    calibration_reports = generate_calibration_report(
        calibration_df,
        prob_col='direction_confidence',
        signal_col='final_signal',
        return_col='future_return',
        regime_col='regime'
    )
    
    # Print overall calibration
    if len(calibration_reports['overall']) > 0:
        print("\n[OVERALL] Calibration by Probability Bin:")
        print(calibration_reports['overall'].to_string(index=False))
    
    # Print per-side calibration
    if len(calibration_reports['per_side']) > 0:
        print("\n[PER-SIDE] Calibration by Probability Bin:")
        for side in ['long', 'short']:
            side_data = calibration_reports['per_side'][calibration_reports['per_side']['side_str'] == side]
            if len(side_data) > 0:
                print(f"\n  {side.upper()}:")
                print(side_data[['prob_bin', 'count', 'hit_rate', 'avg_gross_return', 'median_gross_return', 'avg_abs_return']].to_string(index=False))
    
    # Print per-regime+side calibration (only for combos with trades)
    if len(calibration_reports['per_regime_side']) > 0:
        print("\n[PER-REGIME+SIDE] Calibration by Probability Bin:")
        for (regime, side), group in calibration_reports['per_regime_side'].groupby(['regime', 'side_str']):
            if len(group) > 0:
                print(f"\n  ({regime}, {side}):")
                print(group[['prob_bin', 'count', 'hit_rate', 'avg_gross_return', 'median_gross_return', 'avg_abs_return']].to_string(index=False))
    
    # Build suffix for calibration filename
    cal_suffix_parts = []
    if p_long is not None:
        cal_suffix_parts.append(f"plong{p_long:.2f}")
    if p_short is not None:
        cal_suffix_parts.append(f"pshort{p_short:.2f}")
    if args.recent is not None:
        cal_suffix_parts.append(f"r{args.recent}")
    if args.profile:
        cal_suffix_parts.append(args.profile)
    cal_suffix = "_" + "_".join(cal_suffix_parts) if cal_suffix_parts else ""
    
    # Save overall calibration
    if len(calibration_reports['overall']) > 0:
        cal_overall_path = results_dir / f"calibration_overall{cal_suffix}.csv"
        calibration_reports['overall'].to_csv(cal_overall_path, index=False)
        print(f"\nOverall calibration saved to {cal_overall_path}")
    
    # Save per-side calibration
    if len(calibration_reports['per_side']) > 0:
        cal_side_path = results_dir / f"calibration_per_side{cal_suffix}.csv"
        calibration_reports['per_side'].to_csv(cal_side_path, index=False)
        print(f"Per-side calibration saved to {cal_side_path}")
    
    # Save per-regime+side calibration
    if len(calibration_reports['per_regime_side']) > 0:
        cal_regime_side_path = results_dir / f"calibration_per_regime_side{cal_suffix}.csv"
        calibration_reports['per_regime_side'].to_csv(cal_regime_side_path, index=False)
        print(f"Per-regime+side calibration saved to {cal_regime_side_path}")
    
    print("="*60)
    
    # 13. Cost sanity check
    print("\n" + "="*60)
    print("[DEBUG] Cost sanity check")
    print("="*60)
    raw_total_pnl = pnl_raw.sum()
    net_total_pnl = pnl.sum()
    print(f"  Num trades: {num_trades}")
    print(f"  Cost per trade (fraction): {cost_per_trade:.6f}")
    print(f"  Approx total cost: {total_transaction_costs:.6f}")
    print(f"  Raw total PnL (no cost): {raw_total_pnl:.6f}")
    print(f"  Net total PnL (with cost): {net_total_pnl:.6f}")
    print("="*60)
    
    # 14. Plot equity curve
    print("\nPlotting equity curve...")
    # Build suffix for result filenames based on args
    suffix_parts = []
    if prob_threshold is not None:
        suffix_parts.append(f"p{prob_threshold:.2f}")
    if args.recent is not None:
        suffix_parts.append(f"r{args.recent}")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Equity curve
    axes[0].plot(equity.index, equity.values, label='Equity Curve', linewidth=1.5)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break Even')
    axes[0].set_ylabel('Equity')
    axes[0].set_title('Backtest Equity Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red', label='Drawdown')
    axes[1].set_ylabel('Drawdown')
    axes[1].set_xlabel('Time')
    axes[1].set_title('Drawdown')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    equity_curve_path = results_dir / f"equity_curve{suffix}.png"
    plt.savefig(equity_curve_path, dpi=150, bbox_inches='tight')
    print(f"Equity curve saved to {equity_curve_path}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'position': positions,
        'future_return': future_returns,
        'pnl': pnl,
        'equity': equity,
        'regime': regime_pred.reindex(positions.index)
    })
    results_csv_path = results_dir / f"backtest_results{suffix}.csv"
    results_df.to_csv(results_csv_path)
    print(f"Results saved to {results_csv_path}")
    
    # Save breakdown CSV
    breakdown_csv_path = results_dir / f"backtest_breakdown{suffix}.csv"
    bt.to_csv(breakdown_csv_path)
    print(f"Breakdown saved to {breakdown_csv_path}")
    
    # Save EV details CSV (only for trades)
    trade_mask_ev = bt['final_signal'] != 0
    if trade_mask_ev.any():
        ev_details = pd.DataFrame({
            'timestamp': bt.index[trade_mask_ev],
            'regime': bt.loc[trade_mask_ev, 'regime'],
            'side': bt.loc[trade_mask_ev, 'final_signal'].map({1: 'long', -1: 'short', 0: 'flat'}),
            'prob': bt.loc[trade_mask_ev, 'direction_confidence'],
            'ev': bt.loc[trade_mask_ev, 'ev'],
            'future_return': bt.loc[trade_mask_ev, 'future_return']
        })
        ev_details_path = results_dir / f"trade_ev_details{suffix}.csv"
        ev_details.to_csv(ev_details_path, index=False)
        print(f"EV details saved to {ev_details_path}")
    
    # 15. EV Analysis Debug Mode
    if args.dump_ev_analysis:
        # Check if we have valid EV data
        ev_valid_count = ev_series.dropna().count()
        if ev_valid_count > 0:
            print("\n" + "="*60)
            print("RUNNING EV ANALYSIS (--dump-ev-analysis)")
            print("="*60)
            
            # Build trade DataFrame for analysis
            ev_analysis_df = pd.DataFrame({
                'signal': bt['signal'],
                'future_return': bt['future_return'],
                'ev': bt['ev'],
                'prob': bt['direction_confidence'],
                'regime': bt['regime']
            })
            
            # Run EV diagnostics
            ev_stats = final_signal_gen.compute_ev_diagnostics(
                trade_df=ev_analysis_df,
                output_dir=results_dir,
                min_ev_threshold=args.min_ev
            )
        else:
            print("\n[EV Analysis] Skipped: No valid EV data available.")
            print("  To enable EV analysis, ensure a calibration CSV is available.")
            print("  Use --calibration-csv or let auto-detection find one in results/")
    
    print("\nBacktest complete!")


if __name__ == "__main__":
    main()

