"""
Run a small walk-forward grid over horizon/smoothing on recent data.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_store import FeatureStore
from data.btc_data_loader import load_btc_csv
from train import combined_training, set_global_seed


def _parse_int_list(raw: str) -> List[int]:
    items = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    return items


def _trimmed_mean(values: np.ndarray, lower: float = 0.05, upper: float = 0.95) -> float:
    if values.size == 0:
        return float("nan")
    values = np.sort(values)
    lo = int(len(values) * lower)
    hi = int(len(values) * upper)
    if hi <= lo:
        return float(values.mean())
    return float(values[lo:hi].mean())


def _summarize_results(results_path: Path) -> dict:
    df = pd.read_csv(results_path)
    returns = pd.to_numeric(df.get("cumulative_return"), errors="coerce").dropna()
    gross = pd.to_numeric(df.get("gross_cumulative_return"), errors="coerce").dropna()
    summary = {
        "windows": int(len(df)),
        "avg_return": float(returns.mean()) if not returns.empty else float("nan"),
        "median_return": float(returns.median()) if not returns.empty else float("nan"),
        "trimmed_mean_return": _trimmed_mean(returns.values) if not returns.empty else float("nan"),
        "positive_pct": float((returns > 0).mean()) if not returns.empty else float("nan"),
        "min_return": float(returns.min()) if not returns.empty else float("nan"),
        "max_return": float(returns.max()) if not returns.empty else float("nan"),
        "gross_median_return": float(gross.median()) if not gross.empty else float("nan"),
        "gross_trimmed_mean_return": _trimmed_mean(gross.values) if not gross.empty else float("nan"),
    }
    return summary


def _load_walkforward_data(data_dir: Path) -> Tuple[pd.DataFrame, str]:
    train_path = data_dir / "btcusd_5min_train_2017_2022.csv"
    val_path = data_dir / "btcusd_5min_val_2023.csv"
    if train_path.exists() and val_path.exists():
        df_train_split = load_btc_csv(str(train_path))
        df_val_split = load_btc_csv(str(val_path))
        df = pd.concat([df_train_split, df_val_split]).sort_index()
        return df, f"{train_path.name} + {val_path.name}"

    csv_paths = [
        data_dir / "btcusd_5min.csv",
        data_dir / "btcusd_4H.csv",
        data_dir / "btcusd_1min.csv",
        data_dir / "btc_1m.csv",
        data_dir / "btcusd_1min_volspikes.csv",
        data_dir / "btc_data.csv",
    ]
    for path in csv_paths:
        if path.exists():
            return load_btc_csv(str(path)), path.name

    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        return load_btc_csv(str(csv_files[0])), csv_files[0].name

    raise FileNotFoundError("No CSV files found in data/ directory")


def _apply_date_filter(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    recent_months: int,
) -> pd.DataFrame:
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df.index >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df.index <= end_dt]
    if not start_date and not end_date and recent_months > 0:
        end_dt = df.index.max()
        start_dt = end_dt - pd.DateOffset(months=recent_months)
        df = df[df.index >= start_dt]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search horizon/smoothing on recent data")
    parser.add_argument("--horizons", type=str, default="12,24,36",
                        help="Comma-separated horizon bars list (default: 12,24,36)")
    parser.add_argument("--smoothings", type=str, default="6,8,12",
                        help="Comma-separated smoothing windows list (default: 6,8,12)")
    parser.add_argument("--recent-months", type=int, default=24,
                        help="Use most recent N months if start/end not provided (default: 24)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--train-months", type=int, default=12,
                        help="Walk-forward training window size in months (default: 12)")
    parser.add_argument("--val-months", type=int, default=2,
                        help="Walk-forward validation window size in months (default: 2)")
    parser.add_argument("--slide-months", type=int, default=1,
                        help="Walk-forward slide step in months (default: 1)")
    parser.add_argument("--label-threshold", type=float, default=0.0005,
                        help="Threshold for direction labels (default: 0.0005)")
    parser.add_argument("--regime-labels", type=str, default="indicator", choices=["indicator", "rule"],
                        help="Regime label strategy (default: indicator)")
    parser.add_argument("--eval-mode", type=str, default="nonoverlap", choices=["nonoverlap", "per-bar"],
                        help="Return evaluation mode (default: nonoverlap)")
    parser.add_argument("--fee-bps", type=float, default=1.0,
                        help="Transaction cost per side in bps (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--calibration-method", type=str, default="isotonic", choices=["isotonic", "platt"],
                        help="Calibration method (default: isotonic)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Calibration sharpening alpha (default: 2.0)")
    parser.add_argument("--disable-calibration", action="store_true",
                        help="Disable calibration")
    parser.add_argument("--calibration-source", type=str, default="train", choices=["train", "val"],
                        help="Calibration data source (default: train)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU even if cuML is installed")
    parser.add_argument("--embargo-days", type=int, default=0,
                        help="Embargo gap in days between training and validation (default: 0)")
    parser.add_argument("--purge-bars", type=int, default=0,
                        help="Bars to purge from end of training window (default: 0)")
    parser.add_argument("--purge-horizon", action="store_true",
                        help="Purge horizon_bars from end of training window")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory for grid results (default: results/grid/<timestamp>)")
    args = parser.parse_args()

    horizons = _parse_int_list(args.horizons)
    smoothings = _parse_int_list(args.smoothings)
    if not horizons or not smoothings:
        raise ValueError("Provide at least one horizon and smoothing value.")

    set_global_seed(args.seed)

    try:
        import cudf  # noqa: F401
        import cuml  # noqa: F401
        gpu_available = True
    except ImportError:
        gpu_available = False
    use_gpu = gpu_available and not args.cpu_only

    if args.disable_calibration:
        calibration_method = None
        sharpening_alpha = 1.0
    else:
        calibration_method = args.calibration_method
        sharpening_alpha = max(1.0, min(3.0, args.alpha))

    data_dir = Path.cwd() / "data"
    df, data_path = _load_walkforward_data(data_dir)
    df = _apply_date_filter(df, args.start_date, args.end_date, args.recent_months)
    if df.empty:
        raise ValueError("No data left after date filtering.")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.output_dir) if args.output_dir else (Path.cwd() / "results" / "grid" / timestamp)
    base_out.mkdir(parents=True, exist_ok=True)
    shared_cache = base_out / "cached_features.parquet"

    feature_store = FeatureStore()
    signal_modules = feature_store.signal_modules
    context_modules = feature_store.context_modules

    rows = []
    print(f"[Grid] Data range: {df.index.min()} -> {df.index.max()} ({len(df)} bars)")
    print(f"[Grid] Output: {base_out}")
    print(f"[Grid] Horizons: {horizons}")
    print(f"[Grid] Smoothings: {smoothings}")

    for horizon in horizons:
        for smoothing in smoothings:
            run_id = f"h{horizon}_s{smoothing}"
            models_dir = base_out / run_id / "models"
            results_dir = base_out / run_id / "results"
            models_dir.mkdir(parents=True, exist_ok=True)
            results_dir.mkdir(parents=True, exist_ok=True)

            purge_bars = horizon if args.purge_horizon else args.purge_bars

            print(f"\n[Grid] Running {run_id}...")
            try:
                combined_training(
                    df=df,
                    feature_store=feature_store,
                    signal_modules=signal_modules,
                    context_modules=context_modules,
                    models_dir=models_dir,
                    feature_cache_path=shared_cache,
                    force_recompute_cache=False,
                    train_months=args.train_months,
                    val_months=args.val_months,
                    slide_months=args.slide_months,
                    embargo_days=args.embargo_days,
                    purge_bars=purge_bars,
                    horizon_bars=horizon,
                    label_threshold=args.label_threshold,
                    regime_label_strategy=args.regime_labels,
                    smoothing_window=smoothing,
                    use_gpu=use_gpu,
                    random_state=args.seed,
                    calibration_method=calibration_method,
                    sharpening_alpha=sharpening_alpha,
                    disable_calibration=args.disable_calibration,
                    calibration_source=args.calibration_source,
                    eval_mode=args.eval_mode,
                    transaction_cost_bps=args.fee_bps,
                )
                results_path = models_dir / "training_results.csv"
                summary = _summarize_results(results_path)
                summary.update({
                    "status": "ok",
                    "horizon_bars": horizon,
                    "smoothing_window": smoothing,
                    "results_path": str(results_path),
                    "data_path": data_path,
                })
            except Exception as exc:
                summary = {
                    "status": f"error: {exc}",
                    "horizon_bars": horizon,
                    "smoothing_window": smoothing,
                    "results_path": "",
                    "data_path": data_path,
                }
            rows.append(summary)

    summary_path = base_out / "grid_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Grid] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
