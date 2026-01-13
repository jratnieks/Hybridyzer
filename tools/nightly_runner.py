"""
Nightly training runner for Hybridyzer.
Runs repeated walk-forward training with ablations and keeps the best result.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "train.py"
MODELS_DIR = ROOT / "models"


def _parse_int_list(raw: str) -> List[int]:
    items: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    return items


def _parse_str_list(raw: str) -> List[str]:
    items: List[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(chunk)
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


def _load_results(results_path: Path) -> Dict[str, float]:
    df = pd.read_csv(results_path)
    returns = pd.to_numeric(df.get("cumulative_return"), errors="coerce").dropna()
    drawdowns = pd.to_numeric(df.get("max_drawdown"), errors="coerce").dropna()
    metrics = {
        "windows": float(len(df)),
        "avg_return": float(returns.mean()) if not returns.empty else float("nan"),
        "median_return": float(returns.median()) if not returns.empty else float("nan"),
        "trimmed_mean_return": _trimmed_mean(returns.values) if not returns.empty else float("nan"),
        "worst_drawdown": float(drawdowns.min()) if not drawdowns.empty else float("nan"),
    }
    return metrics


def _copy_if_exists(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _arm_key(cfg: Dict[str, object]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _init_bandit_state(configs: List[Dict[str, object]], prior_std: float, min_plays: int) -> Dict[str, object]:
    arms = {}
    for cfg in configs:
        key = _arm_key(cfg)
        if key in arms:
            continue
        arms[key] = {
            "config": cfg,
            "n": 0,
            "mean": 0.0,
            "m2": 0.0,
        }
    return {
        "policy": "thompson",
        "prior_std": prior_std,
        "min_plays": min_plays,
        "arms": arms,
    }


def _load_bandit_state(path: Path, configs: List[Dict[str, object]], prior_std: float, min_plays: int) -> Dict[str, object]:
    if not path.exists():
        return _init_bandit_state(configs, prior_std, min_plays)
    state = json.loads(path.read_text(encoding="utf-8"))
    arms = state.get("arms", {})
    for cfg in configs:
        key = _arm_key(cfg)
        if key not in arms:
            arms[key] = {
                "config": cfg,
                "n": 0,
                "mean": 0.0,
                "m2": 0.0,
            }
    state["arms"] = arms
    state["policy"] = "thompson"
    state["prior_std"] = prior_std
    state["min_plays"] = min_plays
    return state


def _save_bandit_state(path: Path, state: Dict[str, object]) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _update_bandit(arm: Dict[str, object], reward: float) -> None:
    n = arm.get("n", 0) + 1
    mean = arm.get("mean", 0.0)
    m2 = arm.get("m2", 0.0)
    delta = reward - mean
    mean += delta / n
    delta2 = reward - mean
    m2 += delta * delta2
    arm["n"] = n
    arm["mean"] = mean
    arm["m2"] = m2


def _sample_thompson(arm: Dict[str, object], prior_std: float, rng: random.Random) -> float:
    n = arm.get("n", 0)
    mean = arm.get("mean", 0.0)
    m2 = arm.get("m2", 0.0)
    if n > 1:
        variance = m2 / (n - 1)
        std = max(np.sqrt(variance), prior_std)
    else:
        std = prior_std
    scale = std / np.sqrt(n + 1)
    return rng.gauss(mean, scale)


def _choose_bandit_arm(state: Dict[str, object], rng: random.Random) -> Tuple[str, Dict[str, object]]:
    min_plays = int(state.get("min_plays", 1))
    prior_std = float(state.get("prior_std", 0.05))
    arms = state["arms"]
    underplayed = [k for k, v in arms.items() if v.get("n", 0) < min_plays]
    if underplayed:
        key = rng.choice(underplayed)
        return key, arms[key]
    samples = {}
    for key, arm in arms.items():
        samples[key] = _sample_thompson(arm, prior_std, rng)
    best_key = max(samples, key=samples.get)
    return best_key, arms[best_key]


def _compute_reward(score: float, worst_drawdown: float, max_drawdown: float, dd_penalty: float) -> float:
    if np.isnan(score):
        return float("nan")
    if np.isnan(worst_drawdown):
        return float(score)
    excess = max(0.0, abs(worst_drawdown) - max_drawdown)
    return float(score) - dd_penalty * excess


def _build_configs(
    horizons: List[int],
    smoothings: List[int],
    modules: List[str],
    include_feature_ablations: bool,
    include_module_ablations: bool,
    max_runs: Optional[int],
    random_feature_pool: Optional[List[str]],
    random_feature_subsets: int,
    random_feature_min: int,
    random_feature_max: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []

    def add_config(cfg: Dict[str, object]) -> None:
        if max_runs is not None and len(configs) >= max_runs:
            return
        configs.append(cfg)

    feature_flags = [
        ("disable_ml_features", True),
        ("disable_regime_context", True),
        ("disable_signal_dynamics", True),
        ("disable_rolling_stats", True),
    ]

    for horizon in horizons:
        for smoothing in smoothings:
            add_config({"horizon_bars": horizon, "smoothing_window": smoothing})

    if include_feature_ablations:
        for horizon in horizons:
            for smoothing in smoothings:
                for flag, value in feature_flags:
                    add_config({
                        "horizon_bars": horizon,
                        "smoothing_window": smoothing,
                        flag: value,
                    })

    if include_module_ablations:
        for horizon in horizons:
            for smoothing in smoothings:
                for module in modules:
                    add_config({
                        "horizon_bars": horizon,
                        "smoothing_window": smoothing,
                        "disable_modules": [module],
                    })

    if random_feature_subsets > 0 and random_feature_pool:
        pool = [p for p in random_feature_pool if p]
        max_k = min(random_feature_max, len(pool))
        min_k = min(random_feature_min, max_k)
        for _ in range(random_feature_subsets):
            if max_runs is not None and len(configs) >= max_runs:
                break
            horizon = rng.choice(horizons)
            smoothing = rng.choice(smoothings)
            k = rng.randint(min_k, max_k) if max_k > 0 else 0
            subset = rng.sample(pool, k) if k > 0 else []
            add_config({
                "horizon_bars": horizon,
                "smoothing_window": smoothing,
                "exclude_features": subset,
            })

    return configs


def _build_train_command(args: argparse.Namespace, cfg: Dict[str, object], train_log_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--walkforward",
        "--train-months",
        str(args.train_months),
        "--val-months",
        str(args.val_months),
        "--slide-months",
        str(args.slide_months),
        "--horizon-bars",
        str(cfg["horizon_bars"]),
        "--smoothing-window",
        str(cfg["smoothing_window"]),
        "--label-threshold",
        str(args.label_threshold),
        "--fee-bps",
        str(args.fee_bps),
        "--eval-mode",
        args.eval_mode,
        "--regime-labels",
        args.regime_labels,
        "--seed",
        str(args.seed),
        "--log-file",
        str(train_log_path),
    ]

    if args.calibration_method:
        cmd.extend(["--calibration-method", args.calibration_method])
    if args.alpha is not None:
        cmd.extend(["--alpha", str(args.alpha)])
    if args.disable_calibration:
        cmd.append("--disable-calibration")
    if args.calibration_source:
        cmd.extend(["--calibration-source", args.calibration_source])
    if args.cpu_only:
        cmd.append("--cpu-only")
    if args.runpod:
        cmd.append("--runpod")
    if args.use_full_pipeline:
        cmd.append("--use-full-pipeline")
    if args.embargo_days:
        cmd.extend(["--embargo-days", str(args.embargo_days)])
    if args.purge_bars:
        cmd.extend(["--purge-bars", str(args.purge_bars)])
    if args.purge_horizon:
        cmd.append("--purge-horizon")

    include_features = list(getattr(args, "include_features_list", []) or [])
    exclude_features = list(getattr(args, "exclude_features_list", []) or [])
    include_modules = list(getattr(args, "include_modules_list", []) or [])

    if cfg.get("include_features"):
        include_features.extend(cfg["include_features"])
    if cfg.get("exclude_features"):
        exclude_features.extend(cfg["exclude_features"])
    if cfg.get("include_modules"):
        include_modules.extend(cfg["include_modules"])

    if include_modules:
        cmd.extend(["--include-modules", ",".join(sorted(set(include_modules)))])
    if include_features:
        cmd.extend(["--include-features", ",".join(sorted(set(include_features)))])
    if exclude_features:
        cmd.extend(["--exclude-features", ",".join(sorted(set(exclude_features)))])

    if cfg.get("disable_ml_features"):
        cmd.append("--disable-ml-features")
    if cfg.get("disable_regime_context"):
        cmd.append("--disable-regime-context")
    if cfg.get("disable_signal_dynamics"):
        cmd.append("--disable-signal-dynamics")
    if cfg.get("disable_rolling_stats"):
        cmd.append("--disable-rolling-stats")
    if cfg.get("disable_modules"):
        cmd.extend(["--disable-modules", ",".join(cfg["disable_modules"])])

    return cmd


def _run_train(cmd: List[str], timeout_seconds: Optional[int], runner_log_path: Path) -> Tuple[int, Optional[str]]:
    runner_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(runner_log_path, "w", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
        return proc.returncode, None
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def _log_contains_gpu_error(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    indicators = ["CUDA", "cuML", "out of memory", "CUBLAS", "CUDNN"]
    return any(token in text for token in indicators)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly training runner")
    parser.add_argument("--time-budget-hours", type=float, default=8.0,
                        help="Max wall-clock time for the nightly run (default: 8)")
    parser.add_argument("--max-runs", type=int, default=20,
                        help="Maximum number of runs (default: 20)")
    parser.add_argument("--max-drawdown", type=float, default=0.30,
                        help="Reject runs with abs(max_drawdown) above this value (default: 0.30)")
    parser.add_argument("--promote-best", action="store_true",
                        help="Copy best run artifacts back into models/ at the end")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: results/nightly/<timestamp>)")

    parser.add_argument("--horizons", type=str, default="12,24",
                        help="Comma-separated horizon bars list (default: 12,24)")
    parser.add_argument("--smoothings", type=str, default="8,12",
                        help="Comma-separated smoothing windows list (default: 8,12)")
    parser.add_argument("--module-list", type=str, default="superma,trendmagic,pvt",
                        help="Comma-separated module list for ablations")
    parser.add_argument("--no-feature-ablations", action="store_true",
                        help="Skip feature-group ablations")
    parser.add_argument("--no-module-ablations", action="store_true",
                        help="Skip module ablations")
    parser.add_argument("--include-modules", type=str, default=None,
                        help="Comma-separated module whitelist (others disabled)")
    parser.add_argument("--include-features", type=str, default=None,
                        help="Comma-separated regex patterns of features to keep")
    parser.add_argument("--exclude-features", type=str, default=None,
                        help="Comma-separated regex patterns of features to drop")
    parser.add_argument("--feature-pool", type=str, default=None,
                        help="Comma-separated regex pool for random feature exclusions")
    parser.add_argument("--random-feature-subsets", type=int, default=0,
                        help="Number of random feature exclusion configs to add")
    parser.add_argument("--random-feature-min", type=int, default=1,
                        help="Min patterns to exclude per random config (default: 1)")
    parser.add_argument("--random-feature-max", type=int, default=3,
                        help="Max patterns to exclude per random config (default: 3)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for random feature subsets")
    parser.add_argument("--bandit-thompson", action="store_true",
                        help="Use Thompson sampling to select configs")
    parser.add_argument("--bandit-min-plays", type=int, default=1,
                        help="Minimum plays per arm before sampling (default: 1)")
    parser.add_argument("--bandit-prior-std", type=float, default=0.05,
                        help="Prior stddev for Thompson sampling (default: 0.05)")
    parser.add_argument("--bandit-dd-penalty", type=float, default=0.5,
                        help="Drawdown penalty for bandit reward (default: 0.5)")

    parser.add_argument("--train-months", type=int, default=6,
                        help="Walk-forward training window size in months (default: 6)")
    parser.add_argument("--val-months", type=int, default=1,
                        help="Walk-forward validation window size in months (default: 1)")
    parser.add_argument("--slide-months", type=int, default=1,
                        help="Walk-forward slide step in months (default: 1)")
    parser.add_argument("--label-threshold", type=float, default=0.0005,
                        help="Label threshold (default: 0.0005)")
    parser.add_argument("--fee-bps", type=float, default=1.0,
                        help="Transaction cost per side in bps (default: 1.0)")
    parser.add_argument("--eval-mode", type=str, default="nonoverlap", choices=["nonoverlap", "per-bar"],
                        help="Evaluation mode (default: nonoverlap)")
    parser.add_argument("--regime-labels", type=str, default="indicator", choices=["indicator", "rule"],
                        help="Regime label strategy (default: indicator)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--calibration-method", type=str, default="isotonic",
                        choices=["isotonic", "platt"], help="Calibration method")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Sharpening alpha (default: 2.0)")
    parser.add_argument("--disable-calibration", action="store_true",
                        help="Disable probability calibration")
    parser.add_argument("--calibration-source", type=str, default="train",
                        choices=["train", "val"], help="Calibration data source")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU backend")
    parser.add_argument("--auto-cpu-fallback", action="store_true",
                        help="Retry once on CPU if GPU run fails")
    parser.add_argument("--runpod", action="store_true",
                        help="Use RunPod paths (/workspace/Hybridyzer)")
    parser.add_argument("--use-full-pipeline", action="store_true",
                        help="Use full training pipeline with feature pruning/diagnostics")
    parser.add_argument("--embargo-days", type=int, default=0,
                        help="Embargo days between train/val windows")
    parser.add_argument("--purge-bars", type=int, default=0,
                        help="Bars to purge from training window")
    parser.add_argument("--purge-horizon", action="store_true",
                        help="Purge horizon bars from end of training window")

    args = parser.parse_args()

    horizons = _parse_int_list(args.horizons)
    smoothings = _parse_int_list(args.smoothings)
    modules = _parse_str_list(args.module_list)
    args.include_modules_list = _parse_str_list(args.include_modules) if args.include_modules else []
    args.include_features_list = _parse_str_list(args.include_features) if args.include_features else []
    args.exclude_features_list = _parse_str_list(args.exclude_features) if args.exclude_features else []
    feature_pool_list = _parse_str_list(args.feature_pool) if args.feature_pool else []
    rng = random.Random(args.random_seed)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.output_dir) if args.output_dir else (ROOT / "results" / "nightly" / timestamp)
    base_out.mkdir(parents=True, exist_ok=True)

    config_limit = None if args.bandit_thompson else args.max_runs
    configs = _build_configs(
        horizons=horizons,
        smoothings=smoothings,
        modules=modules,
        include_feature_ablations=not args.no_feature_ablations,
        include_module_ablations=not args.no_module_ablations,
        max_runs=config_limit,
        random_feature_pool=feature_pool_list,
        random_feature_subsets=args.random_feature_subsets,
        random_feature_min=args.random_feature_min,
        random_feature_max=args.random_feature_max,
        rng=rng,
    )

    print(f"[Runner] Output: {base_out}")
    print(f"[Runner] Configs queued: {len(configs)}")
    print(f"[Runner] Time budget: {args.time_budget_hours:.2f} hours")
    if args.random_feature_subsets > 0:
        print(f"[Runner] Random feature subsets: {args.random_feature_subsets}")
    if not configs:
        raise ValueError("No configs generated; check horizons/smoothings/pools.")

    bandit_state_path = base_out / "bandit_state.json"
    bandit_state = None
    if args.bandit_thompson:
        bandit_state = _load_bandit_state(
            bandit_state_path,
            configs,
            prior_std=args.bandit_prior_std,
            min_plays=args.bandit_min_plays,
        )
        _save_bandit_state(bandit_state_path, bandit_state)
        print(f"[Runner] Bandit mode: Thompson sampling over {len(bandit_state['arms'])} arms")

    time_budget_seconds = int(args.time_budget_hours * 3600)
    deadline = time.time() + time_budget_seconds
    best_score = float("-inf")
    best_summary: Optional[Dict[str, object]] = None

    results_log = base_out / "runs.jsonl"
    total_runs = args.max_runs if args.bandit_thompson else len(configs)
    for idx in range(1, total_runs + 1):
        remaining = int(deadline - time.time())
        if remaining <= 0:
            print("[Runner] Time budget exhausted.")
            break

        arm_key = None
        if args.bandit_thompson:
            arm_key, arm = _choose_bandit_arm(bandit_state, rng)
            cfg = arm["config"]
        else:
            cfg = configs[idx - 1]

        run_dir = base_out / f"run_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        train_log_path = run_dir / "train.log"
        runner_log_path = run_dir / "runner.log"
        cfg_path = run_dir / "config.json"
        cfg_for_file = dict(cfg)
        if args.include_modules_list:
            existing = cfg_for_file.get("include_modules", [])
            cfg_for_file["include_modules"] = sorted(set(existing + args.include_modules_list))
        if args.include_features_list:
            existing = cfg_for_file.get("include_features", [])
            cfg_for_file["include_features"] = sorted(set(existing + args.include_features_list))
        if args.exclude_features_list:
            existing = cfg_for_file.get("exclude_features", [])
            cfg_for_file["exclude_features"] = sorted(set(existing + args.exclude_features_list))
        cfg_path.write_text(json.dumps(cfg_for_file, indent=2, sort_keys=True), encoding="utf-8")

        cmd = _build_train_command(args, cfg, train_log_path)
        pre_mtime = MODELS_DIR.joinpath("training_results.csv").stat().st_mtime if (MODELS_DIR / "training_results.csv").exists() else None

        print(f"[Runner] Run {idx}/{total_runs}: horizon={cfg['horizon_bars']} smoothing={cfg['smoothing_window']}")
        print(f"[Runner] Remaining budget: {remaining // 60} min")

        returncode, reason = _run_train(cmd, timeout_seconds=remaining, runner_log_path=runner_log_path)

        if returncode != 0 and args.auto_cpu_fallback and not args.cpu_only:
            if reason == "timeout" or _log_contains_gpu_error(runner_log_path) or _log_contains_gpu_error(train_log_path):
                print("[Runner] GPU run failed; retrying on CPU.")
                args.cpu_only = True
                cmd = _build_train_command(args, cfg, train_log_path)
                returncode, reason = _run_train(cmd, timeout_seconds=remaining, runner_log_path=runner_log_path)

        status = "ok" if returncode == 0 else f"failed:{reason or returncode}"
        results_path = MODELS_DIR / "training_results.csv"
        post_mtime = results_path.stat().st_mtime if results_path.exists() else None

        if status == "ok" and (pre_mtime is None or post_mtime != pre_mtime) and results_path.exists():
            _copy_if_exists(results_path, run_dir / "training_results.csv")
            _copy_if_exists(MODELS_DIR / "training_manifest.json", run_dir / "training_manifest.json")
            summary = _load_results(results_path)

            drawdown_ok = False
            worst_drawdown = summary.get("worst_drawdown", float("nan"))
            if not np.isnan(worst_drawdown):
                drawdown_ok = abs(worst_drawdown) <= args.max_drawdown

            score = summary.get("trimmed_mean_return", float("nan"))
            reward = _compute_reward(
                score=score,
                worst_drawdown=worst_drawdown,
                max_drawdown=args.max_drawdown,
                dd_penalty=args.bandit_dd_penalty,
            )

            if args.bandit_thompson and bandit_state and arm_key is not None and not np.isnan(reward):
                arm = bandit_state["arms"][arm_key]
                _update_bandit(arm, reward)
                _save_bandit_state(bandit_state_path, bandit_state)

            summary_payload = {
                "status": status,
                "score": score,
                "reward": reward,
                "drawdown_ok": drawdown_ok,
                "config": cfg,
                "metrics": summary,
                "run_dir": str(run_dir),
                "bandit": {"arm_key": arm_key} if args.bandit_thompson else None,
            }

            with open(results_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary_payload) + "\n")

            if drawdown_ok and not np.isnan(score) and score > best_score:
                best_score = score
                best_summary = summary_payload
                print(f"[Runner] New best score: {best_score:.4f}")
                best_dir = base_out / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                _copy_if_exists(run_dir / "training_results.csv", best_dir / "training_results.csv")
                _copy_if_exists(run_dir / "training_manifest.json", best_dir / "training_manifest.json")
                for filename in [
                    "regime_model.pkl",
                    "blender_model.pkl",
                    "blender_direction_model.pkl",
                    "feature_columns.txt",
                    "blend_feature_columns.txt",
                ]:
                    _copy_if_exists(MODELS_DIR / filename, best_dir / filename)
        else:
            summary_payload = {
                "status": status,
                "config": cfg,
                "run_dir": str(run_dir),
                "bandit": {"arm_key": arm_key} if args.bandit_thompson else None,
            }
            with open(results_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary_payload) + "\n")
            print(f"[Runner] Run failed or produced no results: {status}")

    if best_summary:
        best_path = base_out / "best.json"
        best_path.write_text(json.dumps(best_summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[Runner] Best run saved to {best_path}")

        if args.promote_best:
            print("[Runner] Promoting best run artifacts to models/")
            for filename in [
                "regime_model.pkl",
                "blender_model.pkl",
                "blender_direction_model.pkl",
                "feature_columns.txt",
                "blend_feature_columns.txt",
                "training_manifest.json",
            ]:
                _copy_if_exists(base_out / "best" / filename, MODELS_DIR / filename)
            _copy_if_exists(base_out / "best" / "training_results.csv", MODELS_DIR / "training_results.csv")
    else:
        print("[Runner] No valid run met drawdown constraints.")


if __name__ == "__main__":
    main()
