# Agent Context - Hybridyzer

> Single source of truth for decisions, constraints, assumptions, and open questions.
> Last updated: 2026-01-09

---

## Project Overview

Hybridyzer is a hybrid trading system that combines multiple Pine Script strategies into modular Python signal engines with ML-based regime detection and signal blending for BTC trading.

### Core Architecture

```
OHLCV Data → FeatureStore → [RegimeDetector, SignalBlender, DirectionBlender] → FinalSignal
                ↑
    Signal Modules: SuperMA, TrendMagic, PVT Eliminator
    Context Modules: PivotRSI, LinRegChannel
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `FeatureStore` | Builds unified features from all modules; supports ablation flags |
| `RegimeDetector` | Classifies market regime (trend_up, trend_down, chop) using cuML/sklearn RandomForest |
| `SignalBlender` | 3-class signal prediction (-1, 0, 1) with probability calibration |
| `DirectionBlender` | Binary direction classifier for trade samples only |
| `train.py` | Main training script with walk-forward and static training modes |
| `backtest.py` | Backtesting with EV-based filtering and calibration |

---

## Decisions

### D1: GPU Backend (cuML over LightGBM)
- **Decision**: Use cuML RandomForestClassifier for GPU training, sklearn for CPU fallback
- **Rationale**: RAPIDS ecosystem provides GPU acceleration without LightGBM dependency issues
- **Date**: Pre-existing

### D2: Probability Calibration Default
- **Decision**: Always enable isotonic calibration with sharpening (alpha=2.0) by default
- **Rationale**: Improves probability estimates for EV-based trade filtering
- **Date**: Pre-existing

### D3: Label System (Horizon-Based)
- **Decision**: Use horizon-based future returns with smoothing for direction labels
- **Rationale**: Reduces noise from single-bar returns
- **Default**: `--horizon-bars=12`, `--label-threshold=0.0005`, `--smoothing-window=12`
- **Date**: 2026-01-07

### D4: Walk-Forward Training
- **Decision**: Support walk-forward training with purge/embargo for label leakage prevention
- **Rationale**: More realistic evaluation than single train/val split
- **Flags**: `--walkforward`, `--purge-bars`, `--embargo-days`
- **Date**: 2026-01-07

### D5: Feature Ablation Flags
- **Decision**: Support disabling feature groups for A/B testing
- **Flags**: `--disable-ml-features`, `--disable-regime-context`, `--disable-signal-dynamics`, `--disable-rolling-stats`, `--disable-modules`, `--include-modules`
- **Date**: 2026-01-07

---

## Constraints

### C1: Data Requirements
- Expects OHLCV CSV with columns: `open`, `high`, `low`, `close`, `volume`
- Datetime index required
- Split datasets: `btcusd_5min_train_2017_2022.csv`, `btcusd_5min_val_2023.csv`, `btcusd_5min_test_2024.csv`

### C2: GPU Requirements
**RunPod (remote):**
- CUDA 12, Python 3.10, RAPIDS 24.08
- Use `environment.runpod.yml` for conda environment
- cuML/cuDF required for GPU training

**Local (WSL2 Ubuntu):**
- WSL2 Ubuntu available with GPU passthrough
- RTX 4070 (8GB VRAM) detected via `nvidia-smi`
- CUDA 13.1 driver installed
- Python 3.12.3 (system) - **Note:** RAPIDS requires Python 3.10-3.11
- Conda env `hybridyzer` uses Python 3.10 with RAPIDS 24.08 (cuML/cuDF installed)
- Fallback: use `--cpu-only` with sklearn (works but slower)

### C3: Feature Count
- FeatureStore generates 1000+ features
- Safe mode drops high-NaN warmup features
- Feature pruning available via `--use-full-pipeline`

---

## Assumptions

### A1: 5-Minute Data Default
- Default `--horizon-bars=12` assumes 5-minute bars (12 bars = 1 hour)
- Adjust for other timeframes

### A2: Transaction Costs
- Default `--fee-bps=1.0` (1 basis point per side)
- Round-trip cost = 2 bps

### A3: Regime Labels
- `indicator` strategy uses `make_regime_labels()` (default)
- `rule` strategy uses `rule_based_regime()` with linreg features

---

## Open Questions

### Q1: Optimal Horizon for 5-Min Data?
- Current default: 12 bars (1 hour)
- Previous default was 48 bars (4 hours)
- Need A/B testing to determine optimal horizon

### Q2: Walk-Forward vs Static Training Performance?
- Walk-forward provides more realistic eval but higher variance
- Need systematic comparison

### Q3: Feature Importance Analysis
- cuML RandomForest doesn't provide feature importances
- Consider alternative importance methods (permutation, SHAP)

### Q4: Walk-forward Training Crash (2026-01-07)
- Training aborted around window 25/77 with `Errno 22: Invalid argument`
- Log ended with a truncated traceback; root cause unknown
- **Resolved**: Added `faulthandler.enable()` and full `traceback.format_exc()` logging to walk-forward exception handlers (2026-01-08)

### ~~Q5: Opus Scan Findings (2026-01-07)~~ RESOLVED
- ~~Static manifest creation inside `if regime_models_direction` can cause NameError if none created~~ **Fixed**
- ~~Indentation in static manifest dict is inconsistent and likely a copy/paste error~~ **Fixed**
- Walk-forward manifest lacks `paths` block (not a bug, just inconsistent with static manifest)

### ~~Q6: Local GPU Training Setup (2026-01-09)~~ RESOLVED
- WSL2 Ubuntu has Miniconda with `hybridyzer` env (RAPIDS 24.08, Python 3.10)
- GPU access confirmed via `nvidia-smi`
- Nightly run executed on 2026-01-09 (see `results/nightly/20260109_060121`)

---

## Proposals

- ~~Capture a short user-provided summary of the recent structural changes and record them here.~~ Done via repo scan
- ~~If needed, perform a targeted repo scan to map renamed/moved files and update references.~~ Done (2026-01-08)
- ~~Add full stack capture or explicit exception logging in walk-forward loop.~~ Done (faulthandler + traceback)
- ~~Fix static manifest scope/indentation in `train.py` to avoid NameError and improve readability.~~ Done

### ~~P1: WSL RAPIDS Environment Setup (2026-01-09)~~ DONE
- Confirmed `hybridyzer` conda env (RAPIDS 24.08, Python 3.10)
- Ran GPU nightly: `results/nightly/20260109_060121` (no runs met drawdown constraint)

## Critiques

- ~~Current context may be stale if structural changes were made; relying on it risks incorrect guidance.~~ Resolved via repo scan
- ~~Static manifest bug noted as "fixed" above is still present in `train.py`~~ Fixed
- ~~Walk-forward training logs can end with a truncated traceback, hindering diagnosis.~~ Fixed with faulthandler

## Alternatives

- ~~Use only a user-written summary (fast, low effort) vs. a direct repo scan (accurate, slower).~~ Chose repo scan

---

## Recent Changes

### 2026-01-07: train.py Major Refactor
- Added `compute_return_metrics()` for comprehensive trade evaluation
- Added walk-forward training with `--walkforward` flag
- Added purge/embargo support (`--purge-bars`, `--embargo-days`)
- Added feature ablation flags
- Added confidence filter `--min-trade-proba`
- Added logging to file with `_Tee` class
- Added training manifest JSON output
- Changed default `--horizon-bars` from 48 to 12
- **Bug fixed**: Static manifest was only written inside `if regime_models_direction:` block

### 2026-01-07: Dependencies
- Added `matplotlib>=3.7.0` to `requirements.txt`
- Added `pyarrow>=10.0.0` to `requirements.txt` (for parquet support)
- Created `environment.runpod.yml` for GPU setup

### 2026-01-07: Claude Working Agreement
- CLAUDE.md mandates agent_context as source of truth for decisions/assumptions
- Action: update this file once user confirms structure changes

### 2026-01-07: Execution
- Run walk-forward training on 5-minute split CSVs in `data/` (user request)

### 2026-01-08: Crash Diagnostics & File Reference Update
- Added `faulthandler` import and `faulthandler.enable()` for crash dumps
- Added `traceback.format_exc()` to walk-forward exception handlers
- Updated File Reference with full repo structure (core/, modules/, tools/)
- Marked Q5 (manifest bug) as resolved
- Updated Q4 (walk-forward crash) with faulthandler addition

### 2026-01-09: Local GPU + Nightly Run
- Confirmed WSL2 conda env `hybridyzer` (RAPIDS 24.08, Python 3.10) with GPU access
- Fixed `faulthandler.enable()` to target the log file handle (avoids `_Tee` fileno errors)
- Ran nightly GPU job: `results/nightly/20260109_060121` (no runs met drawdown constraint)

### 2026-01-09: RunPod Deployment Prep
- Added `setup_runpod.sh` for easy environment setup on network volume
- Committed all core changes to master (de4ce08)
- Key files for RunPod:
  - `environment.runpod.yml` - conda env with RAPIDS 24.08
  - `--runpod` flag on `train.py` and `backtest.py` sets base path to `/workspace/Hybridyzer`
  - Data files (CSV) must be copied separately to `data/` (not in git)
- Training command: `python train.py --runpod --walkforward`
- Nightly runner: `python tools/nightly_runner.py --time-budget-hours 8 --promote-best`

---

## File Reference

### Core
| File | Purpose |
|------|---------|
| `core/feature_store.py` | Feature engineering with ablation support |
| `core/regime_detector.py` | Regime classification (cuML/sklearn) |
| `core/signal_blender.py` | Signal/Direction blending with calibration |
| `core/labeling.py` | Label generation (horizon-based) |
| `core/scalers.py` | Feature scaling utilities |
| `core/hybrid_engine.py` | Runtime hybrid signal engine |
| `core/final_signal.py` | Final signal computation |
| `core/risk_layer.py` | Risk management layer |
| `core/profiles.py` | Configuration profiles |
| `core/training_utils.py` | Training utility functions |

### Modules (Signal & Context)
| File | Purpose |
|------|---------|
| `modules/base.py` | Base module interface |
| `modules/superma.py` | SuperMA signal module |
| `modules/trendmagic.py` | TrendMagic signal module |
| `modules/pvt_eliminator.py` | PVT Eliminator signal module |
| `modules/pivots_rsi.py` | PivotRSI context module |
| `modules/linreg_channel.py` | LinReg Channel context module |
| `modules/ohlcv_context.py` | OHLCV context features |

### Tools
| File | Purpose |
|------|---------|
| `tools/walk_forward.py` | Walk-forward training utilities |
| `tools/timeseries_cv.py` | Time-series cross-validation |
| `tools/grid_search_recent.py` | Hyperparameter grid search |
| `tools/bootstrap_equity.py` | Bootstrap equity curve analysis |
| `tools/audit_backtest.py` | Backtest auditing |
| `tools/nightly_runner.py` | Nightly CI runner |
| `tools/profile_data.py` | Data profiling utilities |

### Top-Level
| File | Purpose |
|------|---------|
| `train.py` | Main training script (~2900 lines) |
| `backtest.py` | Backtesting with calibration |
| `main.py` | Entry point / CLI |
| `environment.runpod.yml` | GPU conda environment spec |
| `requirements.txt` | Pip dependencies |
| `models/` | Trained models and artifacts |
| `results/` | Training logs and metrics |
| `data/` | OHLCV data files |

