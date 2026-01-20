# Agent Context - Hybridyzer

> Single source of truth for decisions, constraints, assumptions, and open questions.
> Last updated: 2026-01-19

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

### D6: Cost-Adjusted Direction Labels
- **Decision**: Add round-trip costs to the label threshold during training by default
- **Rationale**: Prevents the model from learning edges smaller than taker+slippage
- **Defaults**: `--taker-fee-bps=4.5`, `--slippage-bps=1.0`, `--no-cost-adjusted-labels` to disable
- **Date**: 2026-01-16

### D7: Backtest Cost Model (Entry/Exit)
- **Decision**: Apply transaction costs only on position changes (entry/exit/flip), not per bar
- **Rationale**: Costs must reflect trades, not every bar in position
- **Date**: 2026-01-16

### D8: Low-Confidence Trades
- **Decision**: Disable low-confidence passthrough by default
- **Rationale**: Prevents high-churn trading below main probability threshold
- **Default**: `allow_low_confidence=False`
- **Date**: 2026-01-16

### D9: Deep-Train Option
- **Decision**: Add `--deep-train` to increase model capacity for regime + blenders
- **Rationale**: Optional higher-capacity models after cost-aware labels are in place
- **Date**: 2026-01-16

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
- Use `setup_runpod.sh` for automated setup
- cuML/cuDF required for GPU training
- Large VRAM (40-80GB) allows `--deep-train`

**Local (WSL2 Ubuntu):**
- WSL2 Ubuntu available with GPU passthrough
- RTX 4070 Laptop (8GB VRAM) detected via `nvidia-smi`
- CUDA 12.x driver installed
- Use `environment.local.yml` for conda environment (`hybridyzer-local`)
- Use `setup_local.sh` for automated setup
- Python 3.10 (via conda) for RAPIDS compatibility
- **Avoid `--deep-train`** on 8GB VRAM (may OOM)
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
- Default taker fee `--taker-fee-bps=4.5` and slippage `--slippage-bps=1.0`
- Round-trip cost = 11 bps (0.11%)

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

### Q7: NaN Propagation in Feature Generation (2026-01-14)
- **Problem**: Features `superma_topvecMA`, `trendmagic_topavg`, `trendmagic_topsig`, `pvt_topvecMA` were 100% NaN, causing regime detector to fail (all predictions = "trend_up", probabilities = NaN)
- **Root Cause**: NaN values from `wilder_atr` (during warmup period) propagated through stateful vector calculations in `superma.py`, `trendmagic.py`, and `pvt_eliminator.py`
- **Fix Applied**: Modified modules to use `0` for decay values when `ATR` is `NaN` during warmup, preventing NaN propagation
- **Status**: Feature generation fixed, but existing trained models may have been trained on NaN features and need retraining
- **Next Steps**: Retrain models with fixed feature generation to verify regime detector works correctly
- **Related Issues**:
  - Feature prefix mismatch between `FeatureStore.build()` (used in inference) and `FeatureStore.build_features()` (used in training) - **Fixed**: Removed `legacy_` prefix from `build()` to match training
  - `safe_mode` NaN filtering in `build()` was removing features that training kept - **Fixed**: Removed `safe_mode` filtering from `build()` to match training behavior

---

## Local Run Issues & Fixes (2026-01-19)

- **GPU feature engineering disabled**: Training calls `build_features_once(..., use_gpu=False)` and backtest hard-codes `feature_store_use_gpu=False`, so FeatureStore never uses GPU. **Fix**: Add a `--featurestore-gpu` flag (or tie to `use_cuml`) and pass it into FeatureStore for train/backtest.
- **`--recent` still builds full features**: Backtest computes features for all data, then slices. **Fix**: Slice early with a warmup buffer (max lookback) or add `--recent-warmup` to keep indicators correct while speeding local runs.
- **Backtest has no feature cache**: Every run recomputes features even if inputs unchanged. **Fix**: Add optional cache path (e.g., `--feature-cache`) and reuse `FeatureStore.build_and_cache` to speed iteration.
- **cuML pickle version mismatch**: GPU models fail to load across cuML/RAPIDS upgrades (`KeyError: 'n_cols'`). **Fix**: Store cuML/cuDF/RAPIDS versions in the training manifest and model metadata; verify on load and fail fast with actionable message (match env or retrain).
- **Local GPU parity drift**: It is easy to run train/backtest on different conda envs or versions. **Fix**: Add local wrapper scripts (e.g., `tools/train_local.ps1`, `tools/backtest_local.ps1`) that activate the correct env and pin versions.

### Agent Commentary (2026-01-19)

- **GPU feature engineering**: Worth adding, but should be **tied to `use_cuml` by default** rather than a separate flag, to keep config simple. Recommend: `--featurestore-gpu` only if we discover cases where we want GPU models but CPU features (or vice versa). Profile first; if FeatureStore GPU only saves a few percent, keep it opt-in.
- **`--recent` behavior**: Early slicing is a good idea, but **must include a warmup buffer** equal to the maximum lookback across all modules (SuperMA/TrendMagic/PVT + rolling stats). Recommend: `--recent N` + `--recent-warmup M` where `M` defaults conservatively (e.g. 1000 bars) so local iteration is faster but indicators remain valid.
- **Feature cache for backtest**: High ROI. Recommend: add `--feature-cache PATH` which, when present, uses `FeatureStore.build_and_cache` (or equivalent) and **embeds a small metadata JSON** (data hash, ablation flags, horizon/label params) so we can cheaply detect when the cache is stale.
- **cuML version mismatch**: Runtime patching of `__setstate__` is fragile. Better to **treat cuML models as versioned artifacts**: record `cuml`, `cudf`, `rapids`, `cuda` versions in both the training manifest and `model_data`, then on load: (1) compare versions, (2) if incompatible, fail fast with a clear message ("env mismatch, retrain or align versions"), and (3) suggest `--cpu-only` as a fallback. Patches can remain but should be considered best-effort, not relied on.
- **Local GPU parity drift**: Wrapper scripts are good guardrails, but we should also have **inline env checks** in `train.py`/`backtest.py` (e.g., echo active env, Python, and RAPIDS versions) so misconfigurations are obvious even when commands are run manually. Long term, a `tools/check_env.py` or `python -m hybridyzer.check_env` command that validates all of this would be ideal.

---

## Proposals

- ~~Capture a short user-provided summary of the recent structural changes and record them here.~~ Done via repo scan
- ~~If needed, perform a targeted repo scan to map renamed/moved files and update references.~~ Done (2026-01-08)
- ~~Add full stack capture or explicit exception logging in walk-forward loop.~~ Done (faulthandler + traceback)
- ~~Fix static manifest scope/indentation in `train.py` to avoid NameError and improve readability.~~ Done

### ~~P1: WSL RAPIDS Environment Setup (2026-01-09)~~ DONE
- Confirmed `hybridyzer` conda env (RAPIDS 24.08, Python 3.10)
- Ran GPU nightly: `results/nightly/20260109_060121` (no runs met drawdown constraint)

### ~~P2: GPU Pre-Transfer Optimization (2026-01-10)~~ IMPLEMENTED
- **Problem**: Current walk-forward training converts pandas → cuDF 385 times (77 windows × 5 models), causing significant CPU→GPU transfer overhead (~30-40% of training time)
- **Solution**: Pre-load full feature DataFrame to GPU once at start of walk-forward training, then slice cuDF DataFrames per window (no transfer needed)
- **Expected speedup**: 2-3x faster training (from ~4-5 hours to ~1.5-2.5 hours per complete run)
- **Implementation**:
  - Pre-load `cached_features` to cuDF once in `combined_training()` after loading from parquet
  - Modify `slice_window()` to support cuDF slicing (convert to pandas only at end for compatibility)
  - Update `RegimeDetector.fit()`, `SignalBlender.fit()`, `DirectionBlender.fit()` to detect and use cuDF directly if provided
- **Notes**:
  - Apply any window-dependent ops (rolling stats, label smoothing, purge/embargo) after slicing to avoid leakage
  - Track GPU memory pressure; fallback to chunked slicing if the full feature frame is too large
- **Status**: Implemented (2026-01-10)

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

### 2026-01-10: GPU Pre-Transfer Optimization
- Added GPU memory check helper function `_check_gpu_memory_safe_for_preload()` in `train.py`
- Modified `combined_training()` to pre-load `cached_features` to cuDF once if GPU available and memory check passes
- Modified `slice_window()` to support cuDF DataFrames (strip timezone info for cuDF compatibility)
- Updated `RegimeDetector.fit()`, `SignalBlender.fit()`, `DirectionBlender.fit()` to accept cuDF directly
- Added progress bar and heartbeat thread for better monitoring on RunPod
- Added explicit `sys.stdout.flush()` calls for live output on remote systems

### 2026-01-14: NaN Propagation Fix
- **Issue**: Features `superma_topvecMA`, `trendmagic_topavg`, `trendmagic_topsig`, `pvt_topvecMA` were 100% NaN, breaking regime detector predictions
- **Root Cause**: NaN values from `wilder_atr` (during warmup) propagated through stateful vector calculations
- **Fix**: Modified `modules/superma.py`, `modules/trendmagic.py`, `modules/pvt_eliminator.py` to use `0` for decay values when `ATR` is `NaN`
- **Impact**: Feature generation now produces valid values, but existing models trained on NaN features may need retraining
- **Diagnostic**: Created `diagnostic_regime.py` to inspect feature generation and model behavior (temporary, deleted after debugging)

### 2026-01-16: Cost-Aware Training + Backtest Fixes
- Added cost-adjusted direction labels (round-trip costs added to label threshold)
- Added CLI fee inputs: `--taker-fee-bps`, `--slippage-bps`, with `--fee-bps` deprecated
- Backtest now charges costs on entry/exit/flip (position changes) and reports trade-level stats
- Disabled low-confidence passthrough by default (`allow_low_confidence=False`)
- Added `--deep-train` flag with larger model params for RegimeDetector and blenders

### 2026-01-19: Separate Local WSL Setup
- Created dedicated local WSL setup files separate from RunPod:
  - `environment.local.yml` - Conda env for RTX 4070 (8GB VRAM)
  - `requirements_local.txt` - Pip requirements for local setup
  - `setup_local.sh` - Automated setup script for local WSL
- Local uses `hybridyzer-local` conda env to avoid conflicts
- Updated `docs/WSL_SETUP_STEPS.md` with local-specific instructions
- **Note:** Avoid `--deep-train` on 8GB VRAM (risk of OOM)
- Fixed line endings in `setup_local.sh` (CRLF → LF) and added `.gitattributes` to prevent future issues

### 2026-01-19: GPU Auto-Detection for Backtesting
- **Change**: `backtest.py` now auto-detects GPU availability by default
- **Behavior**: 
  - If GPU is available → uses cuML automatically
  - If GPU not available → falls back to sklearn (CPU)
  - `--cpu-only` flag forces CPU mode
  - `--use-cuml` flag explicitly enables GPU (overrides auto-detect)
- **Impact**: No need to specify `--use-cuml` flag anymore; GPU is used automatically when available

### 2026-01-19: GPU Context Loss Handling (WSL)
- **Problem**: WSL can lose GPU context during long training runs, causing `CUDA_ERROR_NO_DEVICE` errors
- **Solution**: Added automatic GPU→CPU fallback when GPU context is lost:
  - `_is_cuda_error()` - Detects CUDA-related exceptions
  - `_check_gpu_available()` - Checks if GPU is still accessible
  - Modified `slice_window()` to raise `RuntimeError("GPU_CONTEXT_LOST")` on CUDA errors
  - Training loop automatically converts cuDF to pandas and continues with CPU when GPU context is lost
- **Impact**: Training can now continue on CPU if GPU becomes unavailable mid-run (common in WSL)
- **Note**: Most `CUDA_ERROR_NO_DEVICE` issues in WSL are caused by using wrong conda env (`hybridyzer` instead of `hybridyzer-local`). Always use `hybridyzer-local` for local WSL setup.

### 2026-01-19: Local Run Efficiency + GPU Version Guardrails
- Added GPU stack metadata capture for model artifacts and manifests (cuML/cuDF/cuPy/CUDA runtime)
- cuML model load now fails fast with a clear error when GPU stack versions mismatch
- FeatureStore GPU use is tied to `use_cuml` by default with `--featurestore-gpu` / `--featurestore-cpu` overrides
- Backtest now supports `--recent-warmup` early slicing plus optional `--feature-cache` with metadata validation
- Both train/backtest print runtime env summary (Python/conda/GPU stack) for quick misconfig detection

---

## Future Enhancements

Extracted from `EDGE_IMPROVEMENT_TODO.txt` (archived to `docs/`). These are post-baseline improvements, not blockers for go-live.

**Integration with Go-Live Checklist**: Each phase in `docs/GO_LIVE_CHECKLIST.md` has an "Enhancement Opportunities" section showing which items to consider at that stage. Phase 7 marks critical safety features that should be implemented before live trading.

### High Priority

| Enhancement | Description | Complexity |
|-------------|-------------|------------|
| Triple-barrier labels | Time + stop-loss + take-profit exit logic instead of fixed horizon | Medium |
| Volatility-scaled position sizing | Size positions by ATR or realized vol, cap leverage | Medium |
| Max daily loss / drawdown pause | Safety guardrail - halt trading on drawdown spike | Low |

### Medium Priority

| Enhancement | Description | Complexity |
|-------------|-------------|------------|
| Volatility-scaled label thresholds | Use ATR or realized vol instead of fixed threshold | Low |
| Asymmetric thresholds | Different thresholds for long vs short | Low |
| Time-of-day / day-of-week features | Seasonality features in FeatureStore | Low |
| No-trade zones | Skip trades during low vol or high spread | Low |
| Weekly walk-forward grid | Automated batch of configs, select by stability | Low |

### Lower Priority / Exploratory

| Enhancement | Description | Complexity |
|-------------|-------------|------------|
| Meta-model for trade quality | Predict trade quality and gate trades | High |
| Regime-specific blenders | Separate blender models per regime | High |
| Feature expansion: vol regime | Realized vol, ATR z-score, vol-of-vol | Medium |
| Feature expansion: trend strength | Slope + distance from MA, ADX-like proxy | Medium |
| Feature expansion: range compression | Range vs ATR, Bollinger width | Medium |
| Feature expansion: volume divergence | Volume z-score, price-volume divergence | Medium |
| Feature expansion: momentum exhaustion | RSI slope, MACD slope | Medium |
| Timeframe comparison | 5m vs 15m, shorter vs longer history | Medium |
| Time-based stop | Exit after N bars if no move | Low |

---

## Go-Live Checklist Review (2026-01-19)

Reviewed and restructured `docs/GO_LIVE_CHECKLIST.md`.

**Problems with original:**
- Unclear when to retrain vs just backtest
- No quantitative pass/fail gates
- Too vague for LLM execution

**Restructured version includes:**
- **Decision table**: "What Changed → Action Required" (retrain vs backtest only)
- **Minimum performance gates**: Sharpe > 0.3, Max DD < 30%, Trades > 30
- **Per-phase success criteria**: Tables with specific PASS conditions
- **Baseline recording step**: Explicit instruction to save Phase 2 metrics
- **IF FAIL / IF PASS branches**: Clear decision tree
- **Troubleshooting quick reference**: Common errors and fixes

**Design goals:**
- ELI5 for human: Simple rules, don't skip steps
- Goal-oriented for LLMs: Unambiguous pass/fail criteria that can be verified

---

## Progress Tracking

Progress through the GO_LIVE checklist is tracked in `docs/PROGRESS_REPORT.md`, including:
- Dates and commands run
- Results and metrics
- Status of each phase
- Key findings and next steps

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
| `environment.runpod.yml` | RunPod conda environment spec |
| `environment.local.yml` | Local WSL conda environment spec |
| `setup_runpod.sh` | RunPod automated setup script |
| `setup_local.sh` | Local WSL automated setup script |
| `requirements.txt` | Base pip dependencies (CPU) |
| `requirements_runpod.txt` | RunPod pip dependencies (GPU) |
| `requirements_local.txt` | Local WSL pip dependencies (GPU) |
| `models/` | Trained models and artifacts |
| `results/` | Training logs and metrics |
| `data/` | OHLCV data files |
