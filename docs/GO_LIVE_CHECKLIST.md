# Hybridyzer Go-Live Checklist

**Goal**: Reliable local workflow → RunPod scaling → feature upgrades → live trading.

**How to use this document**:
- For humans: Follow the phases in order. Don't skip ahead.
- For LLMs: Each phase has explicit PASS/FAIL criteria. Verify all criteria before proceeding.

---

## Quick Reference: When to Retrain vs Backtest

| What Changed | Action Required |
|--------------|-----------------|
| Feature engineering (modules, FeatureStore) | DELETE cache + RETRAIN + BACKTEST |
| Labels (horizon, threshold, cost-adjustment) | RETRAIN + BACKTEST |
| Model hyperparameters (deep-train, trees) | RETRAIN + BACKTEST |
| Backtest thresholds only (--p, --min-ev) | BACKTEST only |
| Nothing (just validating) | BACKTEST only |

**Rule of thumb**: If you touched `train.py` args or `core/` code → retrain. If you only touched `backtest.py` args → backtest only.

---

## Quick Reference: When to Consider Enhancements

| Phase | Enhancement Timing | What's Available |
|-------|-------------------|------------------|
| 1-2 | ❌ Don't add yet | Focus on baseline |
| 3 | ✅ Backtest-only | Asymmetric thresholds, no-trade zones |
| 3.5 | ✅ Systematic search | Nightly runner with `--promote-best` (hyperparameter tuning) |
| 4 | ✅ Training-side | Vol-scaled labels, feature expansion, triple-barrier |
| 5-6 | ✅ Advanced | Meta-model, regime-specific blenders |
| 7 | ⚠️ **CRITICAL** | Drawdown pause, position sizing (before live!) |

See `agent_context.md` → "Future Enhancements" for full list with complexity ratings.

---

## Minimum Performance Gates

These are the minimum thresholds to pass each phase. Adjust if you have domain knowledge suggesting different values.

| Metric | Minimum | Target (nice to have) |
|--------|---------|----------------------|
| Sharpe Ratio | > 0.3 | > 1.0 |
| Max Drawdown | < 30% | < 15% |
| Win Rate | > 40% | > 50% |
| Trade Count | > 30 trades | > 100 trades |
| Profit Factor | > 1.0 | > 1.5 |

**FAIL** = Do not proceed. Fix issues or accept current phase as final.
**PASS** = Proceed to next phase.

---

## Phase 1: Local Training (Build Fresh Models)

### When to run this phase
- First time setup
- After ANY change to features, labels, or model config
- After deleting models or cache

### Prerequisites
- [ ] WSL environment working (`nvidia-smi` shows GPU)
- [ ] Correct conda env activated: `conda activate hybridyzer-local`
- [ ] Data files exist in `data/`

### Commands
```bash
# If features changed, delete cache first:
rm models/cached_features.parquet

# Run training
python train.py --walkforward --recompute-feature-cache --featurestore-gpu
```

### Success Criteria (all must pass)

| Check | How to Verify | PASS condition |
|-------|---------------|----------------|
| Training completed | Console output | "Training complete" message, no crash |
| All windows finished | `models/training_results.csv` | 77 rows (or expected window count) |
| No NaN metrics | `models/training_results.csv` | No NaN in sharpe/return columns |
| Models saved | `ls models/` | `regime_model.pkl`, `blender_model.pkl`, `blender_direction_model.pkl` exist |
| Manifest created | `models/training_manifest.json` | File exists with `runtime_env` section |
| GPU stack recorded | `models/training_manifest.json` | `cuml_version` field is not null |

### If FAIL
- Check error logs in `results/`
- If CUDA error: restart WSL (`wsl --shutdown` from PowerShell, then reopen)
- If cache mismatch: delete `models/cached_features.parquet` and rerun

### If PASS
→ Proceed to Phase 2

> **Enhancement Note**: Don't add enhancements yet. Get baseline working first.

---

## Phase 2: Local Backtest (Validate Models)

### When to run this phase
- Always after Phase 1 completes
- After changing backtest-only parameters
- To establish baseline metrics

### Prerequisites
- [ ] Phase 1 completed successfully
- [ ] Models exist in `models/`

### Commands
```bash
python backtest.py --featurestore-gpu
```

### Success Criteria (all must pass)

| Check | How to Verify | PASS condition |
|-------|---------------|----------------|
| Backtest completed | Console output | No crash, metrics printed |
| No cuML errors | Console output | No "KeyError" or version mismatch |
| Sharpe ratio | Console output or results CSV | **> 0.3** |
| Max drawdown | Console output or results CSV | **< 30%** |
| Trade count | Console output | **> 30 trades** |
| No regime collapse | Console output | All 3 regimes have trades (not 100% one regime) |

### Record Baseline (IMPORTANT)
Save these numbers somewhere (notepad, `results/baseline.txt`, etc.):
```
Phase 2 Baseline:
- Sharpe: ___
- Return: ___% 
- Max DD: ___%
- Trades: ___
- Date: ___
```

### If FAIL
- If cuML mismatch: use `--cpu-only` flag
- If metrics below minimum: DO NOT proceed. Either:
  - Accept this as your baseline and stop here, OR
  - Go back to Phase 1 with different training config

### If PASS
→ Proceed to Phase 3

> **Enhancement Note**: Don't add enhancements yet. Establish baseline metrics first.

---

## Phase 3: Local Tuning (Improve Without Retraining)

### When to run this phase
- Only after Phase 2 passes minimum gates
- Only for backtest parameter tuning (NOT feature/model changes)

### What you CAN tune here (backtest only, no retrain needed)
- `--p` / `--p-long` / `--p-short` (probability thresholds)
- `--min-ev` (expected value filter)

### What you CANNOT tune here (requires going back to Phase 1)
- Feature engineering
- Label parameters
- Model hyperparameters

### Commands
```bash
# Example: try stricter probability threshold
python backtest.py --featurestore-gpu --p 0.6

# Example: try EV filtering
python backtest.py --featurestore-gpu --min-ev 0.001
```

### Success Criteria

| Check | How to Verify | PASS condition |
|-------|---------------|----------------|
| Improvement vs Phase 2 | Compare to baseline | Sharpe OR return improved, OR drawdown decreased |
| No severe regression | Compare to baseline | No metric dropped by more than 50% |
| Trade count acceptable | Console output | Still > 30 trades (didn't over-filter) |

### If FAIL (no improvement found)
- That's OK. Phase 2 baseline is your local result.
- Proceed to Phase 4 with Phase 2 settings.

### If PASS
- Record new best settings
- **Option A**: Proceed to Phase 3.5 (systematic hyperparameter search)
- **Option B**: Proceed directly to Phase 4 (if you're satisfied with manual tuning)

### 💡 Enhancement Opportunities (Optional - backtest-only, no retrain)

These are LOW complexity and don't require retraining. Good time to try if baseline is solid:

| Enhancement | What to do | Why now |
|-------------|------------|---------|
| Asymmetric thresholds | Try `--p-long 0.55 --p-short 0.60` | Tune long/short separately |
| No-trade zones | Add `--min-volatility` filter (if implemented) | Skip low-vol chop |

If you implement these, they only affect backtest - no need to go back to Phase 1.

---

## Phase 3.5: Systematic Hyperparameter Search (Optional but Recommended)

### When to run this phase
- **After Phase 2 passes** (you have a working baseline)
- **Before Phase 4** (find better configs before scaling to RunPod)
- **OR as part of Phase 4** (on RunPod with more compute time)

### What this does
The nightly runner automatically tests multiple configurations and promotes the best one:
- Tests different horizons (12, 24, 36 bars)
- Tests different smoothing windows (6, 8, 12)
- Tests feature ablations (disable ML features, regime context, etc.)
- Tests module combinations
- Uses Thompson sampling (optional) to focus on promising configs
- **`--promote-best`** automatically copies the best run's models to `models/`

### Commands

**Local (4-8 hour budget):**
```bash
# Basic search (tests horizons, smoothing, feature ablations)
python tools/nightly_runner.py --time-budget-hours 4 --promote-best

# With Thompson sampling (smarter config selection)
python tools/nightly_runner.py --time-budget-hours 8 --promote-best --bandit-thompson
```

**RunPod (8-24 hour budget):**
```bash
# Longer search with more configs
python tools/nightly_runner.py --runpod --time-budget-hours 8 --promote-best --bandit-thompson

# Background run (24 hours)
nohup python tools/nightly_runner.py --runpod --time-budget-hours 24 --promote-best --bandit-thompson > training.log 2>&1 &
```

### What `--promote-best` does
After the nightly run completes:
1. Finds the best run (by trimmed mean return, considering drawdown constraints)
2. **Automatically copies** these files from `results/nightly/<timestamp>/best/` → `models/`:
   - `regime_model.pkl`
   - `blender_model.pkl`
   - `blender_direction_model.pkl`
   - `training_manifest.json`
   - `training_results.csv`
   - Feature column files

**Result**: The best configuration becomes your production models without manual copying.

### Success Criteria

| Check | How to Verify | PASS condition |
|-------|---------------|----------------|
| Best run found | Check `results/nightly/<timestamp>/best.json` | File exists with valid metrics |
| Models promoted | Check `models/` timestamps | Model files updated to best run |
| Improvement vs Phase 2 | Compare `best.json` metrics | Better than Phase 2 baseline |
| Drawdown constraint | Check `best.json` | `worst_drawdown` < `--max-drawdown` (default 30%) |

### If FAIL (no runs meet drawdown constraint)
- All tested configs had drawdown > 30%
- **Action**: Review Phase 2 baseline - may need to fix fundamental issues first
- Consider: Lower `--max-drawdown` threshold, or fix training data/labels

### If PASS
- Best config is now in `models/` (thanks to `--promote-best`)
- **Proceed to Phase 4** with the promoted models, OR
- **Run Phase 2 backtest** on the promoted models to validate

### Notes
- **Without `--promote-best`**: You must manually copy the best run's models from `results/nightly/<timestamp>/best/` to `models/`
- **Thompson sampling** (`--bandit-thompson`): Focuses compute on promising configs, more efficient than exhaustive search
- **Time budget**: The runner stops when time expires, so longer budgets = more configs tested
- **Drawdown filter**: Only runs with `max_drawdown < --max-drawdown` (default 30%) are considered for "best"

---

## Phase 4: RunPod Training (Scale Up)

### When to run this phase
- Only after local baseline is established (Phase 2 or 3)
- When you want faster training or to use `--deep-train`

### Prerequisites
- [ ] RunPod instance running with GPU
- [ ] `setup_runpod.sh` executed
- [ ] Data files copied to RunPod `data/`

### Commands
```bash
# On RunPod:
python train.py --runpod --walkforward --recompute-feature-cache

# Optional: deep training (requires 40GB+ VRAM)
python train.py --runpod --walkforward --deep-train --recompute-feature-cache
```

### Success Criteria

| Check | How to Verify | PASS condition |
|-------|---------------|----------------|
| Training completed | Console output | No crash |
| Results comparable to local | Compare metrics | Within 20% of local Sharpe |
| Manifest shows RunPod env | `training_manifest.json` | Different GPU in `runtime_env` |

### If FAIL
- Check RAPIDS version match between local and RunPod
- If version mismatch: decide which env is authoritative

### If PASS
→ Run backtest on RunPod to confirm, then proceed to Phase 5 (when implemented)

### 💡 Enhancement Opportunities (Optional - requires retrain)

RunPod has more compute. Good time for training-side improvements:

| Enhancement | What to do | Why now |
|-------------|------------|---------|
| Volatility-scaled label thresholds | Modify `core/labeling.py` to use ATR | Adapts to market conditions |
| Feature expansion: time-of-day | Add hour/day features to FeatureStore | Capture seasonality |
| Feature expansion: vol regime | Add ATR z-score, realized vol features | Better regime detection |
| Triple-barrier labels | Implement stop/take-profit exit logic | More realistic labels |

⚠️ These require DELETE cache + RETRAIN. Compare results to Phase 4 baseline before keeping.

---

## Phase 5: Pattern Memory (FUTURE - NOT IMPLEMENTED)

> **STATUS**: The `--pattern-memory` flag does not exist yet.
> See `docs/engram_pattern_memory_plan.md` for design.
> Skip this phase until implemented.

### When to run this phase
- Only after Phase 4 passes
- Only after pattern memory feature is implemented

### Success Criteria (draft)
- Improved hit rate vs Phase 4
- No data leakage detected

### 💡 Enhancement Opportunities (Optional)

| Enhancement | What to do | Why now |
|-------------|------------|---------|
| Meta-model for trade quality | Train a model to predict which trades will win | Gate low-quality trades |
| Regime-specific blenders | Separate blender per regime | Specialize predictions |

---

## Phase 6: Thompson Sampling + Deep Train (FUTURE - PARTIAL)

> **STATUS**: `--deep-train` exists. Thompson Sampling does not.

### Deep Train (available now)
```bash
python train.py --runpod --walkforward --deep-train --recompute-feature-cache
```

### Success Criteria
- Deep train Sharpe > standard train Sharpe
- No increase in max drawdown

---

## Phase 7: Live Readiness

### Prerequisites
- [ ] All previous phases passed
- [ ] Test data (`btcusd_5min_test_2024.csv`) NEVER used in training

### Checklist
- [ ] Final backtest on held-out test set (2024 data)
- [ ] Test set Sharpe within 50% of validation Sharpe (no severe overfit)
- [ ] Risk limits defined (max position, max drawdown to halt)
- [ ] Rollback plan documented
- [ ] Paper trading period completed (optional but recommended)

### Minimum Go-Live Gates

| Metric | Requirement |
|--------|-------------|
| Test set Sharpe | > 0.3 |
| Test set Max DD | < 30% |
| Validation vs Test gap | < 50% difference |

### ⚠️ CRITICAL Enhancements Before Live

These are **strongly recommended** before real money:

| Enhancement | What to do | Why critical |
|-------------|------------|--------------|
| Max daily loss / drawdown pause | Halt trading if DD exceeds X% | **Safety net** - prevents catastrophic loss |
| Volatility-scaled position sizing | Size by ATR, cap max position | **Risk management** - don't over-leverage |
| Time-based stop | Exit after N bars if no move | Avoid stuck positions |

Without these, you're trading without a seatbelt.

---

## Rollback Plan

1. Keep last 2-3 model sets with their manifests
2. If live performance degrades:
   - Stop trading
   - Revert to previous model set
   - Run backtest to confirm
   - Record reason and timestamp

---

## Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| CUDA_ERROR_NO_DEVICE | Wrong conda env. Use `hybridyzer-local` not `hybridyzer` |
| KeyError during model load | cuML version mismatch. Use `--cpu-only` or retrain |
| Training crashes mid-run | Check GPU memory. Avoid `--deep-train` on 8GB VRAM |
| Backtest shows 0 trades | Probability threshold too high. Lower `--p` value |
| 100% one regime | Regime detector broken. Check for NaN features, retrain |
