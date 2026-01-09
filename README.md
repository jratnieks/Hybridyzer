# Hybridyzer

A hybrid trading architecture that combines multiple Pine Script strategies into modular Python signal engines with ML-based regime detection and signal blending.

## Overview

Hybridyzer is a sophisticated trading system that:
- Combines multiple trading strategies (SuperMA, TrendMagic, PVT Eliminator) into modular signal engines
- Uses machine learning (LightGBM) for regime detection and signal blending
- Generates comprehensive features including rolling statistics, regime context, and signal dynamics
- Applies risk management layers to produce final trading signals

## Architecture

### Core Components

- **`HybridEngine`**: Central orchestrator that processes data through all modules and produces final signals
- **`FeatureStore`**: Builds unified feature DataFrames from all modules with:
  - Base module features
  - Regime context features (ATR, volatility, kurtosis, skew, volume surge, return bins)
  - Signal dynamics features (streak length, time since signals, distance to extremes, derivatives)
  - Rolling statistics (mean, std, max, min, slope, zscore) for all features
- **`RegimeDetector`**: ML-based regime classification (trend_up, trend_down, chop, high_vol, low_vol)
- **`SignalBlender`**: ML-based signal blending that combines module signals based on regime
- **`RiskLayer`**: Risk management and position sizing

### Signal Modules

- **`SuperMA4hr`**: Hull moving average with top/bottom vector decay
- **`TrendMagicV2`**: Zigzag-based trend following with pyramiding
- **`PVTEliminator`**: Price-volume trend elimination strategy

### Context Modules

- **`PivotRSIContext`**: Pivot-based RSI context features
- **`LinRegChannelContext`**: Linear regression channel features

## Installation

```bash
pip install -r requirements.txt
```

**GPU (WSL/Linux) recommended:** use the conda environment in `environment.runpod.yml`
to avoid RAPIDS binary incompatibilities (pyarrow/numpy are pinned there).

### Requirements

- pandas >= 2.0.0
- numpy >= 1.24.0
- lightgbm >= 4.0.0

## RunPod GPU Setup

For GPU-accelerated training on RunPod, use the provided conda environment file that includes RAPIDS (cuML/cuDF) for CUDA 12, Python 3.10, and RAPIDS 24.08.

### Setup Steps

1. **Install micromamba** (or conda):
   ```bash
   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
   ```

2. **Create the environment**:
   ```bash
   micromamba env create -f environment.runpod.yml
   ```

3. **Activate the environment**:
   ```bash
   micromamba activate hybridyzer
   ```

4. **Run training**:
   ```bash
   python train.py --runpod --use-full-pipeline --calibration-method isotonic --alpha 2.0
   ```

**WSL/local GPU:** same env file works; run from the repo root without `--runpod`.

### VRAM Optimization

If you encounter low VRAM issues:
- Drop `--use-full-pipeline` flag, or
- Add `--disable-calibration` flag

### Data Path

The system expects data files in `/workspace/Hybridyzer/data/` with the following naming convention:
- **Split datasets (preferred for training/evaluation):**
  - `btcusd_5min_train_2017_2022.csv` - Training set (2017-2022)
  - `btcusd_5min_val_2023.csv` - Validation set (2023)
  - `btcusd_5min_test_2024.csv` - Test set (2024)
  - `btcusd_5min_test_2025.csv` - Optional test set (2025)
- **Single file fallback:**
  - `btcusd_5min.csv` - Combined 5-minute data
  - `btcusd_4H.csv` - 4-hour data
  - `btcusd_1min.csv` - 1-minute data
  - Other timeframes following the same pattern

**Note:** The default classification horizon is 12 bars (1 hour for 5-minute data). Labels are generated from each split's own data to prevent leakage.

### Notes

- **cuML/cuDF**: GPU-accelerated libraries are included via the environment file and automatically replace CPU-based operations when available.
- **CPU users**: If you're not using GPU, you can ignore `environment.runpod.yml` and simply run `pip install -r requirements.txt` as described in the standard installation.

## Usage

### Training Models

```bash
python train.py
```

This will:
1. Load BTC OHLCV data from split datasets (`btcusd_5min_train_2017_2022.csv` and `btcusd_5min_val_2023.csv`) or fall back to single file
2. Build comprehensive features from all modules
3. Generate regime labels (rule-based)
4. Generate blender labels (future returns over 1-hour horizon for 5-min data)
5. Train RegimeDetector and SignalBlender models
6. Save models to `models/`

To include trading costs in validation metrics, set `--fee-bps` (e.g., `--fee-bps 1.0` for 1 bp per entry/exit).
To reduce over-trading, set `--min-trade-proba` (e.g., `--min-trade-proba 0.60` to only trade on higher-confidence signals).

### Running Inference

```bash
python inference.py
```

### Main Execution

```bash
python main.py
```

## Data Format

The system expects OHLCV data with:
- `open`, `high`, `low`, `close`: Price columns
- `volume`: Volume column
- Datetime index

Use `data/btc_data_loader.py` to load CSV files:

```python
from data.btc_data_loader import load_btc_csv

df = load_btc_csv("data/btcusd_1min_volspikes.csv")
```

## Feature Engineering

The system generates 1000+ features including:

1. **Base Module Features**: Raw features from each trading module
2. **Regime Context Features**:
   - Rolling ATR (3, 6, 12, 24)
   - Rolling volatility (std of returns)
   - Rolling kurtosis and skew
   - Volume surge ratios
   - Return bins (bucketized)
3. **Signal Dynamics Features**:
   - Signal streak length
   - Time since last long/short signal
   - Distance to previous extreme (hull)
   - Derivatives of topvec and botvec
4. **Rolling Statistics**: Mean, std, max, min, slope, zscore for windows [3, 5, 10, 20]

## Project Structure

```
Hybridyzer/
├── core/
│   ├── feature_store.py      # Feature engineering and aggregation
│   ├── hybrid_engine.py       # Main orchestration engine
│   ├── regime_detector.py    # ML-based regime detection
│   ├── signal_blender.py     # ML-based signal blending
│   └── risk_layer.py          # Risk management
├── modules/
│   ├── base.py                # Base classes for modules
│   ├── superma.py             # SuperMA strategy
│   ├── trendmagic.py          # TrendMagic strategy
│   ├── pvt_eliminator.py      # PVT Eliminator strategy
│   ├── pivots_rsi.py          # Pivot RSI context
│   └── linreg_channel.py      # Linear regression channel context
├── data/
│   ├── btc_data_loader.py     # Data loading utilities
│   └── btcusd_1min_volspikes.csv
├── models/                    # Trained model files (.pkl)
├── train.py                   # Training script
├── inference.py               # Inference script
└── main.py                    # Main entry point
```

## Model Training

The training process uses:
- **LightGBMClassifier** for both regime detection and signal blending
- Walk-forward validation support (6-month training, 1-month validation windows)
- Rule-based regime labels and future return-based blender labels

## License

[Add your license here]

## Author

Josh Ratnieks

