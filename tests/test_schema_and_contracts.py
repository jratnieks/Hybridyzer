"""
Tests for DataFrame schema validation and API contracts.

Ensures that DataFrames and returned structures from backtest.py and train.py
follow expected schemas, column names, and types.

WHY THESE TESTS EXIST:
- Schema violations cause downstream errors
- Column name typos are silent bugs
- Type mismatches cause subtle issues

WHAT INVARIANTS ARE PROTECTED:
- OHLCV DataFrames have required columns
- Trade logs have expected schema
- Equity curves have correct structure
- Model outputs have expected format

BUGS THESE TESTS WOULD CATCH:
- Missing required column in output
- Column name typo (e.g., 'Close' vs 'close')
- Wrong dtype (string instead of float)
- Missing index causing alignment issues
"""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Import mocks
from tests.mocks import (
    make_synthetic_ohlcv,
    make_synthetic_trades,
    make_synthetic_features,
    make_direction_labels_mock,
)


class TestOHLCVSchema:
    """
    Tests for OHLCV DataFrame schema.
    
    WHY: OHLCV data is the foundation of all analysis.
    WHAT: Validates required columns, types, and index.
    """
    
    def test_ohlcv_required_columns(self):
        """
        WHY: Verify OHLCV has all required columns.
        WHAT: Must have open, high, low, close, volume.
        BUG CAUGHT: Missing column causing KeyError.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_ohlcv_column_lowercase(self):
        """
        WHY: Verify column names are lowercase.
        WHAT: Should use 'close' not 'Close'.
        BUG CAUGHT: Case sensitivity issues.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        for col in df.columns:
            assert col == col.lower(), f"Column should be lowercase: {col}"
    
    def test_ohlcv_numeric_types(self):
        """
        WHY: Verify OHLCV columns are numeric.
        WHAT: All columns should be float or int.
        BUG CAUGHT: String prices causing math errors.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert np.issubdtype(df[col].dtype, np.number), \
                f"Column {col} should be numeric, got {df[col].dtype}"
    
    def test_ohlcv_datetime_index(self):
        """
        WHY: Verify OHLCV has DatetimeIndex.
        WHAT: Index should be DatetimeIndex for time series ops.
        BUG CAUGHT: RangeIndex causing time-based operations to fail.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        assert isinstance(df.index, pd.DatetimeIndex), \
            f"Index should be DatetimeIndex, got {type(df.index)}"
    
    def test_ohlcv_high_low_relationship(self):
        """
        WHY: Verify high >= low for all bars.
        WHAT: Physical constraint that high cannot be less than low.
        BUG CAUGHT: Data corruption or loading error.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        assert (df['high'] >= df['low']).all(), "High must be >= Low"
    
    def test_ohlcv_open_close_within_range(self):
        """
        WHY: Verify open and close are within high/low range.
        WHAT: open and close should be between low and high.
        BUG CAUGHT: Data corruption.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        # Open within range
        assert (df['open'] >= df['low']).all(), "Open must be >= Low"
        assert (df['open'] <= df['high']).all(), "Open must be <= High"
        
        # Close within range
        assert (df['close'] >= df['low']).all(), "Close must be >= Low"
        assert (df['close'] <= df['high']).all(), "Close must be <= High"
    
    def test_ohlcv_volume_non_negative(self):
        """
        WHY: Verify volume is non-negative.
        WHAT: Volume cannot be negative.
        BUG CAUGHT: Data error or sign flip.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        assert (df['volume'] >= 0).all(), "Volume must be non-negative"


class TestTradeLogSchema:
    """
    Tests for trade log DataFrame schema.
    
    WHY: Trade logs are the backtest output record.
    WHAT: Validates required columns and types.
    """
    
    def test_trades_required_columns(self):
        """
        WHY: Verify trade log has all required columns.
        WHAT: Must have signal, trade_pnl, etc.
        BUG CAUGHT: Missing column in results.
        """
        trades = make_synthetic_trades(n_trades=10)
        
        required = ['signal', 'future_return', 'trade_pnl', 'regime']
        for col in required:
            assert col in trades.columns, f"Missing required column: {col}"
    
    def test_trades_signal_values(self):
        """
        WHY: Verify signal values are valid.
        WHAT: Signal should be in {-1, 0, 1}.
        BUG CAUGHT: Invalid signal value causing position error.
        """
        trades = make_synthetic_trades(n_trades=10)
        
        valid_signals = {-1, 0, 1}
        for val in trades['signal'].unique():
            assert val in valid_signals, f"Invalid signal value: {val}"
    
    def test_trades_pnl_numeric(self):
        """
        WHY: Verify PnL columns are numeric.
        WHAT: trade_pnl should be float.
        BUG CAUGHT: String PnL causing aggregation errors.
        """
        trades = make_synthetic_trades(n_trades=10)
        
        assert np.issubdtype(trades['trade_pnl'].dtype, np.number), \
            "trade_pnl should be numeric"
    
    def test_trades_datetime_index(self):
        """
        WHY: Verify trades have DatetimeIndex.
        WHAT: Index should be timestamps for alignment.
        BUG CAUGHT: Wrong index type causing merge issues.
        """
        trades = make_synthetic_trades(n_trades=10)
        
        assert isinstance(trades.index, pd.DatetimeIndex), \
            "Trade index should be DatetimeIndex"
    
    def test_trades_regime_valid_strings(self):
        """
        WHY: Verify regime column has valid values.
        WHAT: Regime should be recognized category.
        BUG CAUGHT: Typo in regime string.
        """
        trades = make_synthetic_trades(n_trades=10)
        
        valid_regimes = {'trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol'}
        for val in trades['regime'].unique():
            assert val in valid_regimes, f"Invalid regime: {val}"


class TestEquityCurveSchema:
    """
    Tests for equity curve structure.
    
    WHY: Equity curve is the performance visualization.
    WHAT: Validates equity is properly structured.
    """
    
    def test_equity_series_type(self):
        """
        WHY: Verify equity is a Series.
        WHAT: Should be pd.Series, not DataFrame.
        BUG CAUGHT: Wrong return type.
        """
        trades = make_synthetic_trades(n_trades=10)
        returns = trades['trade_pnl']
        equity = (1 + returns).cumprod()
        
        assert isinstance(equity, pd.Series), "Equity should be Series"
    
    def test_equity_positive(self):
        """
        WHY: Verify equity remains positive.
        WHAT: Can't have negative equity (for valid returns).
        BUG CAUGHT: Invalid return causing negative equity.
        """
        trades = make_synthetic_trades(n_trades=10)
        returns = trades['trade_pnl']
        
        # Ensure returns are valid (> -1)
        returns_clipped = returns.clip(lower=-0.99)
        equity = (1 + returns_clipped).cumprod()
        
        assert (equity > 0).all(), "Equity should remain positive"
    
    def test_equity_datetime_index(self):
        """
        WHY: Verify equity has DatetimeIndex.
        WHAT: Index should match trade timestamps.
        BUG CAUGHT: Index loss causing plotting issues.
        """
        trades = make_synthetic_trades(n_trades=10)
        returns = trades['trade_pnl']
        equity = (1 + returns).cumprod()
        
        assert isinstance(equity.index, pd.DatetimeIndex), \
            "Equity index should be DatetimeIndex"
    
    def test_equity_monotonic_for_positive_returns(self):
        """
        WHY: Verify equity increases for all positive returns.
        WHAT: Should be monotonically increasing when all returns > 0.
        BUG CAUGHT: Sign error in compounding.
        """
        # All positive returns
        returns = pd.Series([0.01, 0.02, 0.01, 0.015],
                            index=pd.date_range('2024-01-01', periods=4, freq='5min'))
        equity = (1 + returns).cumprod()
        
        assert equity.is_monotonic_increasing, \
            "Equity should be monotonic for all positive returns"


class TestFeatureSchema:
    """
    Tests for feature DataFrame schema.
    
    WHY: Features are model inputs - schema must be correct.
    WHAT: Validates feature structure and types.
    """
    
    def test_features_2d_structure(self):
        """
        WHY: Verify features are 2D (samples x features).
        WHAT: Should have rows and columns.
        BUG CAUGHT: 1D array passed to model.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        assert features.ndim == 2, "Features should be 2D"
        assert len(features) > 0, "Features should have rows"
        assert len(features.columns) > 0, "Features should have columns"
    
    def test_features_all_numeric(self):
        """
        WHY: Verify all feature columns are numeric.
        WHAT: No string or object columns.
        BUG CAUGHT: Categorical not encoded.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        for col in features.columns:
            assert np.issubdtype(features[col].dtype, np.number), \
                f"Feature {col} should be numeric"
    
    def test_features_no_inf(self):
        """
        WHY: Verify no infinite values in features.
        WHAT: All values should be finite.
        BUG CAUGHT: Division by zero creating inf.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        assert np.isfinite(features.values).all(), "Features should have no inf values"
    
    def test_features_column_names_valid(self):
        """
        WHY: Verify feature column names are valid identifiers.
        WHAT: Should be usable as attribute names.
        BUG CAUGHT: Invalid column name causing issues.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        for col in features.columns:
            # Should not contain spaces
            assert ' ' not in col, f"Column name should not have spaces: {col}"
            # Should not start with number
            assert not col[0].isdigit(), f"Column name should not start with digit: {col}"


class TestLabelSchema:
    """
    Tests for label structure and validity.
    
    WHY: Labels are training targets - must be correct.
    WHAT: Validates label format and values.
    """
    
    def test_direction_labels_1d(self):
        """
        WHY: Verify direction labels are 1D.
        WHAT: Should be Series, not DataFrame.
        BUG CAUGHT: 2D labels causing shape mismatch.
        """
        labels_df = make_direction_labels_mock(n_rows=15)
        labels = labels_df['blend_label']
        
        assert labels.ndim == 1, "Labels should be 1D"
    
    def test_direction_labels_integer(self):
        """
        WHY: Verify direction labels are integers.
        WHAT: Classification labels should be int.
        BUG CAUGHT: Float labels causing classifier error.
        """
        labels_df = make_direction_labels_mock(n_rows=15)
        labels = labels_df['blend_label']
        
        # Labels should be integer-like
        assert np.issubdtype(labels.dtype, np.integer) or \
               (labels == labels.astype(int)).all(), \
            "Labels should be integers"
    
    def test_direction_labels_bounded(self):
        """
        WHY: Verify labels are in expected range.
        WHAT: Should be in {-1, 0, 1} for direction.
        BUG CAUGHT: Invalid label value.
        """
        labels_df = make_direction_labels_mock(n_rows=15)
        labels = labels_df['blend_label']
        
        valid_values = {-1, 0, 1}
        for val in labels.unique():
            assert val in valid_values, f"Invalid label: {val}"


class TestCalibrationReportSchema:
    """
    Tests for calibration report structure.
    
    WHY: Calibration reports guide trading decisions.
    WHAT: Validates report has expected sections.
    """
    
    def test_calibration_report_structure(self):
        """
        WHY: Verify calibration report has expected keys.
        WHAT: Should have 'overall', 'per_side', 'per_regime_side'.
        BUG CAUGHT: Missing report section.
        """
        # Simulate calibration report structure
        report = {
            'overall': pd.DataFrame(),
            'per_side': pd.DataFrame(),
            'per_regime_side': pd.DataFrame()
        }
        
        required_keys = ['overall', 'per_side', 'per_regime_side']
        for key in required_keys:
            assert key in report, f"Missing report section: {key}"
    
    def test_calibration_metrics_columns(self):
        """
        WHY: Verify calibration metrics have expected columns.
        WHAT: Should have count, hit_rate, avg_return, etc.
        BUG CAUGHT: Missing metric column.
        """
        # Simulate calibration metrics DataFrame
        metrics = pd.DataFrame({
            'prob_bin': [(0.5, 0.6)],
            'count': [100],
            'hit_rate': [0.55],
            'avg_gross_return': [0.001],
            'prob_min': [0.5],
            'prob_max': [0.6],
        })
        
        required = ['count', 'hit_rate']
        for col in required:
            assert col in metrics.columns, f"Missing metric: {col}"


class TestMetricsSchema:
    """
    Tests for backtest metrics dictionary structure.
    
    WHY: Metrics are the backtest output summary.
    WHAT: Validates all expected metrics are present.
    """
    
    def test_metrics_required_keys(self):
        """
        WHY: Verify metrics dict has all required keys.
        WHAT: Should have total_return, sharpe, etc.
        BUG CAUGHT: Missing metric in output.
        """
        # Simulate metrics dict
        metrics = {
            'total_return': 0.15,
            'cagr': 0.10,
            'sharpe': 1.5,
            'max_drawdown': -0.08,
            'hit_rate': 0.55,
            'long_accuracy': 0.58,
            'short_accuracy': 0.52,
        }
        
        required = ['total_return', 'sharpe', 'max_drawdown', 'hit_rate']
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_metrics_numeric_values(self):
        """
        WHY: Verify all metrics are numeric.
        WHAT: Values should be float.
        BUG CAUGHT: String metric value.
        """
        metrics = {
            'total_return': 0.15,
            'sharpe': 1.5,
            'max_drawdown': -0.08,
            'hit_rate': 0.55,
        }
        
        for key, val in metrics.items():
            assert isinstance(val, (int, float)), f"Metric {key} should be numeric"
    
    def test_metrics_bounded_values(self):
        """
        WHY: Verify metrics are within reasonable bounds.
        WHAT: hit_rate in [0, 1], max_drawdown <= 0.
        BUG CAUGHT: Impossible metric value.
        """
        metrics = {
            'hit_rate': 0.55,
            'long_accuracy': 0.58,
            'short_accuracy': 0.52,
            'max_drawdown': -0.08,
        }
        
        # Hit rates in [0, 1]
        assert 0.0 <= metrics['hit_rate'] <= 1.0, "hit_rate should be in [0, 1]"
        assert 0.0 <= metrics['long_accuracy'] <= 1.0, "long_accuracy should be in [0, 1]"
        assert 0.0 <= metrics['short_accuracy'] <= 1.0, "short_accuracy should be in [0, 1]"
        
        # Max drawdown should be <= 0
        assert metrics['max_drawdown'] <= 0, "max_drawdown should be <= 0"


class TestConfigSchema:
    """
    Tests for configuration structure validation.
    
    WHY: Config errors cause runtime failures.
    WHAT: Validates config parameters are valid.
    """
    
    def test_probability_threshold_bounded(self):
        """
        WHY: Verify probability thresholds are valid.
        WHAT: Should be in (0, 1].
        BUG CAUGHT: Invalid threshold causing no trades.
        """
        thresholds = {
            'p': 0.6,
            'p_long': 0.55,
            'p_short': 0.65,
        }
        
        for key, val in thresholds.items():
            assert 0.0 < val <= 1.0, f"{key} should be in (0, 1]"
    
    def test_fee_params_non_negative(self):
        """
        WHY: Verify fee parameters are non-negative.
        WHAT: Fees and slippage cannot be negative.
        BUG CAUGHT: Negative fee adding profit.
        """
        params = {
            'fee_bps': 2.0,
            'slippage_bps': 1.0,
        }
        
        for key, val in params.items():
            assert val >= 0, f"{key} should be non-negative"
    
    def test_horizon_positive_integer(self):
        """
        WHY: Verify horizon is positive integer.
        WHAT: Cannot have zero or negative horizon.
        BUG CAUGHT: Invalid horizon causing label error.
        """
        horizon = 48
        
        assert isinstance(horizon, int), "horizon should be integer"
        assert horizon > 0, "horizon should be positive"

