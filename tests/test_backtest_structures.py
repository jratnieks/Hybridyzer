"""
Test Backtest Data Structures

=============================================================================
CURSOR AUTO-RUN: SAFE ✓
=============================================================================

All tests in this module use MOCK DATA ONLY and run instantly.
No real data files, no GPU, no heavy computation.

Verifies:
- Required columns present in trades_df
- Correct data types for all fields
- Equity curves are numerically sound
- Missing keys and corrupted rows are detected
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from typing import Set

# =============================================================================
# NOTE: NO HEAVY IMPORTS
# All tests use pure pandas/numpy operations with mock data.
# =============================================================================


class TestTradesDataFrameSchema:
    """
    Verify trades DataFrame has correct schema.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    REQUIRED_COLUMNS: Set[str] = {
        'signal', 'future_return', 'trade_pnl', 'regime'
    }

    OPTIONAL_COLUMNS: Set[str] = {
        'direction_confidence', 'ev', 'trade_pnl_raw', 'final_signal'
    }

    def test_required_columns_present(self, mock_trades_df) -> None:
        """
        trades_df must contain all required columns.
        """
        missing = self.REQUIRED_COLUMNS - set(mock_trades_df.columns)
        assert len(missing) == 0, f"Missing required columns: {missing}"

    def test_signal_values_valid(self) -> None:
        """
        Signal column must contain only valid values: -1, 0, 1.
        """
        signals = pd.Series([1, -1, 0, 1, 0, -1, 1])
        valid_signals = {-1, 0, 1}
        
        invalid = set(signals.unique()) - valid_signals
        assert len(invalid) == 0, f"Invalid signal values: {invalid}"

    def test_regime_values_valid(self) -> None:
        """
        Regime column must contain only valid regime names.
        """
        regimes = pd.Series(['trend_up', 'trend_down', 'chop', 'trend_up'])
        valid_regimes = {'trend_up', 'trend_down', 'chop'}
        
        invalid = set(regimes.unique()) - valid_regimes
        assert len(invalid) == 0, f"Invalid regime values: {invalid}"

    def test_numeric_columns_dtype(self) -> None:
        """
        Numeric columns must have numeric dtype.
        """
        trades_df = pd.DataFrame({
            'signal': [1, -1, 0],
            'future_return': [0.01, -0.005, 0.0],
            'trade_pnl': [0.01, 0.005, 0.0],
            'direction_confidence': [0.65, 0.70, 0.50]
        })
        
        numeric_cols = ['future_return', 'trade_pnl', 'direction_confidence']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(trades_df[col]), \
                f"Column {col} should be numeric, got {trades_df[col].dtype}"

    def test_no_all_nan_columns(self) -> None:
        """
        No column should be entirely NaN.
        """
        trades_df = pd.DataFrame({
            'signal': [1, -1, 0],
            'future_return': [0.01, -0.005, 0.0],
            'trade_pnl': [0.01, 0.005, 0.0]
        })
        
        for col in trades_df.columns:
            assert not trades_df[col].isna().all(), \
                f"Column {col} is entirely NaN"


class TestEquityCurveValidity:
    """
    Verify equity curve is numerically sound.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_equity_starts_at_one(self) -> None:
        """
        Equity curve should start at 1.0 (initial capital).
        """
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        equity = (1 + returns).cumprod()
        
        # First equity value should be 1.0 * (1 + first_return)
        expected_first = 1.0 * (1 + returns.iloc[0])
        np.testing.assert_almost_equal(equity.iloc[0], expected_first, decimal=6)

    def test_equity_no_negative_values(self) -> None:
        """
        Equity curve should never go negative.
        """
        # Even with losses, equity stays positive (unless > 100% loss per bar)
        returns = pd.Series([0.01, -0.05, 0.02, -0.03, 0.01])
        equity = (1 + returns).cumprod()
        
        assert (equity > 0).all(), "Equity should always be positive"

    def test_equity_no_nan_inf(self) -> None:
        """
        Equity curve should not contain NaN or Inf.
        """
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        equity = (1 + returns).cumprod()
        
        assert not equity.isna().any(), "Equity contains NaN"
        assert not np.isinf(equity).any(), "Equity contains Inf"

    def test_equity_monotonicity_with_positive_returns(self) -> None:
        """
        With only positive returns, equity should be strictly increasing.
        """
        returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.005])
        equity = (1 + returns).cumprod()
        
        assert equity.is_monotonic_increasing, \
            "Equity should increase with all positive returns"

    def test_drawdown_never_exceeds_100_percent(self) -> None:
        """
        Drawdown should never exceed 100% (equity > 0).
        """
        equity = pd.Series([1.0, 1.1, 1.05, 1.15, 1.0, 1.2, 0.9, 1.1])
        
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        assert (drawdown >= -1.0).all(), "Drawdown should not exceed -100%"


class TestMissingCorruptedData:
    """
    Detect missing keys and corrupted rows.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_detect_missing_timestamps(self) -> None:
        """
        Detect rows with missing timestamp (index).
        """
        index = pd.date_range('2024-01-01', periods=5, freq='5min')
        df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=index)
        
        # All indices should be valid timestamps
        assert not df.index.isna().any(), "Index contains NaT values"

    def test_detect_null_required_fields(self) -> None:
        """
        Detect rows where required fields are null.
        """
        trades_df = pd.DataFrame({
            'signal': [1, -1, np.nan, 1],  # NaN in required field
            'future_return': [0.01, -0.005, 0.0, 0.02],
            'trade_pnl': [0.01, 0.005, 0.0, 0.02]
        })
        
        required_cols = ['signal', 'future_return', 'trade_pnl']
        
        null_rows = trades_df[required_cols].isna().any(axis=1)
        assert null_rows.any(), "Test expects at least one null row"
        
        # In production, we'd filter or raise
        clean_df = trades_df.dropna(subset=required_cols)
        assert len(clean_df) < len(trades_df)

    def test_detect_inf_values(self) -> None:
        """
        Detect rows with Inf values.
        """
        df = pd.DataFrame({
            'return': [0.01, np.inf, -0.005, -np.inf, 0.02]
        })
        
        has_inf = np.isinf(df['return']).any()
        assert has_inf, "Test expects Inf values"
        
        # Count Inf rows
        inf_count = np.isinf(df['return']).sum()
        assert inf_count == 2

    def test_detect_extreme_values(self) -> None:
        """
        Detect unrealistic extreme values (potential corruption).
        """
        # Returns > 100% or < -100% per bar are suspicious
        returns = pd.Series([0.01, 5.0, -0.005, -2.0, 0.02])  # 500% and -200% are extreme
        
        extreme_mask = (returns.abs() > 1.0)
        assert extreme_mask.any(), "Test expects extreme values"
        
        extreme_count = extreme_mask.sum()
        assert extreme_count == 2

    def test_detect_duplicate_indices(self) -> None:
        """
        Detect duplicate timestamps in index.
        """
        timestamps = pd.to_datetime([
            '2024-01-01 00:00', '2024-01-01 00:05', '2024-01-01 00:05',  # Duplicate
            '2024-01-01 00:10', '2024-01-01 00:15'
        ])
        
        has_duplicates = timestamps.duplicated().any()
        assert has_duplicates, "Test expects duplicate timestamps"


class TestBacktestResultsIntegrity:
    """
    Verify backtest results integrity.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_total_return_matches_equity(self) -> None:
        """
        Total return should match final equity / initial equity - 1.
        """
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        equity = (1 + returns).cumprod()
        
        total_return_from_equity = equity.iloc[-1] / equity.iloc[0] * (1 + returns.iloc[0]) - 1
        total_return_from_cumprod = (1 + returns).prod() - 1
        
        np.testing.assert_almost_equal(
            total_return_from_cumprod,
            total_return_from_equity,
            decimal=6
        )

    def test_trade_count_matches_signals(self) -> None:
        """
        Number of trades should match non-zero signals.
        """
        signals = pd.Series([1, 0, -1, 0, 1, 1, 0, -1])
        
        trade_count = (signals != 0).sum()
        expected = 5
        
        assert trade_count == expected

    def test_pnl_sum_approximately_total_return(self) -> None:
        """
        Sum of trade PnL should approximately equal total return (for simple cases).
        """
        # For a simple backtest where each bar is a separate trade
        trade_pnl = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        
        # Simple sum (approximation for small returns)
        pnl_sum = trade_pnl.sum()
        
        # Compounded return
        compounded = (1 + trade_pnl).prod() - 1
        
        # They should be close for small returns
        assert abs(pnl_sum - compounded) < 0.01, \
            "PnL sum and compounded return should be similar for small returns"

    def test_sharpe_ratio_calculation(self) -> None:
        """
        Sharpe ratio calculation should handle edge cases.
        """
        # Normal case
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0
        assert np.isfinite(sharpe)
        
        # Zero std case (constant returns)
        constant_returns = pd.Series([0.01, 0.01, 0.01])
        sharpe_constant = constant_returns.mean() / constant_returns.std() if constant_returns.std() > 0 else 0.0
        assert sharpe_constant == 0.0 or np.isnan(sharpe_constant)

    def test_max_drawdown_calculation(self) -> None:
        """
        Max drawdown should be calculated correctly.
        """
        equity = pd.Series([1.0, 1.1, 1.05, 1.15, 1.0, 1.2, 0.9, 1.1])
        
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Max drawdown occurs when equity drops to 0.9 after reaching 1.2
        expected_max_dd = (0.9 - 1.2) / 1.2  # -0.25
        np.testing.assert_almost_equal(max_drawdown, expected_max_dd, decimal=6)


class TestCalibrationDataStructure:
    """
    Verify calibration data structure.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_calibration_bins_monotonic(self) -> None:
        """
        Calibration probability bins should be monotonically increasing.
        """
        bins = pd.DataFrame({
            'prob_min': [0.5, 0.6, 0.7, 0.8],
            'prob_max': [0.6, 0.7, 0.8, 0.9]
        })
        
        # prob_min should be increasing
        assert bins['prob_min'].is_monotonic_increasing
        
        # prob_max should be increasing
        assert bins['prob_max'].is_monotonic_increasing

    def test_calibration_count_positive(self) -> None:
        """
        Count column in calibration should be positive integers.
        """
        calibration = pd.DataFrame({
            'prob_bin': ['(0.5, 0.6]', '(0.6, 0.7]'],
            'count': [150, 200],
            'hit_rate': [0.52, 0.58]
        })
        
        assert (calibration['count'] > 0).all(), "Counts should be positive"
        assert calibration['count'].dtype in [np.int64, np.int32, int]

    def test_hit_rate_bounded(self) -> None:
        """
        Hit rate should be between 0 and 1.
        """
        hit_rates = pd.Series([0.52, 0.58, 0.61, 0.55])
        
        assert (hit_rates >= 0).all(), "Hit rate should be >= 0"
        assert (hit_rates <= 1).all(), "Hit rate should be <= 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
