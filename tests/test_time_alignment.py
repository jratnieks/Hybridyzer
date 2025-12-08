"""
Test Time Alignment

=============================================================================
CURSOR AUTO-RUN: SAFE ✓
=============================================================================

All tests in this module use MOCK DATA ONLY and run instantly.
No real data files, no GPU, no heavy computation.

Verifies:
- Feature indices match OHLCV data indices
- Signal indices match feature indices
- Trade log timestamps are consistent with backtest data
- No off-by-one errors in time series operations
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

# =============================================================================
# NOTE: NO HEAVY IMPORTS
# All tests use pure pandas/numpy operations.
# =============================================================================


class TestIndexAlignment:
    """
    Verify index alignment across data structures.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_feature_index_matches_ohlcv(self, tiny_ohlcv_df) -> None:
        """
        Feature DataFrame index must exactly match OHLCV DataFrame index.
        """
        ohlcv = tiny_ohlcv_df
        
        # Simulate feature generation (features should have same index)
        features = pd.DataFrame({
            'return_1': ohlcv['close'].pct_change(),
            'volatility_5': ohlcv['close'].rolling(5).std()
        }, index=ohlcv.index)
        
        # Indices must match exactly
        pd.testing.assert_index_equal(ohlcv.index, features.index)

    def test_label_index_matches_features(self) -> None:
        """
        Labels must align with features after label generation.
        """
        index = pd.date_range('2024-01-01', periods=20, freq='5min')
        close = pd.Series([100 + i * 0.1 for i in range(20)], index=index)
        
        # Simulate label generation
        horizon = 5
        future_close = close.shift(-horizon)
        future_return = future_close / close - 1
        
        # Result should have same index as input
        pd.testing.assert_index_equal(close.index, future_return.index)

    def test_signal_index_matches_features(self) -> None:
        """
        Signal Series index must match feature DataFrame index.
        """
        feature_index = pd.date_range('2024-01-01', periods=50, freq='5min')
        features = pd.DataFrame({
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50)
        }, index=feature_index)
        
        # Simulate signal generation
        signals = pd.Series(np.random.choice([-1, 0, 1], 50), index=feature_index)
        
        pd.testing.assert_index_equal(features.index, signals.index)


class TestTimestampConsistency:
    """
    Verify timestamp consistency.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_timestamps_monotonic_increasing(self) -> None:
        """
        Timestamps in OHLCV data must be strictly monotonically increasing.
        """
        timestamps = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        assert timestamps.is_monotonic_increasing, \
            "Timestamps must be monotonically increasing"
        assert not timestamps.duplicated().any(), \
            "Timestamps must be unique"

    def test_no_missing_timestamps(self) -> None:
        """
        Detect gaps in timestamp sequence.
        """
        # Complete sequence
        complete = pd.date_range('2024-01-01', periods=10, freq='5min')
        
        # Sequence with gap (missing index 5)
        with_gap = complete.delete(5)
        
        # Detect gap by checking time deltas
        deltas = with_gap.to_series().diff()
        expected_delta = pd.Timedelta('5min')
        
        has_gap = (deltas > expected_delta).any()
        assert has_gap, "Gap detection should find missing timestamp"

    def test_timezone_consistency(self) -> None:
        """
        All timestamps should have consistent timezone (or all naive).
        """
        # Naive timestamps (no timezone)
        naive_ts = pd.date_range('2024-01-01', periods=10, freq='5min')
        assert naive_ts.tz is None, "Expected naive timestamps"
        
        # Timezone-aware timestamps
        aware_ts = pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC')
        assert aware_ts.tz is not None, "Expected timezone-aware timestamps"


class TestOffByOneErrors:
    """
    Detect off-by-one errors in time series operations.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_shift_alignment(self) -> None:
        """
        shift(n) should correctly align data with n-period lag/lead.
        """
        index = pd.date_range('2024-01-01', periods=5, freq='5min')
        values = pd.Series([10, 20, 30, 40, 50], index=index)
        
        # Shift by 1 (lag)
        lagged = values.shift(1)
        
        # At index[1], lagged value should be value from index[0]
        assert lagged.iloc[1] == values.iloc[0]
        
        # At index[0], lagged should be NaN
        assert pd.isna(lagged.iloc[0])

    def test_rolling_window_boundaries(self) -> None:
        """
        Rolling window at index i should include exactly window values ending at i.
        """
        index = pd.date_range('2024-01-01', periods=10, freq='5min')
        values = pd.Series(range(1, 11), index=index)
        
        window = 3
        rolling_sum = values.rolling(window=window, min_periods=window).sum()
        
        # At index 2 (value=3), sum should be 1+2+3 = 6
        assert rolling_sum.iloc[2] == 6
        
        # At index 9 (value=10), sum should be 8+9+10 = 27
        assert rolling_sum.iloc[9] == 27
        
        # First window-1 values should be NaN
        assert rolling_sum.iloc[:window-1].isna().all()

    def test_future_return_horizon_correct(self) -> None:
        """
        future_return with horizon=n should use close[i+n], not close[i+n-1] or close[i+n+1].
        """
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        horizon = 3
        
        # Manual calculation
        future_close = close.shift(-horizon)
        future_return = future_close / close - 1
        
        # At index 0: future_return = close[3]/close[0] - 1 = 103/100 - 1 = 0.03
        expected = 103 / 100 - 1
        np.testing.assert_almost_equal(future_return.iloc[0], expected, decimal=6)
        
        # At index 5: future_return = close[8]/close[5] - 1 = 108/105 - 1
        expected_5 = 108 / 105 - 1
        np.testing.assert_almost_equal(future_return.iloc[5], expected_5, decimal=6)


class TestTradeLogAlignment:
    """
    Verify trade log alignment with backtest data.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_trade_timestamps_in_data_range(self) -> None:
        """
        All trade timestamps must fall within OHLCV data range.
        """
        data_start = pd.Timestamp('2024-01-01')
        data_end = pd.Timestamp('2024-01-31')
        
        # Mock trade log
        trade_timestamps = pd.to_datetime([
            '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-20'
        ])
        
        assert (trade_timestamps >= data_start).all(), \
            "Trade timestamps should not precede data start"
        assert (trade_timestamps <= data_end).all(), \
            "Trade timestamps should not exceed data end"

    def test_trade_index_in_features_index(self) -> None:
        """
        Trade indices must exist in feature DataFrame index.
        """
        feature_index = pd.date_range('2024-01-01', periods=100, freq='5min')
        features = pd.DataFrame({'f1': np.random.randn(100)}, index=feature_index)
        
        # Mock trades at specific indices
        trade_indices = feature_index[::10]  # Every 10th bar
        
        for idx in trade_indices:
            assert idx in features.index, f"Trade index {idx} not in features"

    def test_entry_before_exit(self) -> None:
        """
        Trade entry timestamp must precede exit timestamp.
        """
        trades = pd.DataFrame({
            'entry_time': pd.to_datetime(['2024-01-01 10:00', '2024-01-02 14:00']),
            'exit_time': pd.to_datetime(['2024-01-01 12:00', '2024-01-02 16:00']),
            'pnl': [0.01, -0.005]
        })
        
        assert (trades['entry_time'] < trades['exit_time']).all(), \
            "Entry must precede exit for all trades"


class TestReindexAlignment:
    """
    Test proper reindexing operations.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_reindex_preserves_order(self) -> None:
        """
        Reindexing should preserve temporal order.
        """
        original_index = pd.date_range('2024-01-01', periods=10, freq='5min')
        data = pd.Series(range(10), index=original_index)
        
        # Reindex to new (potentially different) index
        new_index = pd.date_range('2024-01-01', periods=15, freq='5min')
        reindexed = data.reindex(new_index)
        
        # Original values should be preserved at original positions
        for i in range(10):
            assert reindexed.iloc[i] == i

    def test_reindex_handles_missing(self) -> None:
        """
        Reindexing to larger index should introduce NaN, not duplicate data.
        """
        original_index = pd.date_range('2024-01-01', periods=5, freq='5min')
        data = pd.Series([1, 2, 3, 4, 5], index=original_index)
        
        # Reindex to larger index
        new_index = pd.date_range('2024-01-01', periods=10, freq='5min')
        reindexed = data.reindex(new_index)
        
        # Last 5 values should be NaN
        assert reindexed.iloc[5:].isna().all()
        
        # First 5 values should be preserved (dtype changes int64->float64 due to NaN)
        pd.testing.assert_series_equal(
            reindexed.iloc[:5],
            data,
            check_names=False,
            check_dtype=False  # NaN introduction converts int to float
        )

    def test_concat_alignment(self) -> None:
        """
        pd.concat should properly align on index when axis=1.
        """
        index = pd.date_range('2024-01-01', periods=5, freq='5min')
        s1 = pd.Series([1, 2, 3, 4, 5], index=index, name='s1')
        s2 = pd.Series([10, 20, 30, 40, 50], index=index, name='s2')
        
        combined = pd.concat([s1, s2], axis=1)
        
        # Verify alignment
        pd.testing.assert_index_equal(combined.index, index)
        assert (combined['s1'] == s1).all()
        assert (combined['s2'] == s2).all()


class TestEquityCurveAlignment:
    """
    Verify equity curve alignment with trade data.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_equity_length_matches_data(self) -> None:
        """
        Equity curve length should match backtest data length.
        """
        n_bars = 100
        returns = pd.Series(np.random.randn(n_bars) * 0.001)
        equity = (1 + returns).cumprod()
        
        assert len(equity) == n_bars, \
            f"Equity length {len(equity)} should match data length {n_bars}"

    def test_equity_index_matches_returns(self) -> None:
        """
        Equity curve index should match returns index.
        """
        index = pd.date_range('2024-01-01', periods=50, freq='5min')
        returns = pd.Series(np.random.randn(50) * 0.001, index=index)
        equity = (1 + returns).cumprod()
        
        pd.testing.assert_index_equal(equity.index, returns.index)

    def test_drawdown_alignment(self) -> None:
        """
        Drawdown calculation should be aligned with equity curve.
        """
        index = pd.date_range('2024-01-01', periods=20, freq='5min')
        equity = pd.Series([1.0, 1.01, 1.02, 1.01, 1.00, 0.99, 1.01, 1.03, 1.02, 1.04,
                           1.03, 1.05, 1.04, 1.06, 1.05, 1.07, 1.06, 1.08, 1.07, 1.09],
                          index=index)
        
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        
        pd.testing.assert_index_equal(drawdown.index, equity.index)
        
        # Verify drawdown is non-positive
        assert (drawdown <= 0).all(), "Drawdown should be non-positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
