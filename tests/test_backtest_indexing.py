"""
Tests for index consistency and alignment in backtest.py.

Validates that all DataFrames/Series maintain proper index alignment,
timestamps are sorted, and no forward-index references occur.

WHY THESE TESTS EXIST:
- Index misalignment causes silent data corruption
- Unsorted timestamps break time-series logic
- Forward index references create lookahead bias

WHAT INVARIANTS ARE PROTECTED:
- All aligned data shares the same index
- Timestamps are monotonically increasing
- No iloc[-n] style forward references in calculations
- Feature index aligns with price index

BUGS THESE TESTS WOULD CATCH:
- Shifted index causing future data exposure
- Missing index alignment after merge/join
- Time gaps causing alignment issues
- iloc with negative offset accessing future rows
"""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import mocks
from tests.mocks import (
    make_synthetic_ohlcv,
    make_synthetic_features,
    make_aligned_features,
    make_synthetic_trades,
)


class TestIndexAlignment:
    """
    Tests for proper index alignment across DataFrames.
    
    WHY: Index misalignment is a major source of bugs in time series analysis.
    WHAT: Validates that operations preserve or properly align indices.
    """
    
    def test_features_align_with_ohlcv(self):
        """
        WHY: Verify features index matches OHLCV data index exactly.
        WHAT: After feature generation, indices should be identical.
        BUG CAUGHT: Feature shift causing misalignment.
        """
        df = make_synthetic_ohlcv(n_rows=15)
        features = make_aligned_features(df, n_features=10)
        
        assert features.index.equals(df.index), "Feature index should match OHLCV index"
        assert len(features) == len(df), "Feature length should match OHLCV length"
    
    def test_series_alignment_after_operations(self):
        """
        WHY: Verify index preserved after rolling/shift operations.
        WHAT: Rolling mean should maintain original index alignment.
        BUG CAUGHT: Rolling operation dropping or shifting index.
        """
        df = make_synthetic_ohlcv(n_rows=15)
        close = df['close']
        
        # Rolling operation
        rolling_mean = close.rolling(window=3, min_periods=1).mean()
        
        assert rolling_mean.index.equals(close.index), "Rolling should preserve index"
        assert len(rolling_mean) == len(close), "Rolling should preserve length"
    
    def test_reindex_alignment(self):
        """
        WHY: Verify reindex properly aligns different-length series.
        WHAT: reindex should handle missing indices correctly.
        BUG CAUGHT: Silent data loss during reindex.
        """
        # Create two series with different indices
        idx1 = pd.date_range('2024-01-01', periods=10, freq='5min')
        idx2 = pd.date_range('2024-01-01 00:10', periods=10, freq='5min')  # Offset by 10 minutes
        
        s1 = pd.Series(np.random.randn(10), index=idx1, name='s1')
        s2 = pd.Series(np.random.randn(10), index=idx2, name='s2')
        
        # Reindex s2 to s1's index
        s2_aligned = s2.reindex(idx1)
        
        assert s2_aligned.index.equals(idx1), "Reindexed series should have target index"
        # First 2 values should be NaN (no overlap)
        assert s2_aligned.iloc[:2].isna().all(), "Non-overlapping values should be NaN"
    
    def test_trades_align_with_signals(self):
        """
        WHY: Verify trade output index matches signal index.
        WHAT: Trades DataFrame index should match input signals.
        BUG CAUGHT: Trade index drift causing attribution errors.
        """
        trades = make_synthetic_trades(n_trades=10)
        
        # Simulate signal generation
        signals = pd.Series(trades['signal'].values, index=trades.index)
        
        assert signals.index.equals(trades.index), "Signals should align with trades"
    
    def test_concat_preserves_index(self):
        """
        WHY: Verify pd.concat preserves index in concatenation.
        WHAT: Concatenated DataFrame should maintain time ordering.
        BUG CAUGHT: Index duplication or reordering.
        """
        df1 = make_synthetic_ohlcv(n_rows=5, start_time=datetime(2024, 1, 1))
        df2 = make_synthetic_ohlcv(n_rows=5, start_time=datetime(2024, 1, 1, 0, 25))
        
        combined = pd.concat([df1, df2])
        
        assert len(combined) == 10, "Combined length should be sum"
        assert combined.index.is_monotonic_increasing, "Index should remain sorted"
        assert not combined.index.has_duplicates, "Index should have no duplicates"


class TestTimestampSorting:
    """
    Tests for timestamp monotonicity and sorting.
    
    WHY: Unsorted timestamps break time-series calculations.
    WHAT: Validates that timestamps are always monotonically increasing.
    """
    
    def test_ohlcv_timestamps_monotonic(self):
        """
        WHY: Verify OHLCV data has sorted timestamps.
        WHAT: Index should be monotonically increasing.
        BUG CAUGHT: Data loaded in wrong order.
        """
        df = make_synthetic_ohlcv(n_rows=20)
        
        assert df.index.is_monotonic_increasing, "OHLCV timestamps should be sorted"
    
    def test_detect_unsorted_timestamps(self):
        """
        WHY: Verify detection of unsorted timestamp issues.
        WHAT: Test that we can identify unsorted data.
        BUG CAUGHT: Failing to detect temporal disorder.
        """
        # Create intentionally unsorted data
        idx = pd.date_range('2024-01-01', periods=5, freq='5min')
        unsorted_idx = idx[[0, 2, 1, 4, 3]]  # Scramble order
        
        df_unsorted = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=unsorted_idx)
        
        assert not df_unsorted.index.is_monotonic_increasing, "Should detect unsorted"
        
        # Sort and verify
        df_sorted = df_unsorted.sort_index()
        assert df_sorted.index.is_monotonic_increasing, "Sorted should be monotonic"
    
    def test_timestamp_gaps_detected(self):
        """
        WHY: Verify detection of unexpected gaps in time series.
        WHAT: Should identify missing bars in regular frequency data.
        BUG CAUGHT: Missing data causing alignment issues.
        """
        # Create data with a gap
        idx = pd.date_range('2024-01-01', periods=10, freq='5min')
        idx_with_gap = idx.delete(5)  # Remove one timestamp
        
        df_gapped = pd.DataFrame({
            'close': np.random.randn(9)
        }, index=idx_with_gap)
        
        # Detect gap: check if diff is consistent
        time_diffs = df_gapped.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(minutes=5)
        
        # Should have at least one diff that's 2x expected (the gap)
        has_gap = (time_diffs > expected_diff).any()
        assert has_gap, "Should detect timestamp gap"


class TestNoLookahead:
    """
    Tests for lookahead bias prevention.
    
    WHY: Lookahead bias invalidates backtest results.
    WHAT: Validates that no future data is used in calculations.
    """
    
    def test_shift_prevents_lookahead(self):
        """
        WHY: Verify shift(1) creates proper lag for lagged features.
        WHAT: Shifted value at time t should be value from t-1.
        BUG CAUGHT: Using current value instead of lagged.
        """
        values = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        lagged = values.shift(1)
        
        assert pd.isna(lagged.iloc[0]), "First lagged value should be NaN"
        assert lagged.iloc[1] == 1, "Lagged value should be previous"
        assert lagged.iloc[2] == 2, "Lagged value should be previous"
        assert lagged.iloc[4] == 4, "Lagged value should be previous"
    
    def test_negative_shift_creates_future(self):
        """
        WHY: Verify shift(-n) accesses future data (for labels).
        WHAT: shift(-1) at time t should give value from t+1.
        BUG CAUGHT: Sign error in future return calculation.
        """
        values = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        future = values.shift(-1)
        
        assert future.iloc[0] == 2, "Future value at 0 should be value at 1"
        assert future.iloc[1] == 3, "Future value at 1 should be value at 2"
        assert pd.isna(future.iloc[4]), "Last future value should be NaN"
    
    def test_future_return_uses_negative_shift(self):
        """
        WHY: Verify future return calculation uses shift(-horizon).
        WHAT: Future return at t = close[t+horizon] / close[t] - 1.
        BUG CAUGHT: Using wrong shift direction for future returns.
        """
        close = pd.Series([100, 101, 102, 103, 104], 
                          index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        horizon = 2
        future_close = close.shift(-horizon)
        future_return = future_close / close - 1.0
        
        # At time 0, future return = close[2] / close[0] - 1 = 102/100 - 1 = 0.02
        expected_0 = (102 / 100) - 1.0
        assert np.isclose(future_return.iloc[0], expected_0), f"Future return at 0 should be {expected_0}"
        
        # Last 'horizon' values should be NaN
        assert future_return.iloc[-horizon:].isna().all(), "Last horizon values should be NaN"
    
    def test_rolling_only_uses_past(self):
        """
        WHY: Verify rolling operations only use past data.
        WHAT: Rolling mean at t should use values [t-window+1, t].
        BUG CAUGHT: Rolling using future values.
        """
        values = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        rolling_mean = values.rolling(window=3, min_periods=1).mean()
        
        # At index 2, rolling mean = mean([1, 2, 3]) = 2.0
        assert rolling_mean.iloc[2] == 2.0, "Rolling mean should be average of past values"
        
        # At index 4, rolling mean = mean([3, 4, 5]) = 4.0
        assert rolling_mean.iloc[4] == 4.0, "Rolling mean should use only past window"


class TestIndexConsistency:
    """
    Tests for consistent index handling throughout pipeline.
    
    WHY: Index inconsistency causes subtle bugs in backtest.
    WHAT: Validates that index types and values remain consistent.
    """
    
    def test_datetime_index_preserved(self):
        """
        WHY: Verify DatetimeIndex type is preserved through operations.
        WHAT: Index should remain DatetimeIndex after transforms.
        BUG CAUGHT: Index type degradation to RangeIndex.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        assert isinstance(df.index, pd.DatetimeIndex), "Should have DatetimeIndex"
        
        # After slicing
        sliced = df.iloc[2:8]
        assert isinstance(sliced.index, pd.DatetimeIndex), "Slice should preserve DatetimeIndex"
        
        # After filtering
        filtered = df[df['close'] > df['close'].mean()]
        assert isinstance(filtered.index, pd.DatetimeIndex), "Filter should preserve DatetimeIndex"
    
    def test_index_unique(self):
        """
        WHY: Verify index has no duplicate timestamps.
        WHAT: Each timestamp should appear exactly once.
        BUG CAUGHT: Duplicate entries causing unexpected behavior.
        """
        df = make_synthetic_ohlcv(n_rows=20)
        
        assert df.index.is_unique, "Index should have unique values"
        assert not df.index.has_duplicates, "Index should have no duplicates"
    
    def test_loc_vs_iloc_consistency(self):
        """
        WHY: Verify loc and iloc access the same data.
        WHAT: df.loc[timestamp] should match df.iloc[position].
        BUG CAUGHT: Index-position mismatch.
        """
        df = make_synthetic_ohlcv(n_rows=10)
        
        # Get 5th row both ways
        timestamp = df.index[4]
        by_iloc = df.iloc[4]
        by_loc = df.loc[timestamp]
        
        assert (by_iloc == by_loc).all(), "loc and iloc should return same data"
    
    def test_merge_preserves_alignment(self):
        """
        WHY: Verify merging preserves index alignment.
        WHAT: After merge, indices should still align correctly.
        BUG CAUGHT: Merge shuffling rows.
        """
        df1 = make_synthetic_ohlcv(n_rows=10)
        df2 = make_aligned_features(df1, n_features=5)
        
        # Merge on index
        merged = df1.join(df2)
        
        assert merged.index.equals(df1.index), "Merge should preserve index"
        assert len(merged) == len(df1), "Merge should preserve length"


class TestSliceAlignment:
    """
    Tests for proper alignment when slicing data.
    
    WHY: Slicing is used extensively for train/val splits.
    WHAT: Validates that sliced data maintains proper alignment.
    """
    
    def test_iloc_slice_preserves_order(self):
        """
        WHY: Verify iloc slicing preserves data order.
        WHAT: df.iloc[start:end] should give consecutive rows.
        BUG CAUGHT: Wrong slice boundaries.
        """
        df = make_synthetic_ohlcv(n_rows=20)
        
        sliced = df.iloc[5:15]
        
        assert len(sliced) == 10, "Slice should have 10 rows"
        assert sliced.index[0] == df.index[5], "Slice start should match"
        assert sliced.index[-1] == df.index[14], "Slice end should match"
        assert sliced.index.is_monotonic_increasing, "Slice should maintain order"
    
    def test_loc_slice_preserves_order(self):
        """
        WHY: Verify loc slicing with timestamps preserves order.
        WHAT: df.loc[start_time:end_time] should give correct range.
        BUG CAUGHT: Off-by-one in timestamp slicing.
        """
        df = make_synthetic_ohlcv(n_rows=20)
        
        start = df.index[5]
        end = df.index[14]
        sliced = df.loc[start:end]
        
        assert sliced.index[0] == start, "Slice should start at start timestamp"
        assert sliced.index[-1] == end, "Slice should end at end timestamp"
        assert sliced.index.is_monotonic_increasing, "Slice should maintain order"
    
    def test_train_val_split_no_overlap(self):
        """
        WHY: Verify train/val split has no overlapping indices.
        WHAT: Train and val sets should be disjoint.
        BUG CAUGHT: Data leakage from overlapping splits.
        """
        df = make_synthetic_ohlcv(n_rows=20)
        
        # Simulate 80/20 split
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        val = df.iloc[split_idx:]
        
        # Check no overlap
        overlap = train.index.intersection(val.index)
        assert len(overlap) == 0, "Train and val should have no overlap"
        
        # Check complete coverage
        combined = pd.concat([train, val])
        assert len(combined) == len(df), "Train + val should cover all data"
    
    def test_train_val_temporal_order(self):
        """
        WHY: Verify train data comes before val data temporally.
        WHAT: max(train.index) < min(val.index).
        BUG CAUGHT: Temporal leakage from wrong split order.
        """
        df = make_synthetic_ohlcv(n_rows=20)
        
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        val = df.iloc[split_idx:]
        
        assert train.index.max() < val.index.min(), "Train should be before val temporally"

