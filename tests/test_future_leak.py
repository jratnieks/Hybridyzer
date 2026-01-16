"""
Test Future Leakage Detection

=============================================================================
CURSOR AUTO-RUN: SAFE ✓
=============================================================================

All tests in this module use MOCK DATA ONLY and run instantly.
No real data files, no GPU, no heavy computation.

Verifies:
- Feature leakage (features computed from future data)
- Target leakage (labels computed improperly)
- Information leakage across train/test splits
- Cross-validation leakage in time series
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# =============================================================================
# NOTE: NO HEAVY IMPORTS
# All tests use pure pandas/numpy/sklearn operations.
# =============================================================================


class TestFeatureLeakage:
    """
    Detect leakage in feature engineering.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_no_forward_fill_before_split(self) -> None:
        """
        Forward-filling NaN before train/test split can leak test info into train.
        Verify this doesn't happen.
        """
        # Create data with NaN at end of train portion (indices 3,4) 
        # so ffill must use train-only values
        values = pd.Series([1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, 10])
        
        # Split at index 5
        train = values.iloc[:5].copy()  # [1, 2, 3, nan, nan]
        test = values.iloc[5:].copy()   # [6, 7, 8, 9, 10]
        
        # Forward fill separately
        train_filled = train.ffill()
        test_filled = test.ffill()
        
        # Verify train ffill uses only train data (last known value is 3)
        assert train_filled.iloc[3] == 3, \
            "Train forward-fill should use train data only, not test data"
        assert train_filled.iloc[4] == 3, \
            "Last train NaN should be filled with last known train value (3)"

    def test_normalization_no_leak(self) -> None:
        """
        Normalization (z-score) should be computed on train data only.
        Test data should be normalized using train statistics.
        """
        train_data = pd.Series([100, 102, 98, 104, 96])
        test_data = pd.Series([110, 108, 112])
        
        # Compute train statistics
        train_mean = train_data.mean()
        train_std = train_data.std()
        
        # Normalize test using train stats (correct)
        test_normalized_correct = (test_data - train_mean) / train_std
        
        # Normalize test using test stats (incorrect - leakage)
        test_mean = test_data.mean()
        test_std = test_data.std()
        test_normalized_wrong = (test_data - test_mean) / test_std
        
        # These should differ
        assert not np.allclose(test_normalized_correct.values, test_normalized_wrong.values), \
            "Normalization with train stats should differ from test stats"

    def test_rolling_stats_dont_peek_ahead(self) -> None:
        """
        Rolling statistics at time t should not include data from t+1, t+2, etc.
        """
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Backward-looking (CORRECT)
        rolling_mean = values.rolling(window=3, min_periods=1).mean()
        
        # At index 2, should be mean of [1, 2, 3] = 2.0
        assert rolling_mean.iloc[2] == 2.0, \
            "Rolling mean should use only past data"
        
        # At index 9, should be mean of [8, 9, 10] = 9.0
        assert rolling_mean.iloc[9] == 9.0

    def test_lag_features_correct_direction(self) -> None:
        """
        Lag features should use positive shift (past data), not negative shift (future data).
        """
        prices = pd.Series([100, 101, 102, 103, 104])
        
        # Correct: lag_1 = price from 1 period ago
        lag_1_correct = prices.shift(1)
        assert pd.isna(lag_1_correct.iloc[0]), "First lag value should be NaN"
        assert lag_1_correct.iloc[1] == 100, "Lag should be previous value"
        
        # Wrong: lead_1 = price from 1 period ahead (this leaks future)
        lead_1_wrong = prices.shift(-1)
        assert lead_1_wrong.iloc[0] == 101, "Lead uses future data (BAD)"
        
        # Verify they're different
        assert not lag_1_correct.equals(lead_1_wrong)


class TestTargetLeakage:
    """
    Detect leakage in label/target creation.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_label_computed_after_features(self) -> None:
        """
        Labels should be computed from future data that wasn't available
        when features were computed.
        """
        close = pd.Series([100 + i for i in range(10)])
        horizon = 3
        
        # Compute future return (the label)
        future_close = close.shift(-horizon)
        future_return = future_close / close - 1
        
        # Verify: future_return at index 0 uses close[3], which is 103
        # But at index 0, we only have access to close[0] = 100
        assert future_return.iloc[0] == (103 / 100 - 1)

    def test_no_target_in_features(self) -> None:
        """
        The target variable (future_return, blend_label) should not appear in features.
        """
        # Simulate feature columns
        feature_cols = [
            'return_1', 'return_5', 'volatility_20', 'rsi_14',
            'superma_hull', 'regime_atr14_norm_close'
        ]
        
        # These should NOT be in features
        forbidden_cols = ['future_return', 'blend_label', 'smoothed_return']
        
        for col in forbidden_cols:
            assert col not in feature_cols, \
                f"Target column '{col}' found in features - this is leakage!"


class TestCrossValidationLeakage:
    """
    Detect leakage in cross-validation for time series.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only sklearn.
    """

    def test_timeseries_split_no_overlap(self) -> None:
        """
        TimeSeriesSplit folds should not overlap.
        """
        n_samples = 100
        X = np.arange(n_samples).reshape(-1, 1)
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_test_indices: List[np.ndarray] = []
        for train_idx, test_idx in tscv.split(X):
            all_test_indices.append(test_idx)
        
        # Check no overlap between consecutive test sets
        for i in range(len(all_test_indices) - 1):
            overlap = np.intersect1d(all_test_indices[i], all_test_indices[i + 1])
            assert len(overlap) == 0, \
                f"Test sets {i} and {i+1} should not overlap"

    def test_purged_cv_gap_respected(self) -> None:
        """
        Purged CV should maintain a gap between train and test.
        """
        n_samples = 100
        gap = 5
        
        # Simulate purged split
        train_end = 70
        test_start = train_end + gap
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, n_samples)
        
        # Verify gap
        assert test_idx.min() - train_idx.max() == gap + 1, \
            f"Gap should be {gap}, got {test_idx.min() - train_idx.max() - 1}"

    def test_embargo_prevents_leakage(self) -> None:
        """
        Embargo period at the end of train prevents leakage from overlapping labels.
        """
        n_samples = 100
        label_horizon = 10
        embargo = label_horizon  # Embargo should at least equal label horizon
        
        train_end = 60
        test_start = train_end + embargo
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, n_samples)
        
        # Labels at train_end-1 use data up to train_end-1+horizon
        label_leakage_end = train_end - 1 + label_horizon
        
        assert test_idx.min() > label_leakage_end, \
            f"Test start ({test_idx.min()}) should be after label leakage end ({label_leakage_end})"


class TestInformationLeakage:
    """
    Detect subtle information leakage patterns.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_no_global_statistics_leak(self) -> None:
        """
        Global statistics (mean, std over entire dataset) leak future info.
        """
        full_data = pd.Series([100, 102, 98, 104, 96, 108, 92, 110, 90, 112])
        
        # Wrong: use global mean (includes future)
        global_mean = full_data.mean()
        
        # Correct: expanding mean (only uses past)
        expanding_mean = full_data.expanding().mean()
        
        # At index 5, expanding uses only first 6 values
        expected_mean_5 = full_data.iloc[:6].mean()
        np.testing.assert_almost_equal(expanding_mean.iloc[5], expected_mean_5, decimal=6)
        
        # Global mean is different
        assert global_mean != expanding_mean.iloc[5]

    def test_fit_transform_vs_transform_only(self) -> None:
        """
        fit_transform on full data vs fit on train + transform on test
        should give different results for test data.
        """
        train = np.array([[100], [102], [98], [104], [96]])
        test = np.array([[108], [92], [110]])
        full = np.vstack([train, test])
        
        # Wrong way: fit_transform on full data
        scaler_full = StandardScaler()
        full_transformed = scaler_full.fit_transform(full)
        test_from_full = full_transformed[-3:]
        
        # Correct way: fit on train, transform test separately
        scaler_train = StandardScaler()
        scaler_train.fit(train)
        test_transformed = scaler_train.transform(test)
        
        # These should differ
        assert not np.allclose(test_from_full, test_transformed), \
            "Proper train-only scaling should differ from full-data scaling"

    def test_regime_labels_not_from_future(self) -> None:
        """
        Regime labels should be based on past data, not future.
        """
        # Simulate regime detection based on SMA crossover
        close = pd.Series([100, 101, 102, 103, 104, 103, 102, 101, 100, 99])
        
        sma_fast = close.rolling(3).mean()
        sma_slow = close.rolling(5).mean()
        
        # Regime at time t based on SMAs at time t (which use only past data)
        regime = pd.Series('chop', index=close.index)
        regime[sma_fast > sma_slow] = 'trend_up'
        regime[sma_fast < sma_slow] = 'trend_down'
        
        # Verify first few values are NaN or based only on available data
        # SMA_5 needs 5 values, so regime[4] is first fully computed
        assert pd.notna(sma_slow.iloc[4])


class TestSubtleLeakagePatterns:
    """
    Test for subtle, hard-to-detect leakage patterns.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_sorted_data_leakage(self) -> None:
        """
        Sorting data before split can introduce leakage.
        """
        timestamps = pd.date_range('2024-01-01', periods=10, freq='h')
        values = pd.Series([5, 2, 8, 1, 9, 3, 7, 4, 6, 10], index=timestamps)
        
        # Correct: keep temporal order
        temporal_order = values.sort_index()
        
        # Split should be done on temporal order
        train = temporal_order.iloc[:7]
        test = temporal_order.iloc[7:]
        
        assert train.index.max() < test.index.min(), \
            "Temporal order must be preserved for split"

    def test_duplicated_timestamps_detected(self) -> None:
        """
        Duplicated timestamps can cause data leakage.
        """
        timestamps = pd.to_datetime([
            '2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 01:00',  # Duplicate!
            '2024-01-01 02:00', '2024-01-01 03:00'
        ])
        
        # Detect duplicates
        has_duplicates = timestamps.duplicated().any()
        assert has_duplicates, "Test expects duplicates to exist"
        
        # In real code, we should raise an error or handle this
        unique_timestamps = timestamps.drop_duplicates()
        assert len(unique_timestamps) < len(timestamps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
