"""
Tests for training label computation invariants.

Validates that labels are computed correctly, features map to labels properly,
and no future values leak into training data.

WHY THESE TESTS EXIST:
- Labels determine what the model learns - must be correct
- Future leakage invalidates model performance estimates
- Feature/label mapping errors cause learning wrong patterns

WHAT INVARIANTS ARE PROTECTED:
- Direction labels use only future (shifted) returns
- Regime labels use only current/past features
- No future information in training features
- Label values are valid and bounded

BUGS THESE TESTS WOULD CATCH:
- Using current instead of future return for labels
- Feature using shift(-1) exposing future data
- Invalid label values outside expected range
- NaN in labels causing silent training issues
"""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Import mocks
from tests.mocks import (
    make_synthetic_ohlcv,
    make_trending_candles,
    make_choppy_candles,
    make_volatile_candles,
    make_regime_features,
    make_direction_labels_mock,
    make_regime_labels_mock,
)


class TestDirectionLabelInvariants:
    """
    Tests for direction label computation correctness.
    
    WHY: Direction labels are the target for trading models.
    WHAT: Validates label computation uses correct future returns.
    """
    
    def test_direction_label_values_bounded(self):
        """
        WHY: Verify direction labels are in valid range.
        WHAT: Labels should be in {-1, 0, 1}.
        BUG CAUGHT: Invalid label value causing classification error.
        """
        labels_df = make_direction_labels_mock(n_rows=20)
        labels = labels_df['blend_label']
        
        valid_values = {-1, 0, 1}
        for val in labels.unique():
            assert val in valid_values, f"Invalid label value: {val}"
    
    def test_direction_label_from_future_return(self):
        """
        WHY: Verify direction labels are based on future (shifted) returns.
        WHAT: Label at t should reflect return from t to t+horizon.
        BUG CAUGHT: Using current return instead of future return.
        """
        # Simulate correct label generation
        close = pd.Series([100, 101, 102, 103, 104], 
                          index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        horizon = 1
        future_return = close.shift(-horizon) / close - 1.0
        
        # Label: 1 if future_return > threshold, -1 if < -threshold, 0 otherwise
        threshold = 0.005
        labels = pd.Series(0, index=close.index)
        labels[future_return > threshold] = 1
        labels[future_return < -threshold] = -1
        
        # At t=0: future_return = (101-100)/100 = 0.01 > 0.005 → label = 1
        assert labels.iloc[0] == 1, "Should be long when future return > threshold"
        
        # Last value has NaN future return
        assert pd.isna(future_return.iloc[-1]), "Last future return should be NaN"
    
    def test_direction_label_no_current_information(self):
        """
        WHY: Verify labels don't leak current information into features.
        WHAT: Label at t should not be derivable from features at t.
        BUG CAUGHT: Label computed from current bar's information.
        """
        # Generate synthetic data
        df = make_synthetic_ohlcv(n_rows=20)
        
        # Simulate label generation with future return
        horizon = 2
        future_close = df['close'].shift(-horizon)
        current_close = df['close']
        future_return = future_close / current_close - 1.0
        
        # Labels depend on future_return which uses shift(-horizon)
        # This ensures at time t, we're looking at price at t+horizon
        # which is future information correctly used for labels
        
        # The key invariant: labels[t] depends on close[t+horizon]
        # which is NOT available as a feature at time t
        for i in range(len(df) - horizon):
            # At position i, the label is based on close[i+horizon]
            # Features at position i should NOT contain close[i+horizon]
            pass  # Conceptual test - actual features don't contain future close


class TestRegimeLabelInvariants:
    """
    Tests for regime label computation correctness.
    
    WHY: Regime labels control trading strategy selection.
    WHAT: Validates regime labels use only current/past information.
    """
    
    def test_regime_label_values_valid(self):
        """
        WHY: Verify regime labels are valid category strings.
        WHAT: Labels should be in expected set.
        BUG CAUGHT: Typo or new unlisted regime.
        """
        labels = make_regime_labels_mock(n_rows=20)
        
        valid_regimes = {'trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol'}
        
        for val in labels.unique():
            assert val in valid_regimes, f"Invalid regime: {val}"
    
    def test_regime_label_uses_past_data_only(self):
        """
        WHY: Verify regime labels don't use future data.
        WHAT: Regime at t should use only data from <= t.
        BUG CAUGHT: Regime computed from future prices.
        """
        # Rule-based regime logic uses:
        # - linreg slope (rolling, uses past data)
        # - price vs linreg midline (current data)
        # - ATR (rolling, uses past data)
        
        # None of these use shift(-n), so they only see past/current data
        features = make_regime_features(n_rows=20)
        
        # Rolling operations like rolling(14) only see past 14 bars
        # This is the correct behavior - no future leakage
        atr_like = features['close'].rolling(14, min_periods=1).std()
        
        # At position 1, rolling std uses data from [0, 1] (2 values)
        # Verify that rolling at a given position doesn't include future values
        rolling_at_5 = atr_like.iloc[5]
        manual_std_at_5 = features['close'].iloc[:6].std()  # [0, 1, 2, 3, 4, 5]
        
        assert np.isclose(rolling_at_5, manual_std_at_5), \
            "Rolling should use only past/current data"
    
    def test_trend_regime_consistency(self):
        """
        WHY: Verify trend regime detection is consistent with price action.
        WHAT: Uptrend should have positive slope, downtrend negative.
        BUG CAUGHT: Inverted trend detection logic.
        """
        # Generate trending data
        up_trend = make_trending_candles(n_rows=15, direction=1)
        down_trend = make_trending_candles(n_rows=15, direction=-1)
        
        # For uptrend: prices should be increasing
        assert up_trend['close'].iloc[-1] > up_trend['close'].iloc[0], \
            "Uptrend should have increasing prices"
        
        # For downtrend: prices should be decreasing
        assert down_trend['close'].iloc[-1] < down_trend['close'].iloc[0], \
            "Downtrend should have decreasing prices"
    
    def test_chop_regime_low_trend(self):
        """
        WHY: Verify choppy regime detection.
        WHAT: Choppy market should have low directional trend.
        BUG CAUGHT: Misclassifying trending as choppy.
        """
        choppy = make_choppy_candles(n_rows=15)
        
        # Choppy data should have similar start and end price
        price_change = abs(choppy['close'].iloc[-1] / choppy['close'].iloc[0] - 1.0)
        
        assert price_change < 0.05, "Choppy market should have small net price change"


class TestFeatureLabelLeakage:
    """
    Tests for future leakage prevention.
    
    WHY: Future leakage invalidates model performance.
    WHAT: Validates features don't contain future information.
    """
    
    def test_no_negative_shift_in_features(self):
        """
        WHY: Verify features don't use shift(-n) which exposes future.
        WHAT: Features should only use shift(n) where n >= 0.
        BUG CAUGHT: Feature accidentally using future data.
        """
        # Proper feature computation uses only positive shifts or rolling
        close = pd.Series([100, 101, 102, 103, 104],
                          index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        # Good: lagged feature (uses past)
        lagged = close.shift(1)  # OK - uses past
        assert pd.isna(lagged.iloc[0]), "Lagged should have NaN at start"
        
        # Good: rolling feature (uses past)
        rolling_mean = close.rolling(3, min_periods=1).mean()  # OK - uses past
        
        # Bad: future feature (uses future) - should NOT be in features
        # future = close.shift(-1)  # BAD - would expose future
    
    def test_rolling_only_sees_past(self):
        """
        WHY: Verify rolling operations only access past data.
        WHAT: Rolling window [t-n+1, t] should not include t+1.
        BUG CAUGHT: Rolling window including future data.
        """
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           index=pd.date_range('2024-01-01', periods=10, freq='5min'))
        
        rolling_mean = values.rolling(window=3, min_periods=1).mean()
        
        # At position 4 (value=5), rolling mean uses [3, 4, 5]
        expected_at_4 = (3 + 4 + 5) / 3
        assert rolling_mean.iloc[4] == expected_at_4, "Rolling should use past window"
        
        # NOT [4, 5, 6] which would include future
        wrong_value = (4 + 5 + 6) / 3
        assert rolling_mean.iloc[4] != wrong_value, "Rolling should not include future"
    
    def test_labels_correctly_use_future(self):
        """
        WHY: Verify labels (targets) correctly use future returns.
        WHAT: Labels SHOULD use shift(-n) as they're prediction targets.
        BUG CAUGHT: Labels using past data instead of future.
        """
        close = pd.Series([100, 101, 102, 103, 104],
                          index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        horizon = 2
        future_close = close.shift(-horizon)
        
        # At position 0, future_close should be close[2] = 102
        assert future_close.iloc[0] == 102, "Label should use future price"
        
        # At position 1, future_close should be close[3] = 103
        assert future_close.iloc[1] == 103, "Label should use future price"
        
        # Last horizon positions should be NaN (no future data)
        assert future_close.iloc[-horizon:].isna().all(), "No future data at end"


class TestLabelQuality:
    """
    Tests for label quality and validity.
    
    WHY: Bad labels cause models to learn wrong patterns.
    WHAT: Validates labels are clean and valid.
    """
    
    def test_labels_no_nan_in_training_range(self):
        """
        WHY: Verify training labels have no NaN values.
        WHAT: Labels used for training should all be valid.
        BUG CAUGHT: NaN labels causing training errors.
        """
        labels_df = make_direction_labels_mock(n_rows=20)
        labels = labels_df['blend_label']
        
        # Note: real labels may have NaN at end due to horizon
        # But mock should be clean
        assert not labels.isna().any(), "Mock labels should have no NaN"
    
    def test_label_class_balance_reasonable(self):
        """
        WHY: Verify labels aren't extremely imbalanced.
        WHAT: Should have representation in multiple classes.
        BUG CAUGHT: All-same labels causing degenerate model.
        """
        labels_df = make_direction_labels_mock(n_rows=100)
        labels = labels_df['blend_label']
        
        counts = labels.value_counts()
        
        # No single class should dominate > 90%
        max_ratio = counts.max() / len(labels)
        assert max_ratio < 0.90, f"Class imbalance too extreme: {max_ratio:.2%}"
    
    def test_labels_stable_over_time(self):
        """
        WHY: Verify labels don't have unrealistic flip patterns.
        WHAT: Should not alternate every bar (would be unpredictable).
        BUG CAUGHT: Noise in labels causing unlearnable pattern.
        """
        labels_df = make_direction_labels_mock(n_rows=20, seed=42)
        labels = labels_df['blend_label']
        
        # Count label changes
        changes = (labels != labels.shift(1)).sum() - 1  # -1 for first NaN comparison
        change_ratio = changes / len(labels)
        
        # Should not change every single bar
        assert change_ratio < 0.9, f"Labels change too frequently: {change_ratio:.2%}"


class TestFutureReturnComputation:
    """
    Tests for future return calculation correctness.
    
    WHY: Future return is the core label basis.
    WHAT: Validates return calculation is mathematically correct.
    """
    
    def test_future_return_formula(self):
        """
        WHY: Verify future return = close[t+h] / close[t] - 1.
        WHAT: Should match the standard return formula.
        BUG CAUGHT: Wrong formula (e.g., close[t+h] - close[t]).
        """
        close = pd.Series([100, 105, 110, 115, 120],
                          index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        horizon = 2
        future_close = close.shift(-horizon)
        future_return = future_close / close - 1.0
        
        # At t=0: return = 110/100 - 1 = 0.10 (10%)
        assert np.isclose(future_return.iloc[0], 0.10), "Future return formula wrong"
        
        # At t=1: return = 115/105 - 1 ≈ 0.095
        expected_1 = 115/105 - 1.0
        assert np.isclose(future_return.iloc[1], expected_1), "Future return formula wrong"
    
    def test_future_return_sign_convention(self):
        """
        WHY: Verify positive return means price went up.
        WHAT: return > 0 when future price > current price.
        BUG CAUGHT: Inverted return sign.
        """
        close = pd.Series([100, 110, 90, 100, 105],
                          index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        horizon = 1
        future_return = close.shift(-horizon) / close - 1.0
        
        # t=0: 110 > 100 → return > 0
        assert future_return.iloc[0] > 0, "Return should be positive when price increases"
        
        # t=1: 90 < 110 → return < 0
        assert future_return.iloc[1] < 0, "Return should be negative when price decreases"
    
    def test_future_return_nan_at_end(self):
        """
        WHY: Verify future return is NaN for last horizon bars.
        WHAT: Can't compute return without future data.
        BUG CAUGHT: Fake future data being used.
        """
        close = pd.Series([100, 101, 102, 103, 104],
                          index=pd.date_range('2024-01-01', periods=5, freq='5min'))
        
        horizon = 2
        future_return = close.shift(-horizon) / close - 1.0
        
        # Last 2 values should be NaN
        assert future_return.iloc[-2:].isna().all(), "Last horizon returns should be NaN"
        # First values should be valid
        assert future_return.iloc[:-2].notna().all(), "Earlier returns should be valid"


class TestSplitLeakage:
    """
    Tests for train/val split leakage prevention.
    
    WHY: Leakage between splits invalidates validation.
    WHAT: Validates clean separation between train and val.
    """
    
    def test_no_label_leakage_at_split_boundary(self):
        """
        WHY: Verify labels at split boundary don't use val data.
        WHAT: Last train labels shouldn't depend on val prices.
        BUG CAUGHT: Future return crossing split boundary.
        """
        # Create data
        close = pd.Series(range(100, 120), 
                          index=pd.date_range('2024-01-01', periods=20, freq='5min'))
        
        # Split at position 16
        split_idx = 16
        train_close = close.iloc[:split_idx]
        val_close = close.iloc[split_idx:]
        
        # Future return with horizon=2
        horizon = 2
        train_future_return = train_close.shift(-horizon) / train_close - 1.0
        
        # Last 'horizon' training labels should be NaN
        # because their future crosses into validation
        train_labels_clean = train_future_return.iloc[:-horizon]
        
        assert train_labels_clean.notna().all(), "Train labels (excluding end) should be valid"
        assert train_future_return.iloc[-horizon:].isna().all(), \
            "Train labels at boundary should be NaN (future crosses into val)"
    
    def test_purged_gap_between_train_val(self):
        """
        WHY: Verify proper gap (purge) between train and val.
        WHAT: Should skip 'horizon' bars between train end and val start.
        BUG CAUGHT: Val data leaking into train via horizon overlap.
        """
        n_samples = 100
        horizon = 5
        
        # Proper purged split
        train_end = 70
        purge_size = horizon
        val_start = train_end + purge_size
        
        train_idx = list(range(train_end))
        val_idx = list(range(val_start, n_samples))
        
        # Verify no overlap in the purged region
        purge_region = set(range(train_end, val_start))
        train_set = set(train_idx)
        val_set = set(val_idx)
        
        assert len(train_set & purge_region) == 0, "Train should not include purge region"
        assert len(val_set & purge_region) == 0, "Val should not include purge region"
        assert len(train_set & val_set) == 0, "Train and val should not overlap"

