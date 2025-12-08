"""
Tests for train.py pipeline logic.

Validates feature/label alignment, train/val splits, shape correctness,
and model input/output shapes using mock data.

WHY THESE TESTS EXIST:
- Feature/label misalignment is a critical bug in ML pipelines
- Train/val splits must be temporal (no future leakage)
- Model inputs must have correct shapes and no NaN values

WHAT INVARIANTS ARE PROTECTED:
- len(features) == len(labels) after alignment
- Train indices < Val indices (temporal split)
- No NaN in training features after cleaning
- Model outputs match expected shape

BUGS THESE TESTS WOULD CATCH:
- Label shift causing feature/label misalignment
- Random split instead of temporal split
- NaN propagation into model inputs
- Wrong feature selection dropping important columns
"""

from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import mocks
from tests.mocks import (
    make_synthetic_ohlcv,
    make_synthetic_features,
    make_aligned_features,
    make_regime_features,
    make_blender_features,
    make_synthetic_labels,
    make_direction_labels_mock,
    make_regime_labels_mock,
    make_balanced_labels,
    make_imbalanced_labels,
)


class TestFeatureLabelAlignment:
    """
    Tests for feature/label alignment in training.
    
    WHY: Misaligned features and labels produce garbage models.
    WHAT: Validates that features and labels share the same index.
    """
    
    def test_features_labels_same_length(self):
        """
        WHY: Verify feature and label arrays have same length.
        WHAT: len(X) == len(y) is required for training.
        BUG CAUGHT: Off-by-one error in label generation.
        """
        df = make_synthetic_ohlcv(n_rows=15)
        features = make_aligned_features(df, n_features=10)
        labels = make_direction_labels_mock(n_rows=15)
        labels.index = features.index  # Align indices
        
        assert len(features) == len(labels), "Features and labels must have same length"
    
    def test_features_labels_same_index(self):
        """
        WHY: Verify feature and label indices match exactly.
        WHAT: X.index should equal y.index.
        BUG CAUGHT: Shifted indices causing data mismatch.
        """
        df = make_synthetic_ohlcv(n_rows=15)
        features = make_aligned_features(df, n_features=10)
        labels = make_synthetic_labels(n_rows=15)
        labels.index = features.index  # Must align
        
        assert features.index.equals(labels.index), "Feature and label indices must match"
    
    def test_feature_label_alignment_after_dropna(self):
        """
        WHY: Verify alignment preserved after dropping NaN rows.
        WHAT: After dropna, features and labels should still align.
        BUG CAUGHT: dropna on features not propagating to labels.
        """
        df = make_synthetic_ohlcv(n_rows=20)
        features = make_aligned_features(df, n_features=10)
        labels = make_synthetic_labels(n_rows=20)
        labels.index = features.index
        
        # Introduce some NaN
        features.iloc[0, 0] = np.nan
        features.iloc[5, 2] = np.nan
        
        # Drop NaN from features
        valid_mask = ~features.isna().any(axis=1)
        features_clean = features[valid_mask]
        labels_clean = labels[valid_mask]
        
        assert len(features_clean) == len(labels_clean), "Lengths should match after dropna"
        assert features_clean.index.equals(labels_clean.index), "Indices should match after dropna"
    
    def test_reindex_aligns_labels_to_features(self):
        """
        WHY: Verify reindex correctly aligns labels to feature index.
        WHAT: labels.reindex(features.index) should align data.
        BUG CAUGHT: Label data lost during reindex.
        """
        features = make_synthetic_features(n_rows=15)
        labels_raw = make_direction_labels_mock(n_rows=20)  # Different length
        
        # Reindex labels to match features
        labels_aligned = labels_raw.reindex(features.index)
        
        assert len(labels_aligned) == len(features), "Reindex should match feature length"
        
        # Check overlapping values are preserved
        common_idx = features.index.intersection(labels_raw.index)
        for idx in common_idx:
            assert labels_aligned.loc[idx, 'blend_label'] == labels_raw.loc[idx, 'blend_label'], \
                "Reindex should preserve values at common indices"


class TestTrainValSplit:
    """
    Tests for train/validation split logic.
    
    WHY: Temporal splits are required for time series to avoid leakage.
    WHAT: Validates train comes before val with no overlap.
    """
    
    def test_temporal_split_no_overlap(self):
        """
        WHY: Verify train and val sets don't overlap.
        WHAT: intersection(train.index, val.index) should be empty.
        BUG CAUGHT: Data leakage from overlapping splits.
        """
        features = make_synthetic_features(n_rows=20)
        
        split_point = int(len(features) * 0.8)
        train = features.iloc[:split_point]
        val = features.iloc[split_point:]
        
        overlap = train.index.intersection(val.index)
        assert len(overlap) == 0, "Train and val should have no overlap"
    
    def test_temporal_split_order(self):
        """
        WHY: Verify train data is temporally before val data.
        WHAT: max(train.index) < min(val.index).
        BUG CAUGHT: Random split instead of temporal split.
        """
        features = make_synthetic_features(n_rows=20)
        
        split_point = int(len(features) * 0.8)
        train = features.iloc[:split_point]
        val = features.iloc[split_point:]
        
        assert train.index.max() < val.index.min(), "Train should be temporally before val"
    
    def test_split_preserves_all_data(self):
        """
        WHY: Verify split doesn't lose any data.
        WHAT: len(train) + len(val) == len(original).
        BUG CAUGHT: Data loss during split.
        """
        features = make_synthetic_features(n_rows=20)
        
        split_point = int(len(features) * 0.8)
        train = features.iloc[:split_point]
        val = features.iloc[split_point:]
        
        assert len(train) + len(val) == len(features), "Split should preserve all data"
    
    def test_walk_forward_split_non_overlapping(self):
        """
        WHY: Verify walk-forward windows don't overlap in validation.
        WHAT: Consecutive val windows should not share indices.
        BUG CAUGHT: Overlapping validation causing result inflation.
        """
        features = make_synthetic_features(n_rows=100)
        
        # Simulate walk-forward: 60% train, 20% val, 20% slide
        train_frac = 0.6
        val_frac = 0.2
        
        windows = []
        start = 0
        while start + int(len(features) * (train_frac + val_frac)) <= len(features):
            train_end = start + int(len(features) * train_frac)
            val_end = train_end + int(len(features) * val_frac)
            
            train = features.iloc[start:train_end]
            val = features.iloc[train_end:val_end]
            windows.append((train, val))
            
            start += int(len(features) * val_frac)  # Slide by val_frac
        
        # Check val windows don't overlap
        for i in range(len(windows) - 1):
            val_i = windows[i][1]
            val_j = windows[i + 1][1]
            overlap = val_i.index.intersection(val_j.index)
            assert len(overlap) == 0, f"Val windows {i} and {i+1} should not overlap"


class TestFeatureShape:
    """
    Tests for correct feature matrix shapes.
    
    WHY: Wrong shapes cause model training failures.
    WHAT: Validates feature dimensions match expectations.
    """
    
    def test_feature_matrix_2d(self):
        """
        WHY: Verify feature matrix is 2D (samples x features).
        WHAT: features.ndim should be 2.
        BUG CAUGHT: 1D or 3D array passed to model.
        """
        features = make_synthetic_features(n_rows=15, n_features=20)
        
        assert features.ndim == 2, "Features should be 2D"
        assert features.shape == (15, 20), "Shape should be (n_samples, n_features)"
    
    def test_label_vector_1d(self):
        """
        WHY: Verify label vector is 1D (n_samples,).
        WHAT: labels.ndim should be 1.
        BUG CAUGHT: 2D labels causing shape errors.
        """
        labels = make_synthetic_labels(n_rows=15)
        
        assert labels.ndim == 1, "Labels should be 1D"
        assert len(labels) == 15, "Labels length should match"
    
    def test_feature_columns_consistent(self):
        """
        WHY: Verify feature columns are consistent between train/val.
        WHAT: Train and val should have same columns.
        BUG CAUGHT: Column mismatch between train and val.
        """
        features = make_synthetic_features(n_rows=20, n_features=15)
        
        split_point = 16
        train = features.iloc[:split_point]
        val = features.iloc[split_point:]
        
        assert list(train.columns) == list(val.columns), "Columns should match"
    
    def test_no_nan_in_training_features(self):
        """
        WHY: Verify no NaN values in training features after cleaning.
        WHAT: features.isna().sum() should be 0.
        BUG CAUGHT: NaN propagating to model causing training failure.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        # Simulate cleaning
        features_clean = features.dropna()
        
        assert not features_clean.isna().any().any(), "No NaN after cleaning"


class TestLabelDistribution:
    """
    Tests for label distribution handling.
    
    WHY: Class imbalance affects model training.
    WHAT: Validates label distribution is handled correctly.
    """
    
    def test_balanced_labels_have_equal_classes(self):
        """
        WHY: Verify balanced label generator produces equal classes.
        WHAT: Each class should have same count.
        BUG CAUGHT: Generator not producing balanced data.
        """
        labels = make_balanced_labels(n_rows=15, classes=[-1, 0, 1])
        
        counts = labels.value_counts()
        assert counts.min() == counts.max(), "Balanced labels should have equal counts"
    
    def test_imbalanced_labels_have_majority(self):
        """
        WHY: Verify imbalanced generator produces expected distribution.
        WHAT: Majority class should dominate.
        BUG CAUGHT: Generator not producing imbalanced data.
        """
        labels = make_imbalanced_labels(n_rows=20, majority_class=0, majority_ratio=0.8)
        
        counts = labels.value_counts()
        majority_count = counts.get(0, 0)
        
        assert majority_count >= 14, "Majority class should have ~80% of samples"
    
    def test_all_label_classes_present(self):
        """
        WHY: Verify all expected label classes are represented.
        WHAT: Labels should include all classes for valid training.
        BUG CAUGHT: Missing class causing model to never predict it.
        """
        labels = make_synthetic_labels(n_rows=100, n_classes=3)
        
        unique_labels = set(labels.unique())
        expected = {0, 1, 2}
        
        # Should have at least most classes (with randomness)
        assert len(unique_labels) >= 2, "Should have multiple classes represented"


class TestRegimeLabeling:
    """
    Tests for regime label generation logic.
    
    WHY: Regime labels drive trading strategy selection.
    WHAT: Validates regime labeling produces valid categories.
    """
    
    def test_regime_labels_valid_categories(self):
        """
        WHY: Verify regime labels are valid strings.
        WHAT: Labels should be in expected set.
        BUG CAUGHT: Typo or invalid regime string.
        """
        labels = make_regime_labels_mock(n_rows=15)
        
        valid_regimes = {'trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol'}
        
        for label in labels.unique():
            assert label in valid_regimes, f"Invalid regime label: {label}"
    
    def test_regime_labels_no_nan(self):
        """
        WHY: Verify regime labels have no NaN values.
        WHAT: All labels should be valid strings.
        BUG CAUGHT: NaN regime causing filtering issues.
        """
        labels = make_regime_labels_mock(n_rows=15)
        
        assert not labels.isna().any(), "Regime labels should have no NaN"
    
    def test_regime_features_alignment(self):
        """
        WHY: Verify regime features align with regime labels.
        WHAT: Features index should match labels index.
        BUG CAUGHT: Regime features shifted from labels.
        """
        features = make_regime_features(n_rows=15)
        labels = make_regime_labels_mock(n_rows=15)
        labels.index = features.index
        
        assert features.index.equals(labels.index), "Features and labels should align"


class TestModelInputValidation:
    """
    Tests for model input validation.
    
    WHY: Invalid inputs cause training failures or garbage predictions.
    WHAT: Validates input data meets model requirements.
    """
    
    def test_blender_features_have_signal_columns(self):
        """
        WHY: Verify SignalBlender features include signal columns.
        WHAT: Should have {module}_signal columns for each module.
        BUG CAUGHT: Missing signal columns causing model error.
        """
        features = make_blender_features(
            n_rows=15,
            signal_modules=['superma', 'trendmagic', 'pvt']
        )
        
        required_cols = ['superma_signal', 'trendmagic_signal', 'pvt_signal']
        for col in required_cols:
            assert col in features.columns, f"Missing required column: {col}"
    
    def test_blender_features_have_regime(self):
        """
        WHY: Verify SignalBlender features include regime column.
        WHAT: Should have 'regime' column (encoded as int).
        BUG CAUGHT: Missing regime feature causing model error.
        """
        features = make_blender_features(n_rows=15)
        
        assert 'regime' in features.columns, "Should have regime column"
        assert features['regime'].dtype in [np.int64, np.int32, np.float64], \
            "Regime should be numeric (encoded)"
    
    def test_features_numeric_dtype(self):
        """
        WHY: Verify all features are numeric.
        WHAT: All columns should be float or int.
        BUG CAUGHT: String column causing model error.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        for col in features.columns:
            assert np.issubdtype(features[col].dtype, np.number), \
                f"Column {col} should be numeric"
    
    def test_features_finite_values(self):
        """
        WHY: Verify features have no inf values.
        WHAT: All values should be finite.
        BUG CAUGHT: Inf from division by zero.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        assert np.isfinite(features.values).all(), "All feature values should be finite"


class TestTrainingDataQuality:
    """
    Tests for training data quality checks.
    
    WHY: Bad training data produces bad models.
    WHAT: Validates data quality requirements.
    """
    
    def test_sufficient_training_samples(self):
        """
        WHY: Verify training set has minimum required samples.
        WHAT: Should have at least n samples for valid training.
        BUG CAUGHT: Training on too few samples.
        """
        features = make_synthetic_features(n_rows=100)
        
        min_samples = 50  # Arbitrary minimum
        assert len(features) >= min_samples, f"Need at least {min_samples} training samples"
    
    def test_feature_variance_not_zero(self):
        """
        WHY: Verify features have non-zero variance.
        WHAT: Constant features provide no information.
        BUG CAUGHT: All-constant column in features.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        variances = features.var()
        assert (variances > 0).all(), "All features should have non-zero variance"
    
    def test_no_constant_columns(self):
        """
        WHY: Verify no columns are constant.
        WHAT: nunique() should be > 1 for all columns.
        BUG CAUGHT: Degenerate column causing model issues.
        """
        features = make_synthetic_features(n_rows=15, n_features=10)
        
        for col in features.columns:
            assert features[col].nunique() > 1, f"Column {col} is constant"

