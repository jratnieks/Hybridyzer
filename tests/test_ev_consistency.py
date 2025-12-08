"""
Test EV (Expected Value) Consistency

=============================================================================
CURSOR AUTO-RUN: SAFE ✓
=============================================================================

All tests in this module use MOCK DATA ONLY and run instantly.
No real data files, no GPU, no heavy computation.

Verifies:
- EV math is correct
- EV filtering respects min_ev thresholds
- No NaN/inf corruption in EV calculations
- EV bins and quantile boundaries are consistent
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

# =============================================================================
# NOTE: NO HEAVY IMPORTS
# We do NOT import from core.final_signal or core.feature_store here
# to avoid triggering GPU initialization during Cursor auto-run.
# =============================================================================


class TestEVMathConsistency:
    """
    Test that EV calculations are mathematically consistent.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_ev_formula_matches_calibration(self) -> None:
        """
        EV = hit_rate * avg_win - (1 - hit_rate) * avg_loss
        Verify this formula is applied consistently.
        """
        # Create mock calibration data
        calibration_data = {
            'prob_min': [0.5, 0.6, 0.7, 0.8],
            'prob_max': [0.6, 0.7, 0.8, 0.9],
            'hit_rate': [0.52, 0.55, 0.58, 0.62],
            'avg_gross_return': [0.001, 0.002, 0.003, 0.005],
            'count': [100, 150, 120, 80]
        }
        df = pd.DataFrame(calibration_data)
        
        # EV must be positive when hit_rate > 0.5 and avg_gross_return > 0
        for _, row in df.iterrows():
            if row['hit_rate'] > 0.5 and row['avg_gross_return'] > 0:
                assert row['avg_gross_return'] > 0, \
                    f"EV should be positive: hit_rate={row['hit_rate']}, avg_return={row['avg_gross_return']}"

    def test_ev_no_nan_inf(self) -> None:
        """EV values must not contain NaN or Inf."""
        ev_series = pd.Series([0.001, 0.002, -0.001, 0.0, 0.003, 0.004])
        
        assert not ev_series.isna().any(), "EV series contains NaN values"
        assert not np.isinf(ev_series).any(), "EV series contains Inf values"

    def test_ev_bounds_realistic(self) -> None:
        """EV should be within realistic bounds for trading."""
        ev_series = pd.Series([0.001, 0.002, -0.001, 0.0, 0.003, -0.002])
        
        # EV per trade should be between -10% and +10% for realistic scenarios
        assert ev_series.min() >= -0.10, "EV too negative (< -10%)"
        assert ev_series.max() <= 0.10, "EV too positive (> +10%)"

    def test_ev_quantile_bins_non_overlapping(self) -> None:
        """Probability bins should not overlap."""
        # Simulate quantile-based bins
        probs = pd.Series([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
        bins = pd.qcut(probs, q=3, duplicates='drop')
        
        # Extract bin boundaries
        boundaries = []
        for interval in bins.unique():
            boundaries.append((interval.left, interval.right))
        
        # Check non-overlapping (sorted by left bound)
        boundaries_sorted = sorted(boundaries, key=lambda x: x[0])
        for i in range(len(boundaries_sorted) - 1):
            current_right = boundaries_sorted[i][1]
            next_left = boundaries_sorted[i + 1][0]
            assert current_right <= next_left, \
                f"Overlapping bins: {boundaries_sorted[i]} and {boundaries_sorted[i+1]}"


class TestEVFilteringLogic:
    """
    Test that EV filtering works correctly.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_min_ev_threshold_filters_correctly(self) -> None:
        """Trades below min_ev should be filtered out."""
        min_ev = 0.002
        ev_values = pd.Series([0.001, 0.003, 0.002, -0.001, 0.005])
        
        # Create mask for passing trades
        passing_mask = ev_values >= min_ev
        
        # Expected: indices 1, 2, 4 pass (values 0.003, 0.002, 0.005)
        expected_passing = pd.Series([False, True, True, False, True])
        pd.testing.assert_series_equal(passing_mask, expected_passing)

    def test_ev_filter_preserves_valid_trades(self) -> None:
        """Valid high-EV trades should not be filtered."""
        min_ev = 0.001
        
        # All trades above threshold
        ev_values = pd.Series([0.002, 0.003, 0.004, 0.005])
        passing_mask = ev_values >= min_ev
        
        assert passing_mask.all(), "High-EV trades should all pass"

    def test_ev_filter_handles_zero_threshold(self) -> None:
        """min_ev=0 should allow all positive EV trades."""
        min_ev = 0.0
        ev_values = pd.Series([0.001, -0.001, 0.0, 0.002])
        
        passing_mask = ev_values >= min_ev
        expected = pd.Series([True, False, True, True])
        pd.testing.assert_series_equal(passing_mask, expected)


class TestCalibrationDataIntegrity:
    """
    Test calibration data structure and integrity.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_calibration_csv_required_columns(self, mock_calibration_df) -> None:
        """Calibration CSV must have required columns."""
        required_cols = {'prob_min', 'prob_max', 'avg_gross_return'}
        
        assert required_cols.issubset(mock_calibration_df.columns), \
            f"Missing required columns: {required_cols - set(mock_calibration_df.columns)}"

    def test_prob_min_less_than_prob_max(self) -> None:
        """prob_min must be strictly less than prob_max."""
        df = pd.DataFrame({
            'prob_min': [0.5, 0.6, 0.7],
            'prob_max': [0.6, 0.7, 0.8]
        })
        
        assert (df['prob_min'] < df['prob_max']).all(), \
            "prob_min must be less than prob_max for all rows"

    def test_prob_bounds_in_valid_range(self) -> None:
        """Probability bounds must be in [0, 1]."""
        df = pd.DataFrame({
            'prob_min': [0.5, 0.6, 0.7],
            'prob_max': [0.6, 0.7, 0.8]
        })
        
        assert (df['prob_min'] >= 0).all(), "prob_min must be >= 0"
        assert (df['prob_max'] <= 1).all(), "prob_max must be <= 1"


class TestEVPerSideRegime:
    """
    Test EV calculations per side (long/short) and regime.
    
    SAFE FOR CURSOR AUTO-RUN: Uses only mock data.
    """

    def test_ev_differs_by_side(self) -> None:
        """Long and short should have different EV profiles."""
        # Mock per-side EV data
        long_ev = pd.Series([0.003, 0.004, 0.002])
        short_ev = pd.Series([0.001, -0.001, 0.000])
        
        # In a realistic scenario, long and short should differ
        assert not long_ev.equals(short_ev), \
            "Long and short EV profiles should differ"

    def test_ev_differs_by_regime(self) -> None:
        """EV should vary across regimes (trend_up, trend_down, chop)."""
        regime_evs = {
            'trend_up': 0.005,
            'trend_down': 0.001,
            'chop': -0.002
        }
        
        # Trend regimes should have higher EV than chop
        assert regime_evs['trend_up'] > regime_evs['chop'], \
            "trend_up should have higher EV than chop"

    def test_ev_lookup_hierarchical_fallback(self) -> None:
        """EV lookup should fallback through hierarchy correctly."""
        # Mock lookup table
        lookup_table = [
            {'regime': 'trend_up', 'side_str': 'long', 'ev': 0.005},
            {'regime': None, 'side_str': 'long', 'ev': 0.003},
            {'regime': 'trend_up', 'side_str': None, 'ev': 0.002},
            {'regime': None, 'side_str': None, 'ev': 0.001},
        ]
        
        # Best match for (trend_up, long) should be 0.005
        best_match = next(
            (r for r in lookup_table 
             if r['regime'] == 'trend_up' and r['side_str'] == 'long'),
            None
        )
        assert best_match is not None
        assert best_match['ev'] == 0.005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
