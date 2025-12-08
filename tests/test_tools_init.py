"""
Test Tools Initialization (Smoke Tests)

=============================================================================
CURSOR AUTO-RUN: SAFE ✓
=============================================================================

These are LIGHTWEIGHT pytest wrappers that ONLY test:
- Import statements work
- Classes can be instantiated
- Default parameters are valid

These tests do NOT:
- Load real data files
- Run actual backtests
- Execute heavy computation
- Use GPU resources

For actual tool execution, use manual mode:
    python tools/audit_backtest.py
    python tools/bootstrap_equity.py
    python tools/walk_forward.py
    python tools/timeseries_cv.py
    python tools/profile_data.py
"""

from __future__ import annotations
import pytest
import pandas as pd
import numpy as np

# =============================================================================
# NOTE: We import tools carefully to test initialization only.
# All heavy computation is mocked or skipped.
# =============================================================================


class TestAuditBacktestInit:
    """
    Smoke tests for audit_backtest module.
    
    SAFE FOR CURSOR AUTO-RUN: Only tests imports and basic initialization.
    """

    def test_import_audit_backtest(self) -> None:
        """Verify audit_backtest module can be imported."""
        from tools.audit_backtest import audit_backtest, BacktestAuditResult
        assert audit_backtest is not None
        assert BacktestAuditResult is not None

    def test_backtest_audit_result_dataclass(self) -> None:
        """Verify BacktestAuditResult can be instantiated."""
        from tools.audit_backtest import BacktestAuditResult
        
        result = BacktestAuditResult(
            passed=True,
            errors=[],
            warnings=[],
            stats={'n_rows': 100}
        )
        
        assert result.passed is True
        assert len(result.errors) == 0

    def test_audit_backtest_with_mock_data(self, mock_trades_df) -> None:
        """Verify audit_backtest works with minimal mock data."""
        from tools.audit_backtest import audit_backtest
        
        result = audit_backtest(mock_trades_df, strict=False)
        
        # Should return a result (may pass or fail, just shouldn't crash)
        assert result is not None
        assert hasattr(result, 'passed')
        assert hasattr(result, 'errors')


class TestBootstrapEquityInit:
    """
    Smoke tests for bootstrap_equity module.
    
    SAFE FOR CURSOR AUTO-RUN: Only tests imports and basic initialization.
    """

    def test_import_bootstrap_equity(self) -> None:
        """Verify bootstrap_equity module can be imported."""
        from tools.bootstrap_equity import bootstrap_equity_curve, BootstrapResult
        assert bootstrap_equity_curve is not None
        assert BootstrapResult is not None

    def test_bootstrap_result_dataclass(self) -> None:
        """Verify BootstrapResult can be instantiated."""
        from tools.bootstrap_equity import BootstrapResult
        
        result = BootstrapResult(
            n_bootstraps=100,
            metric_name='sharpe',
            observed_value=1.5,
            bootstrap_mean=1.2,
            bootstrap_std=0.3,
            ci_lower=0.6,
            ci_upper=1.8,
            p_value=0.05,
            is_significant=True,
            bootstrap_distribution=np.array([1.0, 1.2, 1.4])
        )
        
        assert result.n_bootstraps == 100
        assert result.is_significant is True

    def test_bootstrap_with_tiny_data(self) -> None:
        """Verify bootstrap works with minimal data (fast)."""
        from tools.bootstrap_equity import bootstrap_equity_curve
        
        # Tiny dataset for fast test
        np.random.seed(42)
        returns = pd.Series(np.random.randn(30) * 0.01 + 0.001)
        
        # Very few bootstraps for speed
        results = bootstrap_equity_curve(
            returns, 
            n_bootstraps=10,  # Minimal for speed
            block_size=3
        )
        
        assert 'sharpe' in results
        assert results['sharpe'].n_bootstraps == 10


class TestWalkForwardInit:
    """
    Smoke tests for walk_forward module.
    
    SAFE FOR CURSOR AUTO-RUN: Only tests imports and basic initialization.
    """

    def test_import_walk_forward(self) -> None:
        """Verify walk_forward module can be imported."""
        from tools.walk_forward import walk_forward_evaluate, WalkForwardResult
        assert walk_forward_evaluate is not None
        assert WalkForwardResult is not None

    def test_walk_forward_result_dataclass(self) -> None:
        """Verify WalkForwardResult can be instantiated."""
        from tools.walk_forward import WalkForwardResult
        
        result = WalkForwardResult(
            n_folds=5,
            folds=[],
            avg_train_sharpe=1.5,
            avg_test_sharpe=1.0,
            sharpe_decay_ratio=0.33,
            total_oos_return=0.15,
            oos_sharpe=0.8,
            is_robust=True,
            failure_modes=[]
        )
        
        assert result.n_folds == 5
        assert result.is_robust is True


class TestTimeseriesCVInit:
    """
    Smoke tests for timeseries_cv module.
    
    SAFE FOR CURSOR AUTO-RUN: Only tests imports and basic initialization.
    """

    def test_import_timeseries_cv(self) -> None:
        """Verify timeseries_cv module can be imported."""
        from tools.timeseries_cv import (
            PurgedKFold, 
            TimeSeriesCVResult,
            get_embargo_periods
        )
        assert PurgedKFold is not None
        assert TimeSeriesCVResult is not None
        assert get_embargo_periods is not None

    def test_purged_kfold_init(self) -> None:
        """Verify PurgedKFold can be instantiated."""
        from tools.timeseries_cv import PurgedKFold
        
        cv = PurgedKFold(n_splits=5, gap_periods=10)
        
        assert cv.n_splits == 5
        assert cv.gap_periods == 10

    def test_purged_kfold_split_tiny_data(self) -> None:
        """Verify PurgedKFold splits work with tiny data."""
        from tools.timeseries_cv import PurgedKFold
        
        cv = PurgedKFold(n_splits=3, gap_periods=2)
        X = np.arange(30).reshape(-1, 1)
        
        splits = list(cv.split(X))
        assert len(splits) > 0  # Should produce at least some splits

    def test_get_embargo_periods(self) -> None:
        """Verify embargo period calculation."""
        from tools.timeseries_cv import get_embargo_periods
        
        embargo = get_embargo_periods(label_horizon=48)
        assert embargo == 57  # 48 * 1.2 = 57.6 -> 57


class TestProfileDataInit:
    """
    Smoke tests for profile_data module.
    
    SAFE FOR CURSOR AUTO-RUN: Only tests imports and basic initialization.
    """

    def test_import_profile_data(self) -> None:
        """Verify profile_data module can be imported."""
        from tools.profile_data import profile_ohlcv, DataProfileResult
        assert profile_ohlcv is not None
        assert DataProfileResult is not None

    def test_data_profile_result_dataclass(self) -> None:
        """Verify DataProfileResult can be instantiated."""
        from tools.profile_data import DataProfileResult
        
        result = DataProfileResult(
            n_rows=1000,
            date_range=(pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-31')),
            passed=True,
            errors=[],
            warnings=[],
            drift_metrics={'volatility_drift': 0.05},
            anomaly_counts={'price_gaps': 2},
            quality_score=95.0
        )
        
        assert result.n_rows == 1000
        assert result.passed is True

    def test_profile_with_tiny_ohlcv(self, tiny_ohlcv_df) -> None:
        """Verify profile_ohlcv works with tiny data."""
        from tools.profile_data import profile_ohlcv
        
        result = profile_ohlcv(tiny_ohlcv_df)
        
        assert result is not None
        assert hasattr(result, 'passed')
        assert hasattr(result, 'quality_score')


class TestToolsPackageImports:
    """
    Verify the tools package exports work correctly.
    
    SAFE FOR CURSOR AUTO-RUN: Only tests imports.
    """

    def test_import_from_tools_package(self) -> None:
        """Verify tools package exports are accessible."""
        from tools import (
            audit_backtest,
            BacktestAuditResult,
            bootstrap_equity_curve,
            BootstrapResult,
            walk_forward_evaluate,
            WalkForwardResult,
            purged_kfold_cv,
            TimeSeriesCVResult,
            profile_ohlcv,
            DataProfileResult,
        )
        
        # All should be importable
        assert audit_backtest is not None
        assert BacktestAuditResult is not None
        assert bootstrap_equity_curve is not None
        assert BootstrapResult is not None
        assert walk_forward_evaluate is not None
        assert WalkForwardResult is not None
        assert purged_kfold_cv is not None
        assert TimeSeriesCVResult is not None
        assert profile_ohlcv is not None
        assert DataProfileResult is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

