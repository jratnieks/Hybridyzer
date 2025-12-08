"""
Hybridyzer Verifier Layer - Tools

This package provides verification and analysis tools for the Bitcoin trading bot:
- audit_backtest: Invariant checks on trades and equity
- bootstrap_equity: Overfitting detection via bootstrap resampling
- walk_forward: Walk-forward evaluation
- timeseries_cv: Purged time-series cross-validation
- profile_data: OHLCV data drift and anomaly detection

Usage:
    from tools import (
        audit_backtest, BacktestAuditResult,
        bootstrap_equity_curve, BootstrapResult,
        walk_forward_evaluate, WalkForwardResult,
        purged_kfold_cv, TimeSeriesCVResult,
        profile_ohlcv, DataProfileResult,
    )
    
    # Audit backtest results
    result = audit_backtest(trades_df, equity_curve)
    if not result.passed:
        print(result)
    
    # Profile OHLCV data
    profile = profile_ohlcv(ohlcv_df)
    if not profile.passed:
        print(profile)
"""

from .audit_backtest import audit_backtest, audit_from_csv, BacktestAuditResult
from .bootstrap_equity import (
    bootstrap_equity_curve,
    detect_overfitting,
    BootstrapResult,
)
from .walk_forward import walk_forward_evaluate, WalkForwardResult, WalkForwardFold
from .timeseries_cv import (
    purged_kfold_cv,
    PurgedKFold,
    CombinatorialPurgedKFold,
    TimeSeriesCVResult,
    get_embargo_periods,
)
from .profile_data import (
    profile_ohlcv,
    profile_from_csv,
    compare_data_periods,
    DataProfileResult,
)


# Package version
__version__ = "0.1.0"


__all__ = [
    # Version
    "__version__",
    # Audit backtest
    "audit_backtest",
    "audit_from_csv",
    "BacktestAuditResult",
    # Bootstrap equity
    "bootstrap_equity_curve",
    "detect_overfitting",
    "BootstrapResult",
    # Walk-forward
    "walk_forward_evaluate",
    "WalkForwardResult",
    "WalkForwardFold",
    # Time series CV
    "purged_kfold_cv",
    "PurgedKFold",
    "CombinatorialPurgedKFold",
    "TimeSeriesCVResult",
    "get_embargo_periods",
    # Data profiling
    "profile_ohlcv",
    "profile_from_csv",
    "compare_data_periods",
    "DataProfileResult",
]
