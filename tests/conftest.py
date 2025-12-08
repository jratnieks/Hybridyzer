"""
Pytest Configuration for Hybridyzer Verifier Layer

=============================================================================
CURSOR AUTO-RUN SAFETY CONFIGURATION
=============================================================================

This conftest.py ensures that:
1. Only lightweight tests run automatically in Cursor
2. Heavy computation is NEVER triggered by imports
3. GPU initialization is blocked in test mode
4. Real data files are NEVER loaded

Tests are categorized as:
- SAFE: Run automatically in Cursor (mock data only)
- MANUAL: Require explicit invocation (real data, heavy compute)

To run manual tests in WSL:
    pytest tests/ -m "manual" -v
    
To run only safe tests:
    pytest tests/ -m "not manual" -v
"""

from __future__ import annotations
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# =============================================================================
# BLOCK HEAVY IMPORTS AT TEST TIME
# =============================================================================

# Set environment variable to disable GPU detection in modules
os.environ["HYBRIDYZER_TEST_MODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPU from CUDA


def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest with custom markers for test categorization.
    """
    config.addinivalue_line(
        "markers", "manual: marks tests as manual-only (not for Cursor auto-run)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    """
    Automatically skip tests marked as manual, slow, gpu, or integration
    unless explicitly requested.
    
    This ensures Cursor auto-run only executes safe, fast tests.
    """
    # Check if user wants to run manual tests
    markexpr = config.getoption("-m", default="")
    
    # If user explicitly requests manual tests, don't skip them
    if "manual" in markexpr or "slow" in markexpr or "gpu" in markexpr:
        return
    
    skip_manual = pytest.mark.skip(reason="Manual test - run with: pytest -m manual")
    skip_slow = pytest.mark.skip(reason="Slow test - run with: pytest -m slow")
    skip_gpu = pytest.mark.skip(reason="GPU test - run with: pytest -m gpu")
    
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(skip_manual)
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# =============================================================================
# FIXTURES FOR SAFE MOCK DATA
# =============================================================================

@pytest.fixture
def tiny_ohlcv_df():
    """
    Tiny synthetic OHLCV DataFrame for fast tests.
    Only 10 rows - runs instantly.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n = 10
    
    return pd.DataFrame({
        'open': 100 + np.arange(n) * 0.1,
        'high': 101 + np.arange(n) * 0.1,
        'low': 99 + np.arange(n) * 0.1,
        'close': 100 + np.arange(n) * 0.1,
        'volume': [1000] * n
    }, index=pd.date_range('2024-01-01', periods=n, freq='5min'))


@pytest.fixture
def small_ohlcv_df():
    """
    Small synthetic OHLCV DataFrame for tests needing more data.
    Only 50 rows - still fast.
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n = 50
    
    return pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(n) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(n) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.uniform(1000, 2000, n)
    }, index=pd.date_range('2024-01-01', periods=n, freq='5min'))


@pytest.fixture
def mock_trades_df():
    """
    Mock trades DataFrame for schema validation.
    """
    import pandas as pd
    
    return pd.DataFrame({
        'signal': [1, -1, 0, 1, -1],
        'future_return': [0.01, -0.005, 0.0, 0.02, -0.01],
        'trade_pnl': [0.0094, 0.0044, 0.0, 0.0194, 0.0094],
        'trade_pnl_raw': [0.01, 0.005, 0.0, 0.02, 0.01],
        'regime': ['trend_up', 'trend_up', 'chop', 'trend_down', 'chop'],
        'direction_confidence': [0.65, 0.70, 0.50, 0.75, 0.60],
        'ev': [0.002, 0.003, 0.0, 0.005, 0.001],
        'final_signal': [1, -1, 0, 1, -1]
    }, index=pd.date_range('2024-01-01', periods=5, freq='5min'))


@pytest.fixture
def mock_equity_curve():
    """
    Mock equity curve Series for validation.
    """
    import pandas as pd
    
    return pd.Series(
        [1.0, 1.0094, 1.0138, 1.0138, 1.0335, 1.0432],
        index=pd.date_range('2024-01-01', periods=6, freq='5min'),
        name='equity'
    )


@pytest.fixture
def mock_calibration_df():
    """
    Mock calibration DataFrame.
    """
    import pandas as pd
    
    return pd.DataFrame({
        'prob_min': [0.5, 0.6, 0.7, 0.8],
        'prob_max': [0.6, 0.7, 0.8, 0.9],
        'hit_rate': [0.52, 0.55, 0.58, 0.62],
        'avg_gross_return': [0.001, 0.002, 0.003, 0.005],
        'count': [100, 150, 120, 80]
    })


# =============================================================================
# MOCK HEAVY MODULES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_gpu_modules(monkeypatch):
    """
    Mock GPU-related modules to prevent initialization.
    This runs automatically for ALL tests.
    """
    # Create mock modules
    mock_cudf = MagicMock()
    mock_cuml = MagicMock()
    mock_cupy = MagicMock()
    
    # Patch sys.modules to prevent GPU imports
    monkeypatch.setitem(sys.modules, 'cudf', mock_cudf)
    monkeypatch.setitem(sys.modules, 'cuml', mock_cuml)
    monkeypatch.setitem(sys.modules, 'cupy', mock_cupy)
    monkeypatch.setitem(sys.modules, 'cuml.ensemble', mock_cuml)

