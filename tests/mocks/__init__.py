"""
Mock data generators for Hybridyzer verification tests.

This package provides tiny, deterministic synthetic data generators for:
- OHLCV candle data (mock_candles)
- Trade logs (mock_trades)
- Model features (mock_features)
- Training labels (mock_labels)

All mocks generate small DataFrames (5-20 rows) for instant test execution.
These mocks ensure Cursor auto-tests run safely without heavy computation.
"""

from .mock_candles import (
    make_synthetic_ohlcv,
    make_trending_candles,
    make_choppy_candles,
    make_volatile_candles,
)
from .mock_trades import (
    make_synthetic_trades,
    make_winning_trades,
    make_losing_trades,
    make_mixed_trades,
)
from .mock_features import (
    make_synthetic_features,
    make_aligned_features,
    make_regime_features,
    make_blender_features,
)
from .mock_labels import (
    make_synthetic_labels,
    make_direction_labels_mock,
    make_regime_labels_mock,
    make_balanced_labels,
    make_imbalanced_labels,
)

__all__ = [
    # Candles
    "make_synthetic_ohlcv",
    "make_trending_candles",
    "make_choppy_candles",
    "make_volatile_candles",
    # Trades
    "make_synthetic_trades",
    "make_winning_trades",
    "make_losing_trades",
    "make_mixed_trades",
    # Features
    "make_synthetic_features",
    "make_aligned_features",
    "make_regime_features",
    "make_blender_features",
    # Labels
    "make_synthetic_labels",
    "make_direction_labels_mock",
    "make_regime_labels_mock",
    "make_balanced_labels",
    "make_imbalanced_labels",
]

