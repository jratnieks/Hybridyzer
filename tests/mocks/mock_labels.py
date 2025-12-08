"""
Synthetic label data generators for testing.

These generators produce tiny, deterministic label Series/DataFrames
that simulate the labels produced by labeling.py without future data computation.

WHY THIS EXISTS:
- Tests need labels but must not compute actual future returns
- Label/feature alignment is critical and must be tested
- Different label distributions affect model training behavior

WHAT IT PROTECTS:
- Ensures label generation logic is testable without lookahead
- Validates that labels align correctly with features in train.py
- Tests class balance handling in training

HOW IT WORKS:
- Each generator returns small pandas Series/DataFrames (10-20 rows)
- Labels match the actual label format used in the codebase
- Data is deterministic and covers all label classes
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


def make_synthetic_labels(
    n_rows: int = 15,
    n_classes: int = 3,
    start_time: Optional[datetime] = None,
    class_labels: Optional[list] = None,
    seed: int = 42,
) -> pd.Series:
    """
    Generate synthetic classification labels.
    
    Args:
        n_rows: Number of samples (default: 15)
        n_classes: Number of unique classes (default: 3)
        start_time: Starting timestamp (default: 2024-01-01)
        class_labels: Explicit class values (default: 0, 1, ..., n_classes-1)
        seed: Random seed (default: 42)
        
    Returns:
        Series of integer labels with DatetimeIndex
        
    Example:
        >>> y = make_synthetic_labels(n_rows=10, n_classes=3)
        >>> assert len(y) == 10
        >>> assert y.nunique() <= 3
    """
    np.random.seed(seed)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    if class_labels is None:
        class_labels = list(range(n_classes))
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    labels = np.random.choice(class_labels, size=n_rows)
    
    return pd.Series(labels, index=timestamps, name='label')


def make_direction_labels_mock(
    n_rows: int = 15,
    start_time: Optional[datetime] = None,
    include_returns: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic direction labels matching make_direction_labels() output.
    
    WHY: Tests train.py label handling without computing actual future returns.
    WHAT: Produces DataFrame with blend_label and future_return columns.
    
    Args:
        n_rows: Number of samples (default: 15)
        start_time: Starting timestamp
        include_returns: Include future_return column (default: True)
        seed: Random seed (default: 42)
        
    Returns:
        DataFrame matching labeling.py output format:
        - blend_label: Direction label (-1, 0, 1)
        - future_return: Simulated future return (optional)
        - smoothed_return: Smoothed return series (optional)
        
    Note:
        This mock does NOT look at future data - it generates
        synthetic labels that happen to follow a plausible pattern.
    """
    np.random.seed(seed)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    # Generate plausible direction labels (-1, 0, 1)
    # Weighted towards 0 (neutral) for realistic distribution
    blend_label = np.random.choice([-1, 0, 1], size=n_rows, p=[0.25, 0.50, 0.25])
    
    result = pd.DataFrame({'blend_label': blend_label}, index=timestamps)
    
    if include_returns:
        # Generate synthetic returns that roughly align with labels
        # (This is for testing only - not actual future returns)
        base_returns = np.random.normal(0, 0.002, n_rows)
        # Add bias based on label for some consistency
        bias = blend_label * 0.001
        future_return = base_returns + bias
        
        result['future_return'] = future_return
        result['smoothed_return'] = pd.Series(future_return).rolling(3, min_periods=1).mean().values
    
    return result


def make_regime_labels_mock(
    n_rows: int = 15,
    start_time: Optional[datetime] = None,
    regime_distribution: Optional[dict] = None,
    seed: int = 42,
) -> pd.Series:
    """
    Generate synthetic regime labels matching rule_based_regime() output.
    
    WHY: Tests train.py regime handling without expensive feature computation.
    WHAT: Produces Series with string regime labels.
    
    Args:
        n_rows: Number of samples (default: 15)
        start_time: Starting timestamp
        regime_distribution: Dict of regime->probability
                            (default: equal split across regimes)
        seed: Random seed (default: 42)
        
    Returns:
        Series of string regime labels matching train.py format:
        - 'trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol'
    """
    np.random.seed(seed)
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    all_regimes = ['trend_up', 'trend_down', 'chop', 'high_vol', 'low_vol']
    
    if regime_distribution is None:
        # Default: equal probability
        probs = [1/len(all_regimes)] * len(all_regimes)
    else:
        probs = [regime_distribution.get(r, 0) for r in all_regimes]
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
    
    regimes = np.random.choice(all_regimes, size=n_rows, p=probs)
    
    return pd.Series(regimes, index=timestamps, name='regime')


def make_balanced_labels(
    n_rows: int = 15,
    classes: list = None,
    start_time: Optional[datetime] = None,
) -> pd.Series:
    """
    Generate perfectly balanced labels for testing class balance handling.
    
    WHY: Tests that training handles balanced data correctly.
    WHAT: Produces labels with exactly equal class counts.
    
    Args:
        n_rows: Number of samples (will be adjusted to be divisible by n_classes)
        classes: List of class labels (default: [-1, 0, 1])
        start_time: Starting timestamp
        
    Returns:
        Series with exactly equal counts per class
    """
    if classes is None:
        classes = [-1, 0, 1]
    
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    n_classes = len(classes)
    # Adjust n_rows to be divisible by n_classes
    n_rows = (n_rows // n_classes) * n_classes
    per_class = n_rows // n_classes
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    # Create balanced labels
    labels = []
    for c in classes:
        labels.extend([c] * per_class)
    
    # Shuffle for realistic pattern
    np.random.seed(42)
    np.random.shuffle(labels)
    
    return pd.Series(labels, index=timestamps, name='label')


def make_imbalanced_labels(
    n_rows: int = 20,
    majority_class: int = 0,
    majority_ratio: float = 0.8,
    start_time: Optional[datetime] = None,
) -> pd.Series:
    """
    Generate imbalanced labels for testing minority class handling.
    
    WHY: Tests that training handles class imbalance correctly.
    WHAT: Produces labels where one class dominates.
    
    Args:
        n_rows: Number of samples (default: 20)
        majority_class: The dominant class label (default: 0)
        majority_ratio: Fraction of samples in majority class (default: 0.8)
        start_time: Starting timestamp
        
    Returns:
        Series with imbalanced class distribution
    """
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0)
    
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq="5min")
    
    n_majority = int(n_rows * majority_ratio)
    n_minority = n_rows - n_majority
    
    # Split minority between -1 and 1 (if majority is 0)
    minority_classes = [-1, 1] if majority_class == 0 else [c for c in [-1, 0, 1] if c != majority_class]
    
    labels = [majority_class] * n_majority
    for i, c in enumerate(minority_classes):
        count = n_minority // len(minority_classes)
        if i == len(minority_classes) - 1:
            count = n_minority - sum(1 for l in labels if l != majority_class)
        labels.extend([c] * count)
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(labels)
    
    return pd.Series(labels[:n_rows], index=timestamps, name='label')

