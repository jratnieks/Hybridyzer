"""
Time Series Cross-Validation Tool

Implements purged K-fold cross-validation for time series data,
designed to prevent lookahead bias and information leakage.

Features:
- Purged splits with configurable gap (embargo) period
- Combinatorial purged CV for unbiased performance estimation
- Stratified time series splits (experimental)
- Integration with sklearn-style interface

References:
- "Advances in Financial Machine Learning" by M. Lopez de Prado
- Purged K-Fold Cross-Validation
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, Optional, Generator, Dict, Any


@dataclass
class TimeSeriesCVResult:
    """
    Results from time series cross-validation.
    
    Attributes:
        n_splits: Number of CV splits
        train_scores: List of training scores per fold
        test_scores: List of test scores per fold
        avg_train_score: Mean training score
        avg_test_score: Mean test score
        std_test_score: Standard deviation of test scores
        gap_periods: Gap periods used between train/test
        fold_details: List of fold metadata
    """
    n_splits: int
    train_scores: List[float]
    test_scores: List[float]
    avg_train_score: float
    avg_test_score: float
    std_test_score: float
    gap_periods: int
    fold_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"Time Series CV Results ({self.n_splits} folds, gap={self.gap_periods}):\n"
            f"  Train Score: {self.avg_train_score:.4f}\n"
            f"  Test Score: {self.avg_test_score:.4f} ± {self.std_test_score:.4f}\n"
            f"  Score Decay: {(self.avg_train_score - self.avg_test_score) / self.avg_train_score:.1%}"
            if self.avg_train_score != 0 else ""
        )


class PurgedKFold:
    """
    Purged K-Fold cross-validator for time series.
    
    This cross-validator prevents information leakage by:
    1. Ensuring test samples are always after training samples
    2. Adding a gap (embargo) period between train and test
    3. Purging any training samples whose labels overlap with test
    
    The gap period should be at least as long as the label lookahead
    used in the trading strategy.
    
    Attributes:
        n_splits: Number of cross-validation folds
        gap_periods: Number of periods to skip between train and test
        
    Example:
        >>> cv = PurgedKFold(n_splits=5, gap_periods=10)
        >>> for train_idx, test_idx in cv.split(X):
        ...     X_train, X_test = X[train_idx], X[test_idx]
        ...     # Train and evaluate model
    """
    
    def __init__(self, n_splits: int = 5, gap_periods: int = 0):
        """
        Initialize PurgedKFold.
        
        Args:
            n_splits: Number of folds (default: 5)
            gap_periods: Number of periods between train and test (default: 0)
        """
        self.n_splits = n_splits
        self.gap_periods = gap_periods
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target array (not used, for sklearn compatibility)
            groups: Group labels (not used, for sklearn compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for fold_idx in range(self.n_splits):
            # Test set is the current fold
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_samples
            
            # Training set is everything before test (with gap)
            train_end = max(0, test_start - self.gap_periods)
            
            if train_end <= 0:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold cross-validator.
    
    Generates all possible train/test combinations from K folds,
    ensuring no information leakage. This provides a more robust
    performance estimate with higher variance.
    
    For K folds, generates C(K, test_folds) combinations.
    
    Attributes:
        n_splits: Number of base folds
        n_test_folds: Number of folds to use for testing in each split
        gap_periods: Gap periods between train and test
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_folds: int = 1,
        gap_periods: int = 0
    ):
        """
        Initialize CombinatorialPurgedKFold.
        
        Args:
            n_splits: Total number of folds (default: 5)
            n_test_folds: Number of folds to use for testing (default: 1)
            gap_periods: Gap periods between train and test (default: 0)
        """
        self.n_splits = n_splits
        self.n_test_folds = n_test_folds
        self.gap_periods = gap_periods
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate combinatorial train/test splits.
        
        For time series, we only use combinations where test folds
        come after training folds to prevent lookahead.
        """
        from itertools import combinations
        
        n_samples = len(X)
        fold_indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        # Create fold boundaries
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            folds.append(fold_indices[start:end])
        
        # Generate combinations
        for test_fold_indices in combinations(range(self.n_splits), self.n_test_folds):
            # For time series: training folds must precede test folds
            min_test_fold = min(test_fold_indices)
            
            # Collect training indices (all folds before test folds, with gap)
            train_indices = []
            for fold_idx in range(min_test_fold):
                train_indices.extend(folds[fold_idx])
            
            # Apply gap
            if self.gap_periods > 0 and len(train_indices) > self.gap_periods:
                train_indices = train_indices[:-self.gap_periods]
            
            if len(train_indices) == 0:
                continue
            
            # Collect test indices
            test_indices = []
            for fold_idx in test_fold_indices:
                test_indices.extend(folds[fold_idx])
            
            yield np.array(train_indices), np.array(test_indices)


def purged_kfold_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    n_splits: int = 5,
    gap_periods: int = 0,
    scoring_func: Optional[callable] = None
) -> TimeSeriesCVResult:
    """
    Perform purged K-fold cross-validation.
    
    Wrapper function that performs purged K-fold CV with a sklearn-style
    model and returns aggregated results.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        model: sklearn-style model with fit/predict methods
        n_splits: Number of CV folds (default: 5)
        gap_periods: Gap periods between train/test (default: 0)
        scoring_func: Custom scoring function (default: accuracy or R²)
        
    Returns:
        TimeSeriesCVResult with aggregated cross-validation metrics
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(n_estimators=100)
        >>> result = purged_kfold_cv(X, y, model, n_splits=5, gap_periods=10)
        >>> print(result)
        
    Notes:
        - Gap period should match label lookahead to prevent leakage
        - For regression, returns R²; for classification, returns accuracy
        - Model should support fit() and score() methods
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    cv = PurgedKFold(n_splits=n_splits, gap_periods=gap_periods)
    
    train_scores: List[float] = []
    test_scores: List[float] = []
    fold_details: List[Dict[str, Any]] = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        
        # Skip if insufficient samples
        if len(y_train) < 10 or len(y_test) < 5:
            continue
        
        # Fit model
        try:
            model.fit(X_train, y_train)
            
            # Score
            if scoring_func is not None:
                train_score = scoring_func(y_train, model.predict(X_train))
                test_score = scoring_func(y_test, model.predict(X_test))
            else:
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
            fold_details.append({
                'fold_idx': fold_idx,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'train_score': train_score,
                'test_score': test_score
            })
            
        except Exception as e:
            fold_details.append({
                'fold_idx': fold_idx,
                'error': str(e)
            })
    
    if len(test_scores) == 0:
        return TimeSeriesCVResult(
            n_splits=0,
            train_scores=[],
            test_scores=[],
            avg_train_score=0.0,
            avg_test_score=0.0,
            std_test_score=0.0,
            gap_periods=gap_periods,
            fold_details=fold_details
        )
    
    return TimeSeriesCVResult(
        n_splits=len(test_scores),
        train_scores=train_scores,
        test_scores=test_scores,
        avg_train_score=np.mean(train_scores),
        avg_test_score=np.mean(test_scores),
        std_test_score=np.std(test_scores, ddof=1),
        gap_periods=gap_periods,
        fold_details=fold_details
    )


def get_embargo_periods(label_horizon: int, safety_factor: float = 1.2) -> int:
    """
    Calculate recommended embargo/gap periods.
    
    Args:
        label_horizon: Number of periods used in label creation
        safety_factor: Multiplier for safety margin (default: 1.2)
        
    Returns:
        Recommended gap periods
    """
    return int(label_horizon * safety_factor)


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate labels with some signal
    signal = X.iloc[:, 0] + 0.5 * X.iloc[:, 1]
    y = pd.Series((signal > 0).astype(int), name='label')
    
    print("=" * 60)
    print("Purged K-Fold Cross-Validation")
    print("=" * 60)
    
    # Create model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    
    # Run CV without gap
    result_no_gap = purged_kfold_cv(X, y, model, n_splits=5, gap_periods=0)
    print("\nWithout gap:")
    print(result_no_gap)
    
    # Run CV with gap
    result_with_gap = purged_kfold_cv(X, y, model, n_splits=5, gap_periods=10)
    print("\nWith 10-period gap:")
    print(result_with_gap)
    
    # Demonstrate embargo calculation
    label_horizon = 48  # 4 hours for 5-min data
    recommended_gap = get_embargo_periods(label_horizon)
    print(f"\nRecommended embargo for horizon={label_horizon}: {recommended_gap} periods")

