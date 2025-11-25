# core/signal_blender.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
import lightgbm as lgb


class SignalBlender:
    """
    ML-based signal blending using LightGBMClassifier.
    Target is future returns > threshold.
    """

    def __init__(self, return_threshold: float = 0.001, model_params: Optional[dict] = None):
        """
        Initialize signal blender.
        
        Args:
            return_threshold: Threshold for positive returns (default 0.1%)
            model_params: Optional LightGBM parameters
        """
        self.return_threshold = return_threshold
        if model_params is None:
            model_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        self.model_params = model_params
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the signal blending model.
        
        Args:
            X: Feature dataframe (includes features + module signals + regime)
            y: Target labels (1 for long, -1 for short, 0 for flat)
                or future returns (will be converted to binary: sign(return) > threshold)
        """
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
        # Encode regime column if it exists (convert to numeric)
        if 'regime' in X_clean.columns:
            regime_map = {
                'trend_up': 0, 'trend_down': 1, 'chop': 2,
                'high_vol': 3, 'low_vol': 4, 'neutral': 5
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(5)
        
        # Convert y to binary if it's returns
        y_clean = y.copy()
        if y_clean.dtype in [np.float64, np.float32]:
            # If y is returns, convert to binary: 1 if return > threshold, 0 otherwise
            y_clean = (y_clean > self.return_threshold).astype(int)
        else:
            # If y is already signals, convert to binary: 1 for long, 0 for short/flat
            y_clean = (y_clean > 0).astype(int)
        
        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]
        
        # Remove any remaining NaN
        mask = ~(X_clean.isna().any(axis=1) | y_clean.isna())
        X_clean = X_clean[mask]
        y_clean = y_clean[mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid training data after cleaning")
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        
        # Initialize and train model
        self.model = lgb.LGBMClassifier(**self.model_params)
        self.model.fit(X_clean, y_clean)
        
        print(f"Signal blender trained on {len(X_clean)} samples with {len(self.feature_names)} features")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict signal (long/short/flat) for given features.
        
        Args:
            X: Feature dataframe (includes features + module signals + regime)
            
        Returns:
            Series of signals (-1, 0, or +1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
        # Encode regime column if it exists
        if 'regime' in X_clean.columns:
            regime_map = {
                'trend_up': 0, 'trend_down': 1, 'chop': 2,
                'high_vol': 3, 'low_vol': 4, 'neutral': 5
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(5)
        
        # Ensure same feature order as training
        missing_features = set(self.feature_names) - set(X_clean.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feat in missing_features:
                X_clean[feat] = 0
        
        X_clean = X_clean[self.feature_names]
        
        # Predict probabilities
        proba = self.model.predict_proba(X_clean)
        
        # Get probability of positive class (long)
        proba_long = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        
        # Convert to signals: 1 for long, -1 for short (if we had short class), 0 for flat
        # For binary classification, we'll use: 1 if proba > 0.5, 0 otherwise
        # TODO: Extend to support short signals if needed
        signals = pd.Series(
            np.where(proba_long > 0.5, 1, 0),
            index=X_clean.index,
            dtype=int
        )
        
        return signals

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict signal probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Series with probability of long signal
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
        # Encode regime column if it exists
        if 'regime' in X_clean.columns:
            regime_map = {
                'trend_up': 0, 'trend_down': 1, 'chop': 2,
                'high_vol': 3, 'low_vol': 4, 'neutral': 5
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(5)
        
        # Ensure same feature order as training
        missing_features = set(self.feature_names) - set(X_clean.columns)
        if missing_features:
            for feat in missing_features:
                X_clean[feat] = 0
        
        X_clean = X_clean[self.feature_names]
        
        # Align index
        X_clean = X_clean.reindex(X.index)
        
        # Predict probabilities
        proba = self.model.predict_proba(X_clean)
        proba_long = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        
        return pd.Series(proba_long, index=X_clean.index, name='confidence')

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_params': self.model_params,
            'return_threshold': self.return_threshold,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Signal blender saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_params = model_data.get('model_params', {})
        self.return_threshold = model_data.get('return_threshold', self.return_threshold)
        self.feature_names = model_data.get('feature_names', [])
        
        print(f"Signal blender loaded from {path}")
