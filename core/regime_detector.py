# core/regime_detector.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Literal, Optional
import lightgbm as lgb


RegimeType = Literal["trend_up", "trend_down", "chop", "high_vol", "low_vol"]


def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Compute Wilder's ATR."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


class RegimeDetector:
    """
    ML-based regime detection using LightGBMClassifier.
    """

    def __init__(self, model_params: Optional[dict] = None):
        """
        Initialize regime detector.
        
        Args:
            model_params: Optional LightGBM parameters
        """
        if model_params is None:
            model_params = {
                'objective': 'multiclass',
                'num_class': 5,
                'metric': 'multi_logloss',
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
        self.regime_classes = ["trend_up", "trend_down", "chop", "high_vol", "low_vol"]
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the regime detection model.
        
        Args:
            X: Feature dataframe
            y: Regime labels (should be one of: trend_up, trend_down, chop, high_vol, low_vol)
        """
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
        # Ensure y is categorical with correct classes
        y_clean = y.copy()
        valid_regimes = set(self.regime_classes)
        y_clean = y_clean[y_clean.isin(valid_regimes)]
        X_clean = X_clean.loc[y_clean.index]
        
        # Convert to numeric labels
        label_map = {regime: idx for idx, regime in enumerate(self.regime_classes)}
        y_numeric = y_clean.map(label_map)
        
        # Remove any remaining NaN
        mask = ~(X_clean.isna().any(axis=1) | y_numeric.isna())
        X_clean = X_clean[mask]
        y_numeric = y_numeric[mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid training data after cleaning")
        
        # Store feature names
        self.feature_names = X_clean.columns.tolist()
        
        # Initialize and train model
        self.model = lgb.LGBMClassifier(**self.model_params)
        self.model.fit(X_clean, y_numeric)
        
        print(f"Regime detector trained on {len(X_clean)} samples with {len(self.feature_names)} features")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict regime for given features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Series of regime labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
        # Ensure same feature order as training
        missing_features = set(self.feature_names) - set(X_clean.columns)
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feat in missing_features:
                X_clean[feat] = 0
        
        X_clean = X_clean[self.feature_names]
        
        # Align index
        X_clean = X_clean.reindex(X.index)
        
        # Predict
        predictions_numeric = self.model.predict(X_clean)
        
        # Convert back to regime labels
        label_map = {idx: regime for idx, regime in enumerate(self.regime_classes)}
        predictions = pd.Series(
            [label_map[pred] for pred in predictions_numeric],
            index=X_clean.index,
            dtype=object,
            name='regime'
        )
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with probabilities for each regime
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
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
        
        # Create dataframe with regime labels as columns
        proba_df = pd.DataFrame(
            proba,
            index=X_clean.index,
            columns=self.regime_classes
        )
        
        return proba_df

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
            'regime_classes': self.regime_classes,
            'feature_names': self.feature_names
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Regime detector saved to {path}")

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
        self.regime_classes = model_data.get('regime_classes', self.regime_classes)
        self.feature_names = model_data.get('feature_names', [])
        
        print(f"Regime detector loaded from {path}")
