# core/regime_detector.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Literal, Optional

# GPU support: Use cuML if available, fallback to CPU
try:
    import cudf
    from cuml.ensemble import RandomForestClassifier as cuRFClassifier
    GPU_AVAILABLE = True
    print("[RegimeDetector] GPU acceleration available (cuML)")
except ImportError:
    GPU_AVAILABLE = False
    cudf = None
    cuRFClassifier = None
    print("[RegimeDetector] cuML not available, will use CPU fallback if needed")


RegimeType = Literal["trend_up", "trend_down", "chop"]


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
    GPU-accelerated regime detection using cuML RandomForestClassifier.
    Falls back to CPU (pandas) if cuML is unavailable.
    """

    def __init__(self, model_params: Optional[dict] = None, use_gpu: bool = True):
        """
        Initialize regime detector.
        
        Args:
            model_params: Optional cuML RandomForestClassifier parameters
            use_gpu: Whether to use GPU acceleration (default: True, falls back to CPU if unavailable)
        """
        if model_params is None:
            # cuML RandomForestClassifier parameters
            model_params = {
                'n_estimators': 100,
                'max_depth': 16,
                'n_bins': 128,
                'split_criterion': 0,  # 0=GINI, 1=ENTROPY
                'bootstrap': True,
                'max_samples': 0.8,
                'max_features': 0.9,
                'n_streams': 4,
                'random_state': 42
            }
        self.model_params = model_params
        self.use_gpu = use_gpu and GPU_AVAILABLE and (cuRFClassifier is not None)
        self.model: Optional[cuRFClassifier] = None
        self.regime_classes = ["trend_up", "trend_down", "chop"]
        self.feature_names = []
        
        if self.use_gpu:
            print("[RegimeDetector] GPU mode enabled (cuML RandomForestClassifier)")
        else:
            if use_gpu:
                print("[RegimeDetector] GPU requested but cuML unavailable, will use CPU fallback")
            else:
                print("[RegimeDetector] CPU mode")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the regime detection model on GPU using cuML.
        
        Args:
            X: Feature dataframe (pandas, will be converted to cuDF for GPU training)
            y: Regime labels (should be one of: trend_up, trend_down, chop)
        """
        # CPU: Clean features in pandas
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
        
        # GPU: Convert to cuDF for training
        if self.use_gpu:
            print(f"[RegimeDetector] Converting {len(X_clean)} samples to cuDF for GPU training...")
            X_gpu = cudf.from_pandas(X_clean)
            y_gpu = cudf.Series(y_numeric.values, dtype='int32')
            
            # Initialize and train model on GPU
            self.model = cuRFClassifier(**self.model_params)
            self.model.fit(X_gpu, y_gpu)
            
            print(f"[RegimeDetector] GPU training complete: {len(X_clean)} samples, {len(self.feature_names)} features")
        else:
            # CPU fallback: Use scikit-learn RandomForestClassifier
            from sklearn.ensemble import RandomForestClassifier
            print("[RegimeDetector] Using CPU fallback (scikit-learn RandomForestClassifier)")
            self.model = RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 16),
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_clean, y_numeric)
            print(f"[RegimeDetector] CPU training complete: {len(X_clean)} samples, {len(self.feature_names)} features")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict regime for given features (GPU-accelerated if available).
        
        Args:
            X: Feature dataframe (pandas, will be converted to cuDF for GPU prediction)
            
        Returns:
            Series of regime labels (pandas)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # CPU: Clean features in pandas
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
        
        # GPU: Convert to cuDF and predict
        if self.use_gpu:
            X_gpu = cudf.from_pandas(X_clean)
            predictions_numeric_gpu = self.model.predict(X_gpu)
            # Convert back to pandas
            predictions_numeric = predictions_numeric_gpu.to_pandas().values
        else:
            # CPU: Predict directly
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
        Predict regime probabilities (GPU-accelerated if available).
        
        Args:
            X: Feature dataframe (pandas, will be converted to cuDF for GPU prediction)
            
        Returns:
            DataFrame with probabilities for each regime (pandas)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # CPU: Clean features in pandas
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
        
        # GPU: Convert to cuDF and predict probabilities
        if self.use_gpu:
            X_gpu = cudf.from_pandas(X_clean)
            proba_gpu = self.model.predict_proba(X_gpu)
            # Convert back to pandas
            proba = proba_gpu.to_pandas().values
        else:
            # CPU: Predict probabilities directly
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
        # Note: cuML models are pickle-able and will work on GPU when loaded if cuML is available
        model_data = {
            'model': self.model,
            'model_params': self.model_params,
            'regime_classes': self.regime_classes,
            'feature_names': self.feature_names,
            'use_gpu': self.use_gpu  # Store GPU mode preference
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
        # Restore GPU mode if model was trained on GPU (will auto-detect on next predict)
        saved_use_gpu = model_data.get('use_gpu', False)
        if saved_use_gpu and GPU_AVAILABLE and (cuRFClassifier is not None):
            self.use_gpu = True
            print(f"[RegimeDetector] Model loaded (was trained on GPU, will use GPU for inference)")
        else:
            self.use_gpu = False
            print(f"[RegimeDetector] Model loaded (CPU mode)")
        
        print(f"Regime detector loaded from {path}")
