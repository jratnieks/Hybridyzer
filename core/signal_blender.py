# core/signal_blender.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple

# Calibration support
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    IsotonicRegression = None
    CalibratedClassifierCV = None
    print("[SignalBlender] sklearn calibration not available")


class BlenderBase:
    """
    Base class for SignalBlender and DirectionBlender.
    Contains shared calibration, sharpening, and clipping logic.
    """
    
    def _fit_calibration(
        self,
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Fit probability calibration models for each class.
        
        Args:
            X_train: Training features
            y_train: Training labels (numeric: 0, 1, 2 for SignalBlender, 0, 1 for DirectionBlender)
            X_val: Optional validation features (if None, uses X_train)
            y_val: Optional validation labels (if None, uses y_train)
        """
        if not CALIBRATION_AVAILABLE:
            class_name = self.__class__.__name__
            print(f"[{class_name}] Calibration requested but sklearn not available")
            return
        
        # Use validation set if provided, otherwise use training set
        X_cal = X_val if X_val is not None else X_train
        y_cal = y_val if y_val is not None else y_train
        
        if X_cal is None or len(X_cal) == 0:
            class_name = self.__class__.__name__
            print(f"[{class_name}] No calibration data available")
            return
        
        # Get raw probabilities from base model
        if self.use_gpu:
            cudf = self._require_cuml()[0]
            X_gpu = cudf.from_pandas(X_cal)
            proba_raw = self.model.predict_proba(X_gpu).to_pandas().values
        else:
            proba_raw = self.model.predict_proba(X_cal)
        
        # Fit calibrator for each class
        self.calibrators = {}
        n_classes = proba_raw.shape[1]
        
        for class_idx in range(n_classes):
            # Get true binary labels for this class
            y_binary = (y_cal == class_idx).astype(int).values
            
            # Get predicted probabilities for this class
            proba_class = proba_raw[:, class_idx]
            
            # Safety: Skip if fewer than 100 samples for this class
            if y_binary.sum() < 100:
                class_name = self.__class__.__name__
                print(f"[{class_name}] Skipping calibration for class {class_idx} (only {y_binary.sum()} samples, need >= 100)")
                continue
            
            if self.calibration_method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(proba_class, y_binary)
            elif self.calibration_method == 'platt':
                from sklearn.linear_model import LogisticRegression
                # Platt scaling: logistic regression on log-odds
                # Use small regularization to avoid overfitting
                calibrator = LogisticRegression(C=1.0, max_iter=1000)
                # Reshape for sklearn
                calibrator.fit(proba_class.reshape(-1, 1), y_binary)
            else:
                continue
            
            self.calibrators[class_idx] = calibrator
        
        class_name = self.__class__.__name__
        print(f"[{class_name}] Fitted {len(self.calibrators)} {self.calibration_method} calibrators")
    
    def _apply_calibration(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            proba: Raw or partially processed probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities
        """
        if not self.calibrators:
            return proba
        
        n_classes = proba.shape[1]
        for class_idx in range(n_classes):
            if class_idx in self.calibrators:
                calibrator = self.calibrators[class_idx]
                if self.calibration_method == 'isotonic':
                    proba[:, class_idx] = calibrator.predict(proba[:, class_idx])
                elif self.calibration_method == 'platt':
                    from sklearn.linear_model import LogisticRegression
                    proba[:, class_idx] = calibrator.predict_proba(proba[:, class_idx].reshape(-1, 1))[:, 1]
        
        return proba
    
    def _sharpen_probabilities(self, proba: np.ndarray) -> np.ndarray:
        """
        Apply sharpening to probabilities.
        
        Args:
            proba: Probabilities (n_samples, n_classes)
            
        Returns:
            Sharpened probabilities
        """
        if self.sharpening_alpha != 1.0:
            proba = np.power(proba, self.sharpening_alpha)
            # Renormalize to ensure probabilities sum to 1
            proba = proba / proba.sum(axis=1, keepdims=True)
        return proba
    
    def _clip_probabilities(self, proba: np.ndarray) -> np.ndarray:
        """
        Clip probabilities to avoid numerical issues.
        
        Args:
            proba: Probabilities (n_samples, n_classes)
            
        Returns:
            Clipped and renormalized probabilities
        """
        # Safety: Clip probabilities to avoid numerical issues
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        # Renormalize again after clipping
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba
    
    def _apply_calibration_and_sharpening(self, proba_raw: np.ndarray) -> np.ndarray:
        """
        Apply calibration and sharpening to raw probabilities.
        
        Args:
            proba_raw: Raw probabilities from model (n_samples, n_classes)
            
        Returns:
            Calibrated and sharpened probabilities
        """
        proba = proba_raw.copy()
        
        # Apply calibration
        proba = self._apply_calibration(proba)
        
        # Apply sharpening
        proba = self._sharpen_probabilities(proba)
        
        # Clip probabilities
        proba = self._clip_probabilities(proba)

        return proba

    @staticmethod
    def _require_cuml() -> Tuple[object, object]:
        try:
            import cudf  # type: ignore
            from cuml.ensemble import RandomForestClassifier as cuRFClassifier  # type: ignore
        except ImportError as exc:
            raise RuntimeError("cuML/cuDF are required when use_gpu=True") from exc
        return cudf, cuRFClassifier


class SignalBlender(BlenderBase):
    """
    GPU-accelerated signal blending using cuML RandomForestClassifier.
    Falls back to CPU (pandas) if cuML is unavailable.
    Outputs final long/short/flat labels from features.
    """

    def __init__(self, model_params: Optional[dict] = None, use_gpu: bool = False):
        """
        Initialize signal blender.
        
        Args:
            model_params: Optional cuML RandomForestClassifier parameters
            use_gpu: Whether to use GPU acceleration (default: True, falls back to CPU if unavailable)
        """
        if model_params is None:
            # cuML RandomForestClassifier parameters - lightweight for reduced VRAM
            model_params = {
                'n_estimators': 200,
                'max_depth': 22,
                'max_features': 0.6,
                'max_samples': 0.6,
                'n_bins': 128,
                'split_criterion': 0,  # 0=GINI, 1=ENTROPY
                'bootstrap': True,
                'n_streams': 1,  # Critical for reproducibility and lower VRAM spikes
                'random_state': 42
            }
        self.model_params = model_params
        self.use_gpu = use_gpu
        self.model = None
        self.signal_classes = [-1, 0, 1]  # short, flat, long
        self.feature_names = []
        self.feature_importances_ = None  # Will be populated after training
        self.name = "SignalBlender"  # For saving feature importances
        self._cudf = None
        self._cuRFClassifier = None
        
        # Calibration settings - defaults to always calibrate
        self.calibration_method = 'isotonic'  # Default: 'isotonic', can be 'platt' or None to disable
        self.calibrators = {}  # Dict mapping class index to calibrator
        self.sharpening_alpha = 2.0  # Post-hoc sharpening: prob = prob ** alpha (default: 2.0)
        
        if self.use_gpu:
            self._cudf, self._cuRFClassifier = self._require_cuml()
            print("[GPU] cuML enabled for SignalBlender")
        else:
            print("[CPU] sklearn backend active for SignalBlender")

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        calibration_method: Optional[str] = None,
        sharpening_alpha: Optional[float] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        disable_calibration: bool = False
    ) -> None:
        """
        Train the signal blending model.
        Calibration is ALWAYS enabled by default unless explicitly disabled.
        
        Args:
            X: Feature dataframe (includes features + module signals + regime)
            y: Target labels (-1 for short, 0 for flat, 1 for long)
                or future returns (will be converted to signals: sign(return))
            calibration_method: Calibration method ('isotonic', 'platt', or None to use default 'isotonic')
            sharpening_alpha: Post-hoc sharpening parameter (prob = prob ** alpha, default: 2.0)
            X_val: Optional validation features for calibration (if None, auto-splits 20% from X)
            y_val: Optional validation labels for calibration (if None, auto-splits 20% from y)
            disable_calibration: If True, disables calibration entirely
        """
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
        # Encode regime column if it exists (convert to numeric)
        if 'regime' in X_clean.columns:
            regime_map = {
                'trend_up': 0, 'trend_down': 1, 'chop': 2
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(2)  # Default to chop (2)
        
        # Convert y to signal labels (-1, 0, 1)
        y_clean = y.copy()
        if y_clean.dtype in [np.float64, np.float32]:
            # If y is returns, convert to signals: sign(return)
            y_clean = np.sign(y_clean).astype(int)
        else:
            # If y is already signals, ensure they're in [-1, 0, 1]
            y_clean = y_clean.astype(int)
            y_clean = y_clean.clip(-1, 1)
        
        # Ensure y contains only valid signal classes
        valid_signals = set(self.signal_classes)
        y_clean = y_clean[y_clean.isin(valid_signals)]
        X_clean = X_clean.loc[y_clean.index]
        
        # Convert to numeric labels for cuML (0, 1, 2)
        label_map = {-1: 0, 0: 1, 1: 2}  # short=0, flat=1, long=2
        y_numeric = y_clean.map(label_map)
        
        # Align indices
        common_idx = X_clean.index.intersection(y_numeric.index)
        X_clean = X_clean.loc[common_idx]
        y_numeric = y_numeric.loc[common_idx]
        
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
            cudf = self._cudf or self._require_cuml()[0]
            cuRFClassifier = self._cuRFClassifier or self._require_cuml()[1]
            print(f"[SignalBlender] Converting {len(X_clean)} samples to cuDF for GPU training...")
            X_gpu = cudf.from_pandas(X_clean)
            y_gpu = cudf.Series(y_numeric.values, dtype='int32')

            # Initialize and train model on GPU
            self.model = cuRFClassifier(**self.model_params)
            self.model.fit(X_gpu, y_gpu)

            print(f"[SignalBlender] GPU training complete: {len(X_clean)} samples, {len(self.feature_names)} features")
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            print("[CPU] sklearn backend active for SignalBlender")
            self.model = GradientBoostingClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 16),
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_clean, y_numeric)
            print(f"[SignalBlender] CPU training complete: {len(X_clean)} samples, {len(self.feature_names)} features")
        
        # Store feature importances from trained model
        self._store_feature_importances()
        
        # Store calibration settings - use defaults if not specified
        if disable_calibration:
            self.calibration_method = None
            self.sharpening_alpha = 1.0
        else:
            # Default to isotonic if not specified
            self.calibration_method = calibration_method if calibration_method is not None else 'isotonic'
            self.sharpening_alpha = sharpening_alpha if sharpening_alpha is not None else 2.0
        
        # Prepare validation data for calibration
        # If X_val/y_val not provided, automatically split 20% from training data
        X_cal = None
        y_cal_numeric = None
        
        if self.calibration_method and CALIBRATION_AVAILABLE:
            if X_val is not None and y_val is not None:
                # Clean validation features
                X_cal = X_val.copy()
                X_cal = X_cal.replace([np.inf, -np.inf], np.nan)
                X_cal = X_cal.ffill().bfill().fillna(0)
                
                # Encode regime if needed
                if 'regime' in X_cal.columns:
                    regime_map = {'trend_up': 0, 'trend_down': 1, 'chop': 2}
                    X_cal['regime'] = X_cal['regime'].map(regime_map).fillna(2)
                
                # Convert validation labels
                y_cal = y_val.copy()
                if y_cal.dtype in [np.float64, np.float32]:
                    y_cal = np.sign(y_cal).astype(int)
                else:
                    y_cal = y_cal.astype(int).clip(-1, 1)
                
                valid_signals = set(self.signal_classes)
                y_cal = y_cal[y_cal.isin(valid_signals)]
                X_cal = X_cal.loc[y_cal.index]
                
                label_map = {-1: 0, 0: 1, 1: 2}
                y_cal_numeric = y_cal.map(label_map)
                
                # Align indices
                common_idx = X_cal.index.intersection(y_cal_numeric.index)
                X_cal = X_cal.loc[common_idx]
                y_cal_numeric = y_cal_numeric.loc[common_idx]
            else:
                # Auto-split 20% for validation
                from sklearn.model_selection import train_test_split
                print(f"[SignalBlender] Auto-splitting 20% of data for calibration validation...")
                X_train_cal, X_cal, y_train_cal, y_cal_numeric = train_test_split(
                    X_clean, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
                )
                # Retrain on the 80% split (more accurate calibration)
                print(f"[SignalBlender] Retraining on 80% split ({len(X_train_cal)} samples)...")
                if self.use_gpu:
                    cudf = self._cudf or self._require_cuml()[0]
                    cuRFClassifier = self._cuRFClassifier or self._require_cuml()[1]
                    X_train_gpu = cudf.from_pandas(X_train_cal)
                    y_train_gpu = cudf.Series(y_train_cal.values, dtype='int32')
                    self.model = cuRFClassifier(**self.model_params)
                    self.model.fit(X_train_gpu, y_train_gpu)
                else:
                    self.model.fit(X_train_cal, y_train_cal)
                print(f"[SignalBlender] Using {len(X_cal)} samples for calibration")
        
        # Apply calibration if enabled
        if self.calibration_method and CALIBRATION_AVAILABLE:
            self._fit_calibration(X_clean, y_numeric, X_cal, y_cal_numeric)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict signal (long/short/flat) for given features.
        
        Args:
            X: Feature dataframe (includes features + module signals + regime)
            
        Returns:
            Series of signals (-1 for short, 0 for flat, 1 for long)
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
                'trend_up': 0, 'trend_down': 1, 'chop': 2
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(2)  # Default to chop (2)
        
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
            cudf = self._cudf or self._require_cuml()[0]
            X_gpu = cudf.from_pandas(X_clean)
            predictions_numeric_gpu = self.model.predict(X_gpu)
            # Convert back to pandas
            predictions_numeric = predictions_numeric_gpu.to_pandas().values
        else:
            # CPU: Predict directly
            predictions_numeric = self.model.predict(X_clean)
        
        # Convert back to signal labels (-1, 0, 1)
        label_map = {0: -1, 1: 0, 2: 1}  # short=-1, flat=0, long=1
        signals = pd.Series(
            [label_map[pred] for pred in predictions_numeric],
            index=X_clean.index,
            dtype=int,
            name='signal'
        )
        
        return signals

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict signal probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with probabilities for each signal class (short, flat, long)
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
                'trend_up': 0, 'trend_down': 1, 'chop': 2
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(2)  # Default to chop (2)
        
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
            cudf = self._cudf or self._require_cuml()[0]
            X_gpu = cudf.from_pandas(X_clean)
            proba_gpu = self.model.predict_proba(X_gpu)
            # Convert back to pandas
            proba_raw = proba_gpu.to_pandas().values
        else:
            # CPU: Predict probabilities directly
            proba_raw = self.model.predict_proba(X_clean)
        
        # Apply calibration and sharpening
        proba = self._apply_calibration_and_sharpening(proba_raw)
        
        # Create dataframe with signal labels as columns
        proba_df = pd.DataFrame(
            proba,
            index=X_clean.index,
            columns=self.signal_classes
        )
        
        return proba_df

    def _store_feature_importances(self) -> None:
        """
        Store feature importances if available.
        Skip safely when using cuML (GPU), because cuML RandomForest
        does NOT expose feature_importances_.
        """
        # GPU MODE: cuML does NOT support feature importances. Skip gracefully.
        if self.use_gpu:
            print(f"[{self.__class__.__name__}] GPU mode: feature importances unavailable (cuML). Skipping.")
            self.feature_importances_ = np.zeros(len(self.feature_names))
            return
        
        # CPU MODE (sklearn): feature importances available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_
        else:
            print(f"[{self.__class__.__name__}] CPU model has no feature_importances_. Using zeros.")
            self.feature_importances_ = np.zeros(len(self.feature_names))

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
            'signal_classes': self.signal_classes,
            'feature_names': self.feature_names,
            'use_gpu': self.use_gpu,  # Store GPU mode preference
            'calibration_method': self.calibration_method,
            'calibrators': self.calibrators if self.calibrators else {},
            'sharpening_alpha': self.sharpening_alpha,
            'feature_importances_': self.feature_importances_ if self.feature_importances_ is not None else None
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
        self.signal_classes = model_data.get('signal_classes', self.signal_classes)
        self.feature_names = model_data.get('feature_names', [])
        # Restore GPU mode if model was trained on GPU (will auto-detect on next predict)
        saved_use_gpu = model_data.get('use_gpu', False)
        if saved_use_gpu and self.use_gpu:
            try:
                self._cudf, self._cuRFClassifier = self._require_cuml()
                print("[GPU] cuML enabled for SignalBlender (loaded)")
            except Exception as exc:
                print(f"[CPU] SignalBlender trained on GPU but cuML unavailable at load time, falling back to CPU: {exc}")
                self.use_gpu = False
                print("[CPU] sklearn backend active for SignalBlender (loaded)")
        else:
            if saved_use_gpu and not self.use_gpu:
                print("[CPU] SignalBlender trained on GPU, running with CPU backend per configuration")
            self.use_gpu = False
            print("[CPU] sklearn backend active for SignalBlender (loaded)")
        
        # Restore calibration settings
        self.calibration_method = model_data.get('calibration_method', None)
        self.calibrators = model_data.get('calibrators', {})
        self.sharpening_alpha = model_data.get('sharpening_alpha', 1.0)
        self.feature_importances_ = model_data.get('feature_importances_', None)
        if self.calibration_method:
            print(f"[SignalBlender] Calibration: {self.calibration_method}, alpha: {self.sharpening_alpha}")
        
        print(f"Signal blender loaded from {path}")


class DirectionBlender(BlenderBase):
    """
    GPU-accelerated binary direction classifier using cuML RandomForestClassifier.
    Falls back to CPU (scikit-learn) if cuML is unavailable.
    Predicts only direction (1 for long, -1 for short) on trade samples (excludes flat/0).
    """

    def __init__(self, model_params: Optional[dict] = None, use_gpu: bool = False):
        """
        Initialize direction blender.
        
        Args:
            model_params: Optional cuML RandomForestClassifier parameters
            use_gpu: Whether to use GPU acceleration (default: True, falls back to CPU if unavailable)
        """
        if model_params is None:
            # cuML RandomForestClassifier parameters - lightweight for reduced VRAM
            model_params = {
                'n_estimators': 200,
                'max_depth': 22,
                'max_features': 0.6,
                'max_samples': 0.6,
                'n_bins': 128,
                'split_criterion': 0,  # 0=GINI, 1=ENTROPY
                'bootstrap': True,
                'n_streams': 1,  # Critical for reproducibility and lower VRAM spikes
                'random_state': 42
            }
        self.model_params = model_params
        self.use_gpu = use_gpu
        self.model = None
        self.direction_classes = [-1, 1]  # short, long (binary, no flat)
        self.feature_names = []
        self.feature_importances_ = None  # Will be populated after training
        self.name = "DirectionBlender"  # For saving feature importances
        self._cudf = None
        self._cuRFClassifier = None
        
        # Calibration settings - defaults to always calibrate
        self.calibration_method = 'isotonic'  # Default: 'isotonic', can be 'platt' or None to disable
        self.calibrators = {}  # Dict mapping class index to calibrator
        self.sharpening_alpha = 2.0  # Post-hoc sharpening: prob = prob ** alpha (default: 2.0)
        
        if self.use_gpu:
            self._cudf, self._cuRFClassifier = self._require_cuml()
            print("[GPU] cuML enabled for DirectionBlender")
        else:
            print("[CPU] sklearn backend active for DirectionBlender")

    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        calibration_method: Optional[str] = None,
        sharpening_alpha: Optional[float] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        disable_calibration: bool = False
    ) -> None:
        """
        Train the direction blending model.
        Calibration is ALWAYS enabled by default unless explicitly disabled.
        
        Args:
            X: Feature dataframe (includes features + module signals + regime)
            y: Target labels (-1 for short, 1 for long) - should only contain -1 and 1
            calibration_method: Calibration method ('isotonic', 'platt', or None to use default 'isotonic')
            sharpening_alpha: Post-hoc sharpening parameter (prob = prob ** alpha, default: 2.0)
            X_val: Optional validation features for calibration (if None, auto-splits 20% from X)
            y_val: Optional validation labels for calibration (if None, auto-splits 20% from y)
            disable_calibration: If True, disables calibration entirely
        """
        # Clean features
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.ffill().bfill().fillna(0)
        
        # Encode regime column if it exists (convert to numeric)
        if 'regime' in X_clean.columns:
            regime_map = {
                'trend_up': 0, 'trend_down': 1, 'chop': 2
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(2)  # Default to chop (2)
        
        # Convert y to direction labels (-1, 1) - ensure binary
        y_clean = y.copy()
        if y_clean.dtype in [np.float64, np.float32]:
            # If y is returns or other numeric, convert to direction: sign(return)
            y_clean = np.sign(y_clean).astype(int)
        else:
            # If y is already signals, ensure they're in [-1, 1]
            y_clean = y_clean.astype(int)
            y_clean = y_clean.clip(-1, 1)
        
        # Filter to only -1 and 1 (exclude 0/flat)
        valid_directions = set(self.direction_classes)
        y_clean = y_clean[y_clean.isin(valid_directions)]
        X_clean = X_clean.loc[y_clean.index]
        
        # Convert to numeric labels for cuML (0, 1) where -1 -> 0, 1 -> 1
        label_map = {-1: 0, 1: 1}  # short=0, long=1
        y_numeric = y_clean.map(label_map)
        
        # Align indices
        common_idx = X_clean.index.intersection(y_numeric.index)
        X_clean = X_clean.loc[common_idx]
        y_numeric = y_numeric.loc[common_idx]
        
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
            cudf = self._cudf or self._require_cuml()[0]
            cuRFClassifier = self._cuRFClassifier or self._require_cuml()[1]
            print(f"[DirectionBlender] Converting {len(X_clean)} samples to cuDF for GPU training...")
            X_gpu = cudf.from_pandas(X_clean)
            y_gpu = cudf.Series(y_numeric.values, dtype='int32')

            # Initialize and train model on GPU
            self.model = cuRFClassifier(**self.model_params)
            self.model.fit(X_gpu, y_gpu)

            print(f"[DirectionBlender] GPU training complete: {len(X_clean)} samples, {len(self.feature_names)} features")
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            print("[CPU] sklearn backend active for DirectionBlender")
            self.model = GradientBoostingClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 16),
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_clean, y_numeric)
            print(f"[DirectionBlender] CPU training complete: {len(X_clean)} samples, {len(self.feature_names)} features")
        
        # Store feature importances from trained model
        self._store_feature_importances()
        
        # Store calibration settings - use defaults if not specified
        if disable_calibration:
            self.calibration_method = None
            self.sharpening_alpha = 1.0
        else:
            # Default to isotonic if not specified
            self.calibration_method = calibration_method if calibration_method is not None else 'isotonic'
            self.sharpening_alpha = sharpening_alpha if sharpening_alpha is not None else 2.0
        
        # Prepare validation data for calibration
        # If X_val/y_val not provided, automatically split 20% from training data
        X_cal = None
        y_cal_numeric = None
        
        if self.calibration_method and CALIBRATION_AVAILABLE:
            if X_val is not None and y_val is not None:
                # Clean validation features
                X_cal = X_val.copy()
                X_cal = X_cal.replace([np.inf, -np.inf], np.nan)
                X_cal = X_cal.ffill().bfill().fillna(0)
                
                # Encode regime if needed
                if 'regime' in X_cal.columns:
                    regime_map = {'trend_up': 0, 'trend_down': 1, 'chop': 2}
                    X_cal['regime'] = X_cal['regime'].map(regime_map).fillna(2)
                
                # Convert validation labels
                y_cal = y_val.copy()
                if y_cal.dtype in [np.float64, np.float32]:
                    y_cal = np.sign(y_cal).astype(int)
                else:
                    y_cal = y_cal.astype(int).clip(-1, 1)
                
                valid_directions = set(self.direction_classes)
                y_cal = y_cal[y_cal.isin(valid_directions)]
                X_cal = X_cal.loc[y_cal.index]
                
                label_map = {-1: 0, 1: 1}
                y_cal_numeric = y_cal.map(label_map)
                
                # Align indices
                common_idx = X_cal.index.intersection(y_cal_numeric.index)
                X_cal = X_cal.loc[common_idx]
                y_cal_numeric = y_cal_numeric.loc[common_idx]
            else:
                # Auto-split 20% for validation
                from sklearn.model_selection import train_test_split
                print(f"[DirectionBlender] Auto-splitting 20% of data for calibration validation...")
                X_train_cal, X_cal, y_train_cal, y_cal_numeric = train_test_split(
                    X_clean, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
                )
                # Retrain on the 80% split (more accurate calibration)
                print(f"[DirectionBlender] Retraining on 80% split ({len(X_train_cal)} samples)...")
                if self.use_gpu:
                    cudf = self._cudf or self._require_cuml()[0]
                    cuRFClassifier = self._cuRFClassifier or self._require_cuml()[1]
                    X_train_gpu = cudf.from_pandas(X_train_cal)
                    y_train_gpu = cudf.Series(y_train_cal.values, dtype='int32')
                    self.model = cuRFClassifier(**self.model_params)
                    self.model.fit(X_train_gpu, y_train_gpu)
                else:
                    self.model.fit(X_train_cal, y_train_cal)
                print(f"[DirectionBlender] Using {len(X_cal)} samples for calibration")
        
        # Apply calibration if enabled
        if self.calibration_method and CALIBRATION_AVAILABLE:
            self._fit_calibration(X_clean, y_numeric, X_cal, y_cal_numeric)
    
    def _store_feature_importances(self) -> None:
        """
        Store feature importances if available.
        Skip safely when using cuML (GPU), because cuML RandomForest
        does NOT expose feature_importances_.
        """
        # GPU MODE: cuML does NOT support feature importances. Skip gracefully.
        if self.use_gpu:
            print(f"[{self.__class__.__name__}] GPU mode: feature importances unavailable (cuML). Skipping.")
            self.feature_importances_ = np.zeros(len(self.feature_names))
            return
        
        # CPU MODE (sklearn): feature importances available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_
        else:
            print(f"[{self.__class__.__name__}] CPU model has no feature_importances_. Using zeros.")
            self.feature_importances_ = np.zeros(len(self.feature_names))
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict direction (long/short) for given features.
        
        Args:
            X: Feature dataframe (includes features + module signals + regime)
            
        Returns:
            Series of directions (-1 for short, 1 for long)
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
                'trend_up': 0, 'trend_down': 1, 'chop': 2
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(2)  # Default to chop (2)
        
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
            cudf = self._cudf or self._require_cuml()[0]
            X_gpu = cudf.from_pandas(X_clean)
            predictions_numeric_gpu = self.model.predict(X_gpu)
            # Convert back to pandas
            predictions_numeric = predictions_numeric_gpu.to_pandas().values
        else:
            # CPU: Predict directly
            predictions_numeric = self.model.predict(X_clean)
        
        # Convert back to direction labels (-1, 1)
        label_map = {0: -1, 1: 1}  # short=-1, long=1
        directions = pd.Series(
            [label_map[pred] for pred in predictions_numeric],
            index=X_clean.index,
            dtype=int,
            name='direction'
        )
        
        return directions

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict direction probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with probabilities for each direction class (short, long)
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
                'trend_up': 0, 'trend_down': 1, 'chop': 2
            }
            X_clean['regime'] = X_clean['regime'].map(regime_map).fillna(2)  # Default to chop (2)
        
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
            cudf = self._cudf or self._require_cuml()[0]
            X_gpu = cudf.from_pandas(X_clean)
            proba_gpu = self.model.predict_proba(X_gpu)
            # Convert back to pandas
            proba_raw = proba_gpu.to_pandas().values
        else:
            # CPU: Predict probabilities directly
            proba_raw = self.model.predict_proba(X_clean)
        
        # Apply calibration and sharpening
        proba = self._apply_calibration_and_sharpening(proba_raw)
        
        # Create dataframe with direction labels as columns
        proba_df = pd.DataFrame(
            proba,
            index=X_clean.index,
            columns=self.direction_classes
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
            'direction_classes': self.direction_classes,
            'feature_names': self.feature_names,
            'use_gpu': self.use_gpu,  # Store GPU mode preference
            'calibration_method': self.calibration_method,
            'calibrators': self.calibrators if self.calibrators else {},
            'sharpening_alpha': self.sharpening_alpha,
            'feature_importances_': self.feature_importances_ if self.feature_importances_ is not None else None
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Direction blender saved to {path}")

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
        self.direction_classes = model_data.get('direction_classes', self.direction_classes)
        self.feature_names = model_data.get('feature_names', [])
        # Restore GPU mode if model was trained on GPU (will auto-detect on next predict)
        saved_use_gpu = model_data.get('use_gpu', False)
        if saved_use_gpu and self.use_gpu:
            self._cudf, self._cuRFClassifier = self._require_cuml()
            print("[GPU] cuML enabled for DirectionBlender (loaded)")
        else:
            if saved_use_gpu and not self.use_gpu:
                print("[CPU] DirectionBlender trained on GPU, running with CPU backend per configuration")
            self.use_gpu = False
            print("[CPU] sklearn backend active for DirectionBlender (loaded)")
        
        # Restore calibration settings
        self.calibration_method = model_data.get('calibration_method', None)
        self.calibrators = model_data.get('calibrators', {})
        self.sharpening_alpha = model_data.get('sharpening_alpha', 1.0)
        self.feature_importances_ = model_data.get('feature_importances_', None)
        if self.calibration_method:
            print(f"[DirectionBlender] Calibration: {self.calibration_method}, alpha: {self.sharpening_alpha}")
        
        print(f"Direction blender loaded from {path}")