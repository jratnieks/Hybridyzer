# core/scalers.py
"""
GPU-aware feature scalers with CPU fallback.
Supports z-score normalization, min-max scaling, and robust scaling (IQR-based).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional

# GPU support with automatic CPU fallback
try:
    import cupy as cp
    import cudf
    # Test if GPU is actually accessible
    try:
        test_arr = cp.array([1.0, 2.0, 3.0])
        _ = test_arr * 2
        GPU_AVAILABLE = True
    except (RuntimeError, Exception):
        GPU_AVAILABLE = False
        cp = None
        cudf = None
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    cudf = None


class Scaler:
    """
    Base class for feature scalers.
    """
    
    def fit(self, df):
        """Fit scaling parameters from data."""
        raise NotImplementedError
    
    def transform(self, df):
        """Transform data using fitted parameters."""
        raise NotImplementedError
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def save(self, path: str) -> None:
        """Save scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        """Load scaler from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class ZScoreScaler(Scaler):
    """
    Z-score normalization: (x - mean) / std
    GPU-aware with CuPy/cuDF support, falls back to CPU.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize z-score scaler.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE and (cp is not None and cudf is not None)
        self.params = {}
        self.is_fitted = False
    
    def _is_gpu_df(self, df) -> bool:
        """Check if input is a cuDF DataFrame."""
        return self.use_gpu and hasattr(df, '__class__') and 'cudf' in str(type(df))
    
    def fit(self, df) -> None:
        """Fit scaling parameters (mean and std) from data."""
        if self._is_gpu_df(df):
            # GPU path: cuDF
            numeric_cols = [col for col in df.columns if df[col].dtype in [cp.float32, cp.float64, cp.int32, cp.int64]]
            for col in numeric_cols:
                mean = float(df[col].mean())
                std = float(df[col].std())
                self.params[col] = {'mean': mean, 'std': std}
        else:
            # CPU path: pandas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean = float(df[col].mean())
                std = float(df[col].std())
                self.params[col] = {'mean': mean, 'std': std}
        
        self.is_fitted = True
    
    def transform(self, df):
        """Transform data using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        
        if self._is_gpu_df(df):
            # GPU path: cuDF
            df_scaled = df.copy()
            for col, params in self.params.items():
                if col in df_scaled.columns:
                    mean = params['mean']
                    std = params['std']
                    if std > 0:
                        df_scaled[col] = (df_scaled[col] - mean) / std
                    else:
                        df_scaled[col] = cp.float32(0.0)
            return df_scaled
        else:
            # CPU path: pandas
            df_scaled = df.copy()
            for col, params in self.params.items():
                if col in df_scaled.columns:
                    mean = params['mean']
                    std = params['std']
                    if std > 0:
                        df_scaled[col] = (df_scaled[col] - mean) / std
                    else:
                        df_scaled[col] = 0.0
            return df_scaled


class MinMaxScaler(Scaler):
    """
    Min-max scaling: (x - min) / (max - min)
    GPU-aware with CuPy/cuDF support, falls back to CPU.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize min-max scaler.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE and (cp is not None and cudf is not None)
        self.params = {}
        self.is_fitted = False
    
    def _is_gpu_df(self, df) -> bool:
        """Check if input is a cuDF DataFrame."""
        return self.use_gpu and hasattr(df, '__class__') and 'cudf' in str(type(df))
    
    def fit(self, df) -> None:
        """Fit scaling parameters (min and max) from data."""
        if self._is_gpu_df(df):
            # GPU path: cuDF
            numeric_cols = [col for col in df.columns if df[col].dtype in [cp.float32, cp.float64, cp.int32, cp.int64]]
            for col in numeric_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                range_val = max_val - min_val
                self.params[col] = {'min': min_val, 'max': max_val, 'range': range_val}
        else:
            # CPU path: pandas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                range_val = max_val - min_val
                self.params[col] = {'min': min_val, 'max': max_val, 'range': range_val}
        
        self.is_fitted = True
    
    def transform(self, df):
        """Transform data using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        
        if self._is_gpu_df(df):
            # GPU path: cuDF
            df_scaled = df.copy()
            for col, params in self.params.items():
                if col in df_scaled.columns:
                    min_val = params['min']
                    range_val = params['range']
                    if range_val > 0:
                        df_scaled[col] = (df_scaled[col] - min_val) / range_val
                    else:
                        df_scaled[col] = cp.float32(0.0)
            return df_scaled
        else:
            # CPU path: pandas
            df_scaled = df.copy()
            for col, params in self.params.items():
                if col in df_scaled.columns:
                    min_val = params['min']
                    range_val = params['range']
                    if range_val > 0:
                        df_scaled[col] = (df_scaled[col] - min_val) / range_val
                    else:
                        df_scaled[col] = 0.0
            return df_scaled


class RobustScaler(Scaler):
    """
    Robust scaling (IQR-based): (x - median) / IQR
    GPU-aware with CuPy/cuDF support, falls back to CPU.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize robust scaler.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: True)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE and (cp is not None and cudf is not None)
        self.params = {}
        self.is_fitted = False
    
    def _is_gpu_df(self, df) -> bool:
        """Check if input is a cuDF DataFrame."""
        return self.use_gpu and hasattr(df, '__class__') and 'cudf' in str(type(df))
    
    def fit(self, df) -> None:
        """Fit scaling parameters (median and IQR) from data."""
        if self._is_gpu_df(df):
            # GPU path: cuDF
            numeric_cols = [col for col in df.columns if df[col].dtype in [cp.float32, cp.float64, cp.int32, cp.int64]]
            for col in numeric_cols:
                median = float(df[col].median())
                q75 = float(df[col].quantile(0.75))
                q25 = float(df[col].quantile(0.25))
                iqr = q75 - q25
                # Fallback std if IQR is 0
                std = float(df[col].std()) if iqr == 0 else None
                self.params[col] = {'median': median, 'q25': q25, 'q75': q75, 'iqr': iqr, 'std': std}
        else:
            # CPU path: pandas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                median = float(df[col].median())
                q75 = float(df[col].quantile(0.75))
                q25 = float(df[col].quantile(0.25))
                iqr = q75 - q25
                # Fallback std if IQR is 0
                std = float(df[col].std()) if iqr == 0 else None
                self.params[col] = {'median': median, 'q25': q25, 'q75': q75, 'iqr': iqr, 'std': std}
        
        self.is_fitted = True
    
    def transform(self, df):
        """Transform data using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        
        if self._is_gpu_df(df):
            # GPU path: cuDF
            df_scaled = df.copy()
            for col, params in self.params.items():
                if col in df_scaled.columns:
                    median = params['median']
                    iqr = params['iqr']
                    std = params['std']
                    if iqr > 0:
                        df_scaled[col] = (df_scaled[col] - median) / iqr
                    elif std is not None and std > 0:
                        df_scaled[col] = (df_scaled[col] - median) / std
                    else:
                        df_scaled[col] = cp.float32(0.0)
            return df_scaled
        else:
            # CPU path: pandas
            df_scaled = df.copy()
            for col, params in self.params.items():
                if col in df_scaled.columns:
                    median = params['median']
                    iqr = params['iqr']
                    std = params['std']
                    if iqr > 0:
                        df_scaled[col] = (df_scaled[col] - median) / iqr
                    elif std is not None and std > 0:
                        df_scaled[col] = (df_scaled[col] - median) / std
                    else:
                        df_scaled[col] = 0.0
            return df_scaled

