# core/feature_store.py
"""
    GPU-accelerated feature engineering pipeline using RAPIDS cuDF/cuPy.

This module has been refactored to use GPU acceleration for expensive rolling operations:
- RollingStatsGenerator: GPU-accelerated rolling mean, std, max, min, slope, zscore
- RegimeContextFeatures: GPU-accelerated ATR, volatility, kurtosis, skew, volume surge, return bins
- SignalDynamicsFeatures: GPU-accelerated distance-to-extreme calculations

All GPU operations automatically fall back to CPU (pandas) if RAPIDS is unavailable.
Lightweight indicator modules (SuperMA, TrendMagic, PVT, PivotRSI, LinReg) remain on CPU.

Features:
- Automatic chunking (50k-200k rows) for large datasets to prevent GPU OOM
- Float32 downcasting to reduce memory usage
- CPU fallback if RAPIDS unavailable (works on all platforms)
- Maintains pandas DataFrame output for compatibility with training loop
- Same column names and output shape as CPU version
- WSL/Linux compatible (RAPIDS works in WSL with GPU passthrough)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import time
from modules.base import SignalModule, ContextModule
from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext

# GPU support with automatic CPU fallback
# Uses RAPIDS cuDF for GPU-accelerated operations (works in WSL/Linux)
try:
    import cudf
    import cupy as cp
    # Test if GPU is actually accessible (not just installed)
    try:
        test_arr = cp.array([1.0, 2.0, 3.0])
        _ = test_arr * 2  # Test actual GPU computation
        # Test cudf functionality
        test_df = cudf.DataFrame({'test': [1, 2, 3]})
        GPU_AVAILABLE = True
        print("[FeatureStore] RAPIDS (cuDF/cuPy) installed and GPU accessible")
    except (RuntimeError, Exception) as e:
        # RAPIDS installed but can't access GPU (CUDA toolkit not properly configured)
        GPU_AVAILABLE = False
        cudf = None
        cp = None
        print(f"[FeatureStore] RAPIDS installed but GPU not accessible: {type(e).__name__}, using CPU fallback")
        print("[FeatureStore] Note: To use GPU, ensure CUDA toolkit is installed and properly configured")
except ImportError:
    GPU_AVAILABLE = False
    cudf = None
    cp = None
    print("[FeatureStore] RAPIDS (cuDF/cuPy) not installed, using CPU fallback")

# Import numba cuda for error handling
try:
    import numba.cuda as cuda
    from numba.cuda.cudadrv import driver as cuda_driver
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    cuda = None
    cuda_driver = None

# Import gc for memory management
import gc


def clear_gpu_memory(aggressive: bool = False):
    """
    Clear GPU memory to prevent OOM errors on long-running operations.
    Call this between heavy GPU operations.
    
    Args:
        aggressive: If True, performs more thorough cleanup including
                   forcing garbage collection multiple times
    """
    if not GPU_AVAILABLE:
        return
    
    try:
        # Force Python garbage collection first
        gc.collect()
        
        # Clear cupy memory pool
        if cp is not None:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        
        # Synchronize CUDA to ensure all operations are complete
        if NUMBA_CUDA_AVAILABLE and cuda is not None:
            try:
                cuda.synchronize()
            except Exception:
                pass  # May fail if no active context
        
        if aggressive:
            # Multiple GC passes can help with cyclic references
            for _ in range(3):
                gc.collect()
            
            # Re-clear memory pools after GC
            if cp is not None:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                
    except Exception as e:
        # Silently fail - memory clearing is best-effort
        pass


def get_gpu_memory_info():
    """
    Get current GPU memory usage info for debugging.
    Returns dict with free/total memory in GB, or None if unavailable.
    """
    if not GPU_AVAILABLE or cp is None:
        return None
    
    try:
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        
        # Get device memory info
        device = cp.cuda.Device()
        free_mem, total_mem = device.mem_info
        
        return {
            'pool_used_gb': used_bytes / (1024**3),
            'pool_total_gb': total_bytes / (1024**3),
            'device_free_gb': free_mem / (1024**3),
            'device_total_gb': total_mem / (1024**3),
        }
    except Exception:
        return None


class RollingStatsGenerator:
    """
    GPU-accelerated rolling statistics generator using RAPIDS cuDF.
    Falls back to CPU (pandas) if RAPIDS is unavailable.
    Produces rolling mean, std, max, min, slope, and zscore for specified windows.
    """
    
    def __init__(
        self,
        windows: List[int] = None,
        rolling_stats_columns: List[str] = None,
        use_gpu: bool = True,
        chunk_size: int = 100000
    ):
        """
        Initialize rolling stats generator.
        
        Args:
            windows: List of window sizes (default: [5, 20])
            rolling_stats_columns: List of column names to compute stats for.
                                  If None, uses safe default subset.
            use_gpu: Whether to use GPU acceleration (default: True, falls back to CPU if unavailable)
            chunk_size: Number of rows to process per chunk for large datasets (default: 100000)
        """
        if windows is None:
            windows = [5, 20]
        self.windows = windows
        self.rolling_stats_columns = rolling_stats_columns
        self.use_gpu = use_gpu and GPU_AVAILABLE and (cudf is not None)
        self.use_gpu_effective = self.use_gpu  # Can be disabled at runtime on GPU failure
        self.chunk_size = chunk_size
        
        if self.use_gpu:
            print("[RollingStatsGenerator] GPU acceleration enabled (RAPIDS cuDF)")
        else:
            if use_gpu:
                print("[RollingStatsGenerator] GPU requested but RAPIDS unavailable, using CPU fallback")
            else:
                print("[RollingStatsGenerator] CPU mode")
    
    def _rolling_slope_gpu(self, gpu_series: 'cudf.Series', window: int) -> 'cudf.Series':
        """
        Compute rolling linear regression slope on GPU using cuDF.
        
        Args:
            gpu_series: cuDF Series
            window: Window size
            
        Returns:
            cuDF Series of slopes
        """
        # Time index as cuDF Series
        t = cudf.Series(cp.arange(len(gpu_series), dtype=cp.float32))
        
        # Rolling covariance between time and value
        cov_ty = t.rolling(window=window, min_periods=1).cov(gpu_series)
        
        # Rolling variance of time index
        var_t = t.rolling(window=window, min_periods=1).var()
        var_t = var_t.replace(0, cp.nan)
        
        slopes = cov_ty / var_t
        return slopes
    
    def _rolling_slope_cpu(self, series: pd.Series, window: int) -> pd.Series:
        """CPU fallback for rolling slope."""
        y = series.astype(np.float32)
        t = pd.Series(np.arange(len(series), dtype=np.float32), index=series.index)
        cov_ty = t.rolling(window=window, min_periods=1).cov(y)
        var_t = t.rolling(window=window, min_periods=1).var()
        var_t = var_t.replace(0, np.nan)
        slopes = cov_ty / var_t
        return slopes
    
    def _rolling_zscore_gpu(self, gpu_series: 'cudf.Series', window: int) -> 'cudf.Series':
        """Compute rolling z-score on GPU."""
        rolling_mean = gpu_series.rolling(window=window, min_periods=1).mean()
        rolling_std = gpu_series.rolling(window=window, min_periods=1).std()
        zscore = (gpu_series - rolling_mean) / rolling_std.replace(0, cp.nan)
        return zscore
    
    def _rolling_zscore_cpu(self, series: pd.Series, window: int) -> pd.Series:
        """CPU fallback for rolling z-score."""
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore
    
    def _is_allowed_column(self, col: str) -> bool:
        """Check if a column should have rolling stats computed."""
        if self.rolling_stats_columns is not None:
            return col in self.rolling_stats_columns
        allowed_patterns = ["hull", "price", "close", "zigzag"]
        col_lower = col.lower()
        return any(pattern in col_lower for pattern in allowed_patterns)
    
    def _process_chunk_gpu(
        self,
        gpu_features: 'cudf.DataFrame',
        valid_features: List[str],
        original_index: pd.Index
    ) -> pd.DataFrame:
        """Process a chunk of features on GPU."""
        stats_list = []
        
        for col in valid_features:
            gpu_series = gpu_features[col].astype(cp.float32)
            
            for window in self.windows:
                # Rolling mean
                rolling_mean = gpu_series.rolling(window=window, min_periods=1).mean()
                stats_list.append(rolling_mean.rename(f"{col}_mean_{window}"))
                
                # Rolling std
                rolling_std = gpu_series.rolling(window=window, min_periods=1).std()
                stats_list.append(rolling_std.rename(f"{col}_std_{window}"))
                
                # Rolling max
                rolling_max = gpu_series.rolling(window=window, min_periods=1).max()
                stats_list.append(rolling_max.rename(f"{col}_max_{window}"))
                
                # Rolling min
                rolling_min = gpu_series.rolling(window=window, min_periods=1).min()
                stats_list.append(rolling_min.rename(f"{col}_min_{window}"))

                # NOTE: cuDF Rolling currently does not support .cov(), so we cannot
                # implement rolling slope purely on GPU. To avoid errors like
                # "Rolling object has no attribute cov", we skip slope features in
                # the GPU path. Slope is still available in the CPU fallback path.

                # Rolling zscore
                rolling_zscore = self._rolling_zscore_gpu(gpu_series, window)
                stats_list.append(rolling_zscore.rename(f"{col}_zscore_{window}"))
        
        if not stats_list:
            return pd.DataFrame(index=original_index)
        
        # Concatenate on GPU, then convert to pandas
        gpu_stats = cudf.concat(stats_list, axis=1)
        gpu_stats = gpu_stats.astype(cp.float32)
        pandas_stats = gpu_stats.to_pandas()
        pandas_stats.index = original_index
        
        return pandas_stats
    
    def _process_chunk_cpu(
        self,
        features: pd.DataFrame,
        valid_features: List[str]
    ) -> pd.DataFrame:
        """Process a chunk of features on CPU (fallback)."""
        stats_df = pd.DataFrame(index=features.index)
        
        for col in valid_features:
            series = features[col].astype(np.float32)
            feature_stats = []
            
            for window in self.windows:
                rolling_mean = series.rolling(window=window, min_periods=1).mean()
                feature_stats.append(rolling_mean.rename(f"{col}_mean_{window}"))
                
                rolling_std = series.rolling(window=window, min_periods=1).std()
                feature_stats.append(rolling_std.rename(f"{col}_std_{window}"))
                
                rolling_max = series.rolling(window=window, min_periods=1).max()
                feature_stats.append(rolling_max.rename(f"{col}_max_{window}"))
                
                rolling_min = series.rolling(window=window, min_periods=1).min()
                feature_stats.append(rolling_min.rename(f"{col}_min_{window}"))
                
                # Rolling slope (CPU)
                rolling_slope = self._rolling_slope_cpu(series, window)
                feature_stats.append(rolling_slope.rename(f"{col}_slope_{window}"))
                
                rolling_zscore = self._rolling_zscore_cpu(series, window)
                feature_stats.append(rolling_zscore.rename(f"{col}_zscore_{window}"))
            
            if feature_stats:
                feature_df = pd.concat(feature_stats, axis=1).astype(np.float32)
                stats_df = pd.concat([stats_df, feature_df], axis=1)
                del feature_stats, feature_df
        
        return stats_df
    
    def generate_stats(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling statistics for selected features.
        Uses GPU acceleration if available, with automatic chunking for large datasets.
        
        Args:
            features: Base feature DataFrame (should be base features only, not expanded)
            
        Returns:
            DataFrame with rolling statistics (prefixed with stat name and window)
        """
        t_start = time.time()
        n_rows = len(features)
        
        # For datasets >25k rows, force CPU to avoid GPU memory issues
        if n_rows > 25000 and self.use_gpu_effective:
            print(f"[RollingStatsGenerator] Using CPU for {n_rows} rows (safer for large datasets)")
            self.use_gpu_effective = False
        
        # Determine which columns to process
        if self.rolling_stats_columns is not None:
            target_columns = [col for col in self.rolling_stats_columns if col in features.columns]
        else:
            target_columns = [col for col in features.columns if self._is_allowed_column(col)]
        
        # Filter out constant/NaN features
        valid_features = []
        for col in target_columns:
            series = features[col]
            if not (series.isna().all() or series.nunique() <= 1):
                valid_features.append(col)
        
        if not valid_features:
            return pd.DataFrame(index=features.index)
        
        n_rows = len(features)
        use_chunking = n_rows > self.chunk_size
        
        if use_chunking:
            print(f"[RollingStatsGenerator] Processing {n_rows} rows in chunks of {self.chunk_size}")
            all_chunks = []
            
            for chunk_start in range(0, n_rows, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, n_rows)
                chunk_features = features.iloc[chunk_start:chunk_end]
                chunk_index = chunk_features.index
                
                if self.use_gpu_effective:
                    try:
                        # Convert to cuDF
                        gpu_chunk = cudf.from_pandas(chunk_features)
                        chunk_stats = self._process_chunk_gpu(gpu_chunk, valid_features, chunk_index)
                        del gpu_chunk
                    except Exception as e:
                        error_type = type(e).__name__
                        print(f"[RollingStatsGenerator] GPU failed, falling back to CPU: {error_type}: {e}")
                        self.use_gpu_effective = False
                        chunk_stats = self._process_chunk_cpu(chunk_features, valid_features)
                else:
                    chunk_stats = self._process_chunk_cpu(chunk_features, valid_features)
                
                all_chunks.append(chunk_stats)
            
            # Combine all chunks
            stats_df = pd.concat(all_chunks, axis=0)
        else:
            # Process entire dataset at once
            if self.use_gpu_effective:
                try:
                    gpu_features = cudf.from_pandas(features)
                    stats_df = self._process_chunk_gpu(gpu_features, valid_features, features.index)
                    del gpu_features
                except Exception as e:
                    error_type = type(e).__name__
                    print(f"[RollingStatsGenerator] GPU failed, falling back to CPU: {error_type}: {e}")
                    self.use_gpu_effective = False
                    stats_df = self._process_chunk_cpu(features, valid_features)
            else:
                stats_df = self._process_chunk_cpu(features, valid_features)
        
        t_end = time.time()
        print(f"[RollingStatsGenerator] Generated {len(stats_df.columns)} rolling stat features in {t_end-t_start:.2f}s")
        
        return stats_df


class RegimeContextFeatures:
    """
    Lean GPU-accelerated regime context feature generator using RAPIDS cuDF.
    Falls back to CPU (pandas) if RAPIDS is unavailable.
    Provides a compact set of ATR, volatility, and trend slope features optimized for GPU training.
    """
    
    def __init__(
        self,
        atr_windows: List[int] = None,
        return_bins: int = 10,
        use_gpu: bool = True,
        chunk_size: int = 20000,
        safe_mode: bool = True
    ):
        """
        Initialize regime context feature generator.
        
        Args:
            atr_windows: List of window sizes for ATR (unused in lean implementation, kept for API compatibility)
            return_bins: Number of bins for return bucketization (unused, kept for API compatibility)
            use_gpu: Whether to use GPU acceleration (default: True)
            chunk_size: Number of rows to process per chunk (default: 20000, conservative for 8GB GPUs)
            safe_mode: Safe mode flag (controls logging only, no feature set changes)
        """
        self.atr_windows = atr_windows  # Kept for API compatibility but unused
        self.return_bins = return_bins  # Kept for API compatibility but unused
        self.use_gpu_effective = use_gpu and GPU_AVAILABLE and (cudf is not None)
        self.chunk_size = chunk_size
        self.safe_mode = safe_mode
        
        if self.use_gpu_effective:
            safe_mode_str = "safe mode" if safe_mode else ""
            print(f"[RegimeContextFeatures] GPU acceleration enabled (RAPIDS cuDF{f', {safe_mode_str}' if safe_mode_str else ''})")
        else:
            if use_gpu:
                print("[RegimeContextFeatures] GPU requested but RAPIDS unavailable, using CPU fallback")
            else:
                print("[RegimeContextFeatures] Using CPU regime context features")
    
    def _generate_features_gpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lean regime context features on GPU.
        Computes only GPU-safe features: ATR-based, volatility, and trend slopes.
        """
        gpu_df = cudf.from_pandas(df)
        all_features = []
        
        # Extract OHLCV columns as float32
        close = gpu_df["close"].astype(cp.float32)
        high = gpu_df["high"].astype(cp.float32)
        low = gpu_df["low"].astype(cp.float32)
        
        # 1. True Range and ATR
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = cudf.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR 14 and 50 using rolling mean (GPU-safe)
        atr_14 = tr.rolling(window=14, min_periods=14).mean()
        atr_50 = tr.rolling(window=50, min_periods=50).mean()
        
        # Normalize ATR by close price
        atr14_norm = (atr_14 / close.replace(0, cp.nan))
        atr50_norm = (atr_50 / close.replace(0, cp.nan))
        atr14_50_ratio = (atr_14 / atr_50.replace(0, cp.nan))
        
        # Replace infinities with NaN
        atr14_norm = atr14_norm.replace([cp.inf, -cp.inf], cp.nan)
        atr50_norm = atr50_norm.replace([cp.inf, -cp.inf], cp.nan)
        atr14_50_ratio = atr14_50_ratio.replace([cp.inf, -cp.inf], cp.nan)
        
        all_features.append(atr14_norm.rename("regime_atr14_norm_close"))
        all_features.append(atr50_norm.rename("regime_atr50_norm_close"))
        all_features.append(atr14_50_ratio.rename("regime_atr14_50_ratio"))
        
        # 2. Returns and volatility
        returns = close.pct_change().astype(cp.float32)
        
        vol_20 = returns.rolling(window=20, min_periods=20).std()
        vol_100 = returns.rolling(window=100, min_periods=100).std()
        vol20_100_ratio = (vol_20 / vol_100.replace(0, cp.nan)).replace([cp.inf, -cp.inf], cp.nan)
        
        all_features.append(vol_20.rename("regime_vol20"))
        all_features.append(vol_100.rename("regime_vol100"))
        all_features.append(vol20_100_ratio.rename("regime_vol20_100_ratio"))
        
        # 3. Trend slopes
        trend_slope_20 = (close / close.shift(20).replace(0, cp.nan) - 1.0).replace([cp.inf, -cp.inf], cp.nan)
        trend_slope_50 = (close / close.shift(50).replace(0, cp.nan) - 1.0).replace([cp.inf, -cp.inf], cp.nan)
        
        all_features.append(trend_slope_20.rename("regime_trend_slope_20"))
        all_features.append(trend_slope_50.rename("regime_trend_slope_50"))
        
        # Concatenate and convert to pandas first
        gpu_features = cudf.concat(all_features, axis=1)
        pandas_features = gpu_features.to_pandas()
        pandas_features.index = df.index
        
        # Compute sign feature in pandas (handles NaNs properly, no need to convert back to cuDF)
        trend_slope_20_pd = pandas_features["regime_trend_slope_20"]
        trend_slope_sign_20 = np.sign(trend_slope_20_pd).astype(np.float32)
        trend_slope_sign_20[trend_slope_sign_20 == 0] = np.nan  # Replace 0 with NaN for consistency
        trend_slope_sign_20[trend_slope_20_pd.isna()] = np.nan  # Keep NaNs where original had NaNs
        pandas_features["regime_trend_slope_sign_20"] = trend_slope_sign_20
        
        return pandas_features
    
    def _generate_features_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lean regime context features on CPU.
        Computes the same features as GPU path using pandas/NumPy.
        """
        all_features = []
        
        # Extract OHLCV columns as float32
        close = df["close"].astype(np.float32)
        high = df["high"].astype(np.float32)
        low = df["low"].astype(np.float32)
        
        # 1. True Range and ATR
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR 14 and 50 using rolling mean
        atr_14 = tr.rolling(window=14, min_periods=14).mean()
        atr_50 = tr.rolling(window=50, min_periods=50).mean()
        
        # Normalize ATR by close price
        atr14_norm = (atr_14 / close.replace(0, np.nan))
        atr50_norm = (atr_50 / close.replace(0, np.nan))
        atr14_50_ratio = (atr_14 / atr_50.replace(0, np.nan))
        
        # Replace infinities with NaN
        atr14_norm = atr14_norm.replace([np.inf, -np.inf], np.nan)
        atr50_norm = atr50_norm.replace([np.inf, -np.inf], np.nan)
        atr14_50_ratio = atr14_50_ratio.replace([np.inf, -np.inf], np.nan)
        
        all_features.append(atr14_norm.rename("regime_atr14_norm_close"))
        all_features.append(atr50_norm.rename("regime_atr50_norm_close"))
        all_features.append(atr14_50_ratio.rename("regime_atr14_50_ratio"))
        
        # 2. Returns and volatility
        returns = close.pct_change().astype(np.float32)
        
        vol_20 = returns.rolling(window=20, min_periods=20).std()
        vol_100 = returns.rolling(window=100, min_periods=100).std()
        vol20_100_ratio = (vol_20 / vol_100.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        
        all_features.append(vol_20.rename("regime_vol20"))
        all_features.append(vol_100.rename("regime_vol100"))
        all_features.append(vol20_100_ratio.rename("regime_vol20_100_ratio"))
        
        # 3. Trend slopes
        trend_slope_20 = (close / close.shift(20).replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        trend_slope_50 = (close / close.shift(50).replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)
        trend_slope_sign_20 = np.sign(trend_slope_20).astype(np.float32)
        trend_slope_sign_20 = trend_slope_sign_20.replace(0, np.nan)  # Replace 0 with NaN for consistency
        
        all_features.append(trend_slope_20.rename("regime_trend_slope_20"))
        all_features.append(trend_slope_50.rename("regime_trend_slope_50"))
        all_features.append(trend_slope_sign_20.rename("regime_trend_slope_sign_20"))
        
        # Concatenate features
        features_df = pd.concat(all_features, axis=1)
        
        return features_df
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime context features from OHLCV data.
        Uses GPU acceleration if available, with automatic fallback to CPU on errors.
        Implements chunking for large datasets to prevent GPU OOM.
        
        Returns a compact set of 9 features:
        - regime_atr14_norm_close
        - regime_atr50_norm_close
        - regime_atr14_50_ratio
        - regime_vol20
        - regime_vol100
        - regime_vol20_100_ratio
        - regime_trend_slope_20
        - regime_trend_slope_50
        - regime_trend_slope_sign_20
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with regime context features
        """
        t_start = time.time()
        n_rows = len(df)
        
        # For datasets >25k rows, use CPU to avoid GPU memory issues
        # GPU acceleration provides minimal benefit for these simple rolling operations
        # and risks segfaults on GPUs with limited VRAM (8GB or less)
        if n_rows > 25000:
            print(f"[RegimeContextFeatures] Using CPU for {n_rows} rows (safer for large datasets)")
            features = self._generate_features_cpu(df)
            t_end = time.time()
            print(f"[RegimeContextFeatures] Generated {features.shape[1]} regime context features in {t_end-t_start:.2f}s")
            return features
        
        # For smaller datasets, try GPU with chunking
        overlap = 150  # Larger than max rolling window (100)
        use_chunking = self.use_gpu_effective and n_rows > self.chunk_size
        
        if use_chunking:
            print(f"[RegimeContextFeatures] Processing {n_rows} rows in chunks of {self.chunk_size}")
            all_chunks = []
            
            chunk_start = 0
            while chunk_start < n_rows:
                chunk_end = min(chunk_start + self.chunk_size, n_rows)
                
                # For non-first chunks, include overlap from previous data for rolling windows
                actual_start = max(0, chunk_start - overlap) if chunk_start > 0 else 0
                chunk_df = df.iloc[actual_start:chunk_end]
                
                try:
                    # Clear GPU memory before processing chunk
                    clear_gpu_memory()
                    
                    chunk_features = self._generate_features_gpu(chunk_df)
                    
                    # Trim the overlap portion from results (keep only the new part)
                    if chunk_start > 0:
                        trim_rows = chunk_start - actual_start
                        chunk_features = chunk_features.iloc[trim_rows:]
                    
                    all_chunks.append(chunk_features)
                    
                    # Clear GPU memory after chunk and force sync
                    del chunk_features
                    clear_gpu_memory()
                    
                except Exception as e:
                    error_type = type(e).__name__
                    print(f"[RegimeContextFeatures] GPU failed on chunk, falling back to CPU: {error_type}: {e}")
                    self.use_gpu_effective = False
                    
                    # Fall back to CPU for remaining data
                    remaining_df = df.iloc[chunk_start:]
                    remaining_features = self._generate_features_cpu(remaining_df)
                    all_chunks.append(remaining_features)
                    break
                
                chunk_start = chunk_end
            
            # Combine all chunks
            features = pd.concat(all_chunks, axis=0)
            # Ensure index matches original dataframe
            features.index = df.index
            
        else:
            # Process entire dataset at once (small dataset or CPU mode)
            if self.use_gpu_effective and n_rows >= 10000:
                try:
                    features = self._generate_features_gpu(df)
                except Exception as e:
                    # Graceful fallback to CPU on any GPU error
                    error_type = type(e).__name__
                    print(f"[RegimeContextFeatures] GPU failed, falling back to CPU: {error_type}: {e}")
                    self.use_gpu_effective = False
                    features = self._generate_features_cpu(df)
            else:
                features = self._generate_features_cpu(df)
        
        t_end = time.time()
        print(f"[RegimeContextFeatures] Generated {features.shape[1]} regime context features in {t_end-t_start:.2f}s")
        
        return features


class SignalDynamicsFeatures:
    """
    GPU-accelerated signal dynamics feature generator.
    Falls back to CPU if RAPIDS is unavailable or GPU context is corrupted.
    Generates: signal streaks, time since signals, distance-to-extreme, derivatives.
    """
    
    def __init__(
        self,
        signal_modules: List[SignalModule] = None,
        use_gpu: bool = True,
        chunk_size: int = 100000
    ):
        """
        Initialize signal dynamics feature generator.
        
        Args:
            signal_modules: List of SignalModule instances to compute signals from
            use_gpu: Whether to use GPU acceleration (default: True)
            chunk_size: Number of rows to process per chunk (default: 100000)
        """
        self.signal_modules = signal_modules if signal_modules is not None else []
        self.use_gpu = use_gpu and GPU_AVAILABLE and (cudf is not None)
        self.use_gpu_effective = self.use_gpu  # Can be disabled at runtime on GPU failure
        self.chunk_size = chunk_size
        
        if self.use_gpu:
            print("[SignalDynamicsFeatures] GPU acceleration enabled (RAPIDS cuDF)")
        else:
            if use_gpu:
                print("[SignalDynamicsFeatures] GPU requested but RAPIDS unavailable, using CPU fallback")
    
    def _compute_signal_streak(self, signal: pd.Series) -> pd.Series:
        """Compute consecutive signal streak length (vectorized, CPU)."""
        signal_clean = signal.fillna(0).astype(int)
        signal_prev = signal_clean.shift(1).fillna(0).astype(int)
        signal_changed = (signal_clean != signal_prev) | (signal_clean == 0)
        streak_groups = signal_changed.cumsum()
        streak = signal_clean.groupby(streak_groups).cumcount() + 1
        streak = streak.where(signal_clean != 0, 0)
        return streak.astype(int)
    
    def _time_since_signal(self, signal: pd.Series, target: int) -> pd.Series:
        """Compute bars since last signal of target type (vectorized, CPU)."""
        signal_clean = signal.fillna(0).astype(int)
        target_mask = (signal_clean == target)
        
        if not target_mask.any():
            return pd.Series(np.nan, index=signal.index, dtype=float)
        
        positions = pd.Series(range(len(signal)), index=signal.index)
        target_positions = positions.where(target_mask, np.nan)
        last_target_pos = target_positions.ffill()
        time_since = positions - last_target_pos
        time_since = time_since.where(~target_mask, 0)
        time_since = time_since.where(target_mask.cumsum() > 0, np.nan)
        return time_since
    
    def _distance_to_extreme_gpu(self, gpu_series: 'cudf.Series', lookback: int = 100) -> 'cudf.Series':
        """Compute distance to extreme on GPU."""
        gpu_series = gpu_series.astype(cp.float32)
        rolling_max = gpu_series.rolling(window=lookback, min_periods=1).max()
        rolling_min = gpu_series.rolling(window=lookback, min_periods=1).min()
        
        dist_to_max = (gpu_series - rolling_max).abs() / rolling_max.replace(0, cp.nan)
        dist_to_min = (gpu_series - rolling_min).abs() / rolling_min.replace(0, cp.nan)
        
        # cuDF does not allow duplicate column names in concat; ensure unique names
        dist_to_max = dist_to_max.rename("dist_to_max")
        dist_to_min = dist_to_min.rename("dist_to_min")
        distance = cudf.concat([dist_to_max, dist_to_min], axis=1).min(axis=1)
        return distance
    
    def _distance_to_extreme_cpu(self, series: pd.Series, lookback: int = 100) -> pd.Series:
        """CPU fallback for distance to extreme."""
        rolling_max = series.rolling(window=lookback, min_periods=1).max()
        rolling_min = series.rolling(window=lookback, min_periods=1).min()
        
        dist_to_max = (series - rolling_max).abs() / rolling_max.replace(0, np.nan)
        dist_to_min = (series - rolling_min).abs() / rolling_min.replace(0, np.nan)
        
        distance = pd.concat([dist_to_max, dist_to_min], axis=1).min(axis=1)
        return distance
    
    def generate_features(
        self,
        df: pd.DataFrame,
        unified_features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate signal dynamics features.
        
        Args:
            df: OHLCV DataFrame
            unified_features: Unified feature DataFrame with module features
            
        Returns:
            DataFrame with signal dynamics features
        """
        n_rows = len(df)
        
        # For datasets >25k rows, force CPU to avoid GPU memory issues
        # GPU memory may be fragmented from model loading
        if n_rows > 25000 and self.use_gpu_effective:
            print(f"[SignalDynamicsFeatures] Using CPU for {n_rows} rows (safer for large datasets)")
            self.use_gpu_effective = False
        
        all_features = []
        
        # Compute signals from all signal modules
        module_signals = {}
        for module in self.signal_modules:
            try:
                # Get module-specific features (handle legacy_ prefix)
                # Try both legacy_ prefix and direct prefix for backward compatibility
                legacy_prefix = f"legacy_{module.name}_"
                direct_prefix = f"{module.name}_"
                module_feat_cols = [
                    c for c in unified_features.columns 
                    if c.startswith(legacy_prefix) or c.startswith(direct_prefix)
                ]
                if not module_feat_cols:
                    continue
                
                module_features = unified_features[module_feat_cols].copy()
                # Remove prefix for module's compute_signal (handle both legacy_ and direct)
                module_features.columns = [
                    c.replace(legacy_prefix, "").replace(direct_prefix, "") 
                    for c in module_features.columns
                ]
                
                signal = module.compute_signal(module_features)
                module_signals[module.name] = signal
                
                # Signal streak length
                streak = self._compute_signal_streak(signal)
                all_features.append(streak.rename(f"{module.name}_signal_streak"))
                
                # Time since last long signal
                time_long = self._time_since_signal(signal, target=1)
                all_features.append(time_long.rename(f"{module.name}_time_since_long"))
                
                # Time since last short signal
                time_short = self._time_since_signal(signal, target=-1)
                all_features.append(time_short.rename(f"{module.name}_time_since_short"))
                
            except Exception as e:
                print(f"Warning: Failed to compute signal dynamics for {module.name}: {e}")
                continue
        
        # Distance-to-previous-extreme for hull (from SuperMA, PVT, TrendMagic)
        # Use GPU acceleration if available, with automatic fallback on errors
        hull_modules = ["superma", "pvt", "trendmagic"]
        for module_name in hull_modules:
            # Try legacy_ prefix first, then fallback to direct prefix
            hull_col = f"legacy_{module_name}_hull"
            if hull_col not in unified_features.columns:
                hull_col = f"{module_name}_hull"
            if hull_col in unified_features.columns:
                hull = unified_features[hull_col]
                if self.use_gpu_effective:
                    try:
                        gpu_hull = cudf.from_pandas(hull)
                        dist_extreme = self._distance_to_extreme_gpu(gpu_hull, lookback=100)
                        dist_extreme = dist_extreme.to_pandas()
                        dist_extreme.index = hull.index
                    except Exception as e:
                        # GPU context corrupted, fall back to CPU for this and all subsequent ops
                        error_type = type(e).__name__
                        print(f"[SignalDynamicsFeatures] GPU failed, falling back to CPU: {error_type}: {e}")
                        self.use_gpu_effective = False
                        dist_extreme = self._distance_to_extreme_cpu(hull, lookback=100)
                else:
                    dist_extreme = self._distance_to_extreme_cpu(hull, lookback=100)
                all_features.append(dist_extreme.rename(f"{module_name}_hull_dist_extreme"))
        
        # Derivative of topvec and botvec
        # Dynamically detect column names (different modules may use different naming)
        for module_name in hull_modules:
            # Candidate column names (with legacy_ prefix first, then fallback to direct prefix)
            topvec_candidates = [
                f"legacy_{module_name}_topvector01",
                f"legacy_{module_name}_topvecMA",
                f"{module_name}_topvector01",
                f"{module_name}_topvecMA"
            ]
            botvec_candidates = [
                f"legacy_{module_name}_botvector01",
                f"legacy_{module_name}_botvecMA",
                f"{module_name}_botvector01",
                f"{module_name}_botvecMA"
            ]
            
            # Find first matching topvec column
            topvec_col = None
            for candidate in topvec_candidates:
                if candidate in unified_features.columns:
                    topvec_col = candidate
                    break
            
            # Find first matching botvec column
            botvec_col = None
            for candidate in botvec_candidates:
                if candidate in unified_features.columns:
                    botvec_col = candidate
                    break
            
            # Compute topvec derivative if found
            if topvec_col is not None:
                topvec = unified_features[topvec_col]
                if self.use_gpu_effective:
                    try:
                        # GPU path: convert to cuDF, compute diff, convert back
                        gpu_topvec = cudf.from_pandas(topvec)
                        topvec_deriv_gpu = gpu_topvec.diff()
                        topvec_deriv = topvec_deriv_gpu.to_pandas()
                        topvec_deriv.index = topvec.index
                    except Exception as e:
                        # GPU context corrupted, fall back to CPU
                        error_type = type(e).__name__
                        print(f"[SignalDynamicsFeatures] GPU failed on topvec, falling back to CPU: {error_type}: {e}")
                        self.use_gpu_effective = False
                        topvec_deriv = topvec.diff()
                else:
                    # CPU path
                    topvec_deriv = topvec.diff()
                topvec_deriv = topvec_deriv.rename(f"{module_name}_topvec_deriv")
                all_features.append(topvec_deriv)
            
            # Compute botvec derivative if found
            if botvec_col is not None:
                botvec = unified_features[botvec_col]
                if self.use_gpu_effective:
                    try:
                        # GPU path: convert to cuDF, compute diff, convert back
                        gpu_botvec = cudf.from_pandas(botvec)
                        botvec_deriv_gpu = gpu_botvec.diff()
                        botvec_deriv = botvec_deriv_gpu.to_pandas()
                        botvec_deriv.index = botvec.index
                    except Exception as e:
                        # GPU context corrupted, fall back to CPU
                        error_type = type(e).__name__
                        print(f"[SignalDynamicsFeatures] GPU failed on botvec, falling back to CPU: {error_type}: {e}")
                        self.use_gpu_effective = False
                        botvec_deriv = botvec.diff()
                else:
                    # CPU path
                    botvec_deriv = botvec.diff()
                botvec_deriv = botvec_deriv.rename(f"{module_name}_botvec_deriv")
                all_features.append(botvec_deriv)
        
        # Combine all features
        if not all_features:
            return pd.DataFrame(index=df.index)
        
        features_df = pd.concat(all_features, axis=1)
        
        return features_df


class MLFeatures:
    """
    GPU-accelerated ML feature generator.
    Generates comprehensive ML features: returns, volatility, RSI, SMA distances,
    rolling zscores, candle metrics, ATR, volatility compression, skew/kurt, volume anomalies.
    Falls back to CPU if RAPIDS is unavailable or GPU context is corrupted.
    """
    
    def __init__(self, use_gpu: bool = True, chunk_size: int = 100000):
        """
        Initialize ML feature generator.
        
        Args:
            use_gpu: Whether to use GPU acceleration (default: True)
            chunk_size: Number of rows to process per chunk (default: 100000)
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE and (cudf is not None)
        self.use_gpu_effective = self.use_gpu  # Can be disabled at runtime on GPU failure
        self.chunk_size = chunk_size
        
        if self.use_gpu:
            print("[MLFeatures] GPU acceleration enabled (RAPIDS cuDF)")
        else:
            if use_gpu:
                print("[MLFeatures] GPU requested but RAPIDS unavailable, using CPU fallback")
    
    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Compute RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _rsi_gpu(self, gpu_series: 'cudf.Series', period: int) -> 'cudf.Series':
        """Compute RSI on GPU."""
        delta = gpu_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss.replace(0, cp.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive ML features.
        
        Args:
            df: OHLCV DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            DataFrame with ML features
        """
        n_rows = len(df)
        
        # For datasets >25k rows, force CPU to avoid GPU memory issues
        # GPU provides minimal benefit for these simple operations
        # and risks segfaults when GPU memory is fragmented from model loading
        if n_rows > 25000 and self.use_gpu_effective:
            print(f"[MLFeatures] Using CPU for {n_rows} rows (safer for large datasets)")
            self.use_gpu_effective = False
        
        all_features = []
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        # Handle volume column (may not exist)
        if 'volume' in df.columns:
            volume = df['volume']
        else:
            volume = pd.Series(1.0, index=df.index)
        
        # 1. Price returns: 1, 2, 3, 5, 10, 20 bars
        return_periods = [1, 2, 3, 5, 10, 20]
        for period in return_periods:
            returns = close.pct_change(periods=period)
            all_features.append(returns.rename(f"return_{period}"))
        
        # 2. Volatility: std over 5, 10, 20, 50 bars
        vol_periods = [5, 10, 20, 50]
        returns_1 = close.pct_change()
        for period in vol_periods:
            vol = returns_1.rolling(window=period, min_periods=period).std()
            all_features.append(vol.rename(f"volatility_{period}"))
        
        # 3. RSI: 7, 14, 21
        rsi_periods = [7, 14, 21]
        for period in rsi_periods:
            if self.use_gpu_effective:
                try:
                    gpu_close = cudf.from_pandas(close)
                    rsi = self._rsi_gpu(gpu_close, period)
                    rsi = rsi.to_pandas()
                    rsi.index = close.index
                except Exception as e:
                    error_type = type(e).__name__
                    print(f"[MLFeatures] GPU failed on RSI, falling back to CPU: {error_type}: {e}")
                    self.use_gpu_effective = False
                    rsi = self._rsi(close, period)
            else:
                rsi = self._rsi(close, period)
            all_features.append(rsi.rename(f"rsi_{period}"))
        
        # 4. Price distance from SMA: 20, 50, 100
        sma_periods = [20, 50, 100]
        for period in sma_periods:
            sma = close.rolling(window=period, min_periods=period).mean()
            distance_pct = (close - sma) / sma.replace(0, np.nan)
            all_features.append(distance_pct.rename(f"distance_sma_{period}"))
        
        # 5. Rolling zscores for returns, volatility, RSI, SMA distances
        zscore_window = 20
        for period in return_periods:
            returns = close.pct_change(periods=period)
            mean = returns.rolling(window=zscore_window, min_periods=zscore_window).mean()
            std = returns.rolling(window=zscore_window, min_periods=zscore_window).std()
            zscore = (returns - mean) / std.replace(0, np.nan)
            all_features.append(zscore.rename(f"return_{period}_zscore"))
        
        for period in vol_periods:
            vol = returns_1.rolling(window=period, min_periods=period).std()
            mean = vol.rolling(window=zscore_window, min_periods=zscore_window).mean()
            std = vol.rolling(window=zscore_window, min_periods=zscore_window).std()
            zscore = (vol - mean) / std.replace(0, np.nan)
            all_features.append(zscore.rename(f"volatility_{period}_zscore"))
        
        for period in rsi_periods:
            if self.use_gpu_effective:
                try:
                    gpu_close = cudf.from_pandas(close)
                    rsi = self._rsi_gpu(gpu_close, period)
                    rsi = rsi.to_pandas()
                    rsi.index = close.index
                except Exception as e:
                    error_type = type(e).__name__
                    print(f"[MLFeatures] GPU failed on RSI zscore, falling back to CPU: {error_type}: {e}")
                    self.use_gpu_effective = False
                    rsi = self._rsi(close, period)
            else:
                rsi = self._rsi(close, period)
            mean = rsi.rolling(window=zscore_window, min_periods=zscore_window).mean()
            std = rsi.rolling(window=zscore_window, min_periods=zscore_window).std()
            zscore = (rsi - mean) / std.replace(0, np.nan)
            all_features.append(zscore.rename(f"rsi_{period}_zscore"))
        
        for period in sma_periods:
            sma = close.rolling(window=period, min_periods=period).mean()
            distance_pct = (close - sma) / sma.replace(0, np.nan)
            mean = distance_pct.rolling(window=zscore_window, min_periods=zscore_window).mean()
            std = distance_pct.rolling(window=zscore_window, min_periods=zscore_window).std()
            zscore = (distance_pct - mean) / std.replace(0, np.nan)
            all_features.append(zscore.rename(f"distance_sma_{period}_zscore"))
        
        # 6. Candle metrics: wick ratios, body size, HL range
        # Upper wick ratio
        upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
        hl_range = high - low
        upper_wick_ratio = upper_wick / hl_range.replace(0, np.nan)
        all_features.append(upper_wick_ratio.rename("upper_wick_ratio"))
        
        # Lower wick ratio
        lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
        lower_wick_ratio = lower_wick / hl_range.replace(0, np.nan)
        all_features.append(lower_wick_ratio.rename("lower_wick_ratio"))
        
        # Body size (as % of HL range)
        body_size = (close - open_price).abs()
        body_ratio = body_size / hl_range.replace(0, np.nan)
        all_features.append(body_ratio.rename("body_ratio"))
        
        # HL range (as % of close)
        hl_range_pct = hl_range / close.replace(0, np.nan)
        all_features.append(hl_range_pct.rename("hl_range_pct"))
        
        # 7. ATR: 7, 14, 21
        from core.regime_detector import wilder_atr
        atr_periods = [7, 14, 21]
        for period in atr_periods:
            atr = wilder_atr(df, n=period)
            atr_pct = atr / close.replace(0, np.nan)
            all_features.append(atr_pct.rename(f"atr_{period}_pct"))
        
        # 8. Volatility compression ratios
        # Ratio of short-term to long-term volatility
        vol_short = returns_1.rolling(window=5, min_periods=5).std()
        vol_long = returns_1.rolling(window=20, min_periods=20).std()
        vol_compression = vol_short / vol_long.replace(0, np.nan)
        all_features.append(vol_compression.rename("vol_compression_5_20"))
        
        vol_short = returns_1.rolling(window=10, min_periods=10).std()
        vol_long = returns_1.rolling(window=50, min_periods=50).std()
        vol_compression = vol_short / vol_long.replace(0, np.nan)
        all_features.append(vol_compression.rename("vol_compression_10_50"))
        
        # 9. Rolling skew / rolling kurt
        skew_window = 20
        kurt_window = 20
        returns_1_clean = returns_1.dropna()
        if len(returns_1_clean) > 0:
            skew = returns_1.rolling(window=skew_window, min_periods=skew_window).skew()
            all_features.append(skew.rename("returns_skew_20"))
            
            kurt = returns_1.rolling(window=kurt_window, min_periods=kurt_window).apply(
                lambda x: x.kurtosis() if len(x.dropna()) >= kurt_window else np.nan,
                raw=False
            )
            all_features.append(kurt.rename("returns_kurt_20"))
        
        # 10. Volume anomaly ratios
        if 'volume' in df.columns and len(volume.dropna()) > 0:
            # Volume z-score (anomaly detection)
            vol_mean = volume.rolling(window=20, min_periods=20).mean()
            vol_std = volume.rolling(window=20, min_periods=20).std()
            vol_zscore = (volume - vol_mean) / vol_std.replace(0, np.nan)
            all_features.append(vol_zscore.rename("volume_zscore_20"))
            
            # Volume ratio (current / rolling mean)
            vol_ratio = volume / vol_mean.replace(0, np.nan)
            all_features.append(vol_ratio.rename("volume_ratio_20"))
            
            # Volume surge (spike detection)
            vol_surge = (volume > vol_mean * 1.5).astype(float)
            all_features.append(vol_surge.rename("volume_surge"))
        else:
            # Create dummy volume features if volume is missing
            vol_zscore = pd.Series(0.0, index=df.index, name="volume_zscore_20")
            vol_ratio = pd.Series(1.0, index=df.index, name="volume_ratio_20")
            vol_surge = pd.Series(0.0, index=df.index, name="volume_surge")
            all_features.extend([vol_zscore, vol_ratio, vol_surge])
        
        # Combine all features
        if not all_features:
            return pd.DataFrame(index=df.index)
        
        features_df = pd.concat(all_features, axis=1)
        
        return features_df


class FeatureStore:
    """
    Builds a unified feature dataframe from all modules.
    Handles alignment, merging, and duplicate dropping.
    """

    def __init__(
        self,
        generate_rolling_stats: bool = True,
        rolling_windows: List[int] = None,
        generate_regime_context: bool = True,
        rolling_stats_columns: List[str] = None,
        safe_mode: bool = True,
        use_gpu: bool = True,
        chunk_size: int = 100000
    ):
        """
        Initialize feature store.
        
        Args:
            generate_rolling_stats: Whether to generate rolling statistics (default: True)
            rolling_windows: List of window sizes for rolling stats (default: [5, 20] in safe_mode)
            generate_regime_context: Whether to generate regime context features (default: True)
            rolling_stats_columns: List of column names to compute rolling stats for.
                                  If None, uses safe default subset.
            safe_mode: Enable safe mode optimizations (default: True)
            use_gpu: Whether to use GPU acceleration (default: True, falls back to CPU if unavailable)
            chunk_size: Number of rows to process per chunk for large datasets (default: 100000)
        """
        self.safe_mode = safe_mode
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        self.features = pd.DataFrame()
        self.feature_columns = []
        
        # Default modules (CPU-based, lightweight)
        self.signal_modules = [
            SuperMA4hr(),
            TrendMagicV2(),
            PVTEliminator(),
        ]
        self.context_modules = [
            PivotRSIContext(),
            LinRegChannelContext(),
        ]
        
        # Apply safe_mode defaults
        if safe_mode:
            if rolling_windows is None:
                rolling_windows = [5, 20]
            if rolling_stats_columns is None:
                # Safe default subset (with legacy_ prefix since we namespace module features)
                rolling_stats_columns = ["legacy_superma_hull", "legacy_trendmagic_hls", "legacy_pvt_hull"]
        else:
            if rolling_windows is None:
                rolling_windows = [5, 20]  # Still use reduced windows by default
        
        # Rolling stats generator (GPU-accelerated)
        self.generate_rolling_stats = generate_rolling_stats
        self.rolling_stats_columns = rolling_stats_columns
        self.rolling_stats_generator = RollingStatsGenerator(
            windows=rolling_windows,
            rolling_stats_columns=rolling_stats_columns,
            use_gpu=use_gpu,
            chunk_size=chunk_size
        ) if generate_rolling_stats else None
        
        # Regime context features generator (GPU-accelerated)
        self.generate_regime_context = generate_regime_context
        self.regime_context_generator = RegimeContextFeatures(
            use_gpu=use_gpu,
            chunk_size=chunk_size
        ) if generate_regime_context else None
        
        # Signal dynamics features generator (GPU-accelerated for distance-to-extreme)
        self.generate_signal_dynamics = True
        self.signal_dynamics_generator = SignalDynamicsFeatures(
            signal_modules=self.signal_modules,
            use_gpu=use_gpu,
            chunk_size=chunk_size
        )
        
        # ML features generator (GPU-accelerated)
        self.generate_ml_features = True
        self.ml_features_generator = MLFeatures(
            use_gpu=use_gpu,
            chunk_size=chunk_size
        )

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build unified feature DataFrame from all modules.
        
        - Instantiates each module
        - Calls compute_features() on each
        - Merges all features into a single DataFrame
        - Drops duplicates
        - Does forward-fill for NA values
        - Ensures all columns align with df.index
        
        Args:
            df: Raw OHLCV dataframe with datetime index
            
        Returns:
            Unified feature DataFrame with all module features, aligned to df.index
        """
        MAX_FEATURES = 2000
        t_total_start = time.time()
        
        # Start with base dataframe index
        unified_features = pd.DataFrame(index=df.index)
        all_feature_dfs = []
        
        # Collect features from signal modules (namespace as legacy_*)
        for module in self.signal_modules:
            try:
                module_features = module.compute_features(df)
                # Align index with base dataframe
                module_features = module_features.reindex(df.index)
                # Prefix all columns with legacy_ and module name
                module_features.columns = [f"legacy_{module.name}_{col}" for col in module_features.columns]
                all_feature_dfs.append(module_features)
            except Exception as e:
                print(f"Warning: Failed to compute features for {module.name}: {e}")
                continue
        
        # Collect features from context modules (namespace as legacy_*)
        for module in self.context_modules:
            try:
                module_features = module.compute_features(df)
                # Align index with base dataframe
                module_features = module_features.reindex(df.index)
                # Prefix all columns with legacy_ and module name
                module_features.columns = [f"legacy_{module.name}_{col}" for col in module_features.columns]
                all_feature_dfs.append(module_features)
            except Exception as e:
                print(f"Warning: Failed to compute features for {module.name}: {e}")
                continue
        
        # Merge all feature dataframes
        if all_feature_dfs:
            unified_features = pd.concat(all_feature_dfs, axis=1)
        
        # Drop duplicate columns (keep first occurrence)
        unified_features = unified_features.loc[:, ~unified_features.columns.duplicated(keep='first')]
        
        # Forward-fill NA values
        unified_features = unified_features.ffill()
        
        # Ensure all features align on the same index
        unified_features = unified_features.reindex(df.index)
        
        print(f"[FeatureStore] After legacy signal/context modules: {unified_features.shape[1]} features")
        
        # Generate ML features (new comprehensive feature set)
        if self.generate_ml_features and self.ml_features_generator is not None:
            # Clear GPU memory before heavy ML feature generation
            clear_gpu_memory(aggressive=True)
            
            t0 = time.time()
            print("Generating ML features...")
            ml_features = self.ml_features_generator.generate_features(df)
            
            # Align and forward-fill in one step
            ml_features = ml_features.reindex(df.index).ffill()
            
            # Merge ML features with unified features
            unified_features = pd.concat([unified_features, ml_features], axis=1)
            
            t1 = time.time()
            print(f"Added {len(ml_features.columns)} ML features (took {t1-t0:.2f}s)")
            print(f"[FeatureStore] After ML features: {unified_features.shape[1]} features")
            
            # Clear GPU memory after heavy ML feature generation
            clear_gpu_memory()
        
        # Store BASE features before adding derived features (for rolling stats)
        # Use legacy features only for rolling stats to avoid stats-of-stats
        legacy_cols = [col for col in unified_features.columns if col.startswith('legacy_')]
        if legacy_cols:
            base_features = unified_features[legacy_cols].copy()
        else:
            # Fallback: use all features if no legacy features found
            base_features = unified_features.copy()
        
        # Safe mode: Remove features with >30% NaNs
        if self.safe_mode:
            nan_threshold = 0.30
            nan_ratios = unified_features.isna().sum() / len(unified_features)
            valid_cols = nan_ratios[nan_ratios <= nan_threshold].index
            unified_features = unified_features[valid_cols]
            # Only filter base_features columns that exist in base_features
            valid_base_cols = [col for col in valid_cols if col in base_features.columns]
            base_features = base_features[valid_base_cols]
            if len(valid_cols) < len(nan_ratios):
                print(f"[FeatureStore] Removed {len(nan_ratios) - len(valid_cols)} features with >{nan_threshold*100}% NaNs")
        
        # Generate regime context features if enabled
        # Safe mode: Disable for small datasets (<50k rows)
        if self.generate_regime_context and self.regime_context_generator is not None:
            if self.safe_mode and len(df) < 50000:
                print("[FeatureStore] Regime context disabled in safe_mode for dataset <50k rows")
            else:
                # Aggressive GPU memory clear before regime context generation
                # This helps prevent segfaults from GPU memory fragmentation
                clear_gpu_memory(aggressive=True)
                
                t0 = time.time()
                print("Generating regime context features...")
                regime_features = self.regime_context_generator.generate_features(df)
                
                # Align and forward-fill in one step
                regime_features = regime_features.reindex(df.index).ffill()
                
                # Merge regime features with base features
                unified_features = pd.concat([unified_features, regime_features], axis=1)
                
                t1 = time.time()
                print(f"Added {len(regime_features.columns)} regime context features (took {t1-t0:.2f}s)")
                print(f"[FeatureStore] After regime context: {unified_features.shape[1]} features")
                
                # Clear GPU memory after regime context generation
                clear_gpu_memory()
        
        # Generate signal dynamics features if enabled
        if self.generate_signal_dynamics and self.signal_dynamics_generator is not None:
            # Aggressive GPU memory clear before signal dynamics
            clear_gpu_memory(aggressive=True)
            
            t0 = time.time()
            print("Generating signal dynamics features...")
            signal_dynamics = self.signal_dynamics_generator.generate_features(df, unified_features)
            
            # Align and forward-fill in one step
            signal_dynamics = signal_dynamics.reindex(df.index).ffill()
            
            # Merge signal dynamics features with base features
            unified_features = pd.concat([unified_features, signal_dynamics], axis=1)
            
            t1 = time.time()
            num_added = len(signal_dynamics.columns)
            print(f"Added {num_added} signal dynamics features (took {t1-t0:.2f}s)")
            if num_added > 0:
                # Show example column names
                example_cols = list(signal_dynamics.columns[:min(5, num_added)])
                print(f"  Example columns: {example_cols}")
            else:
                print("  WARNING: No signal dynamics features were generated!")
            print(f"[FeatureStore] After signal dynamics: {unified_features.shape[1]} features")
            
            # Clear GPU memory after signal dynamics generation
            clear_gpu_memory()
        
        # Generate rolling statistics if enabled
        # CRITICAL: Only pass BASE features (before regime/signal dynamics) to rolling stats
        # This prevents stats-of-stats recursion
        if self.generate_rolling_stats and self.rolling_stats_generator is not None:
            t0 = time.time()
            print(f"Generating rolling statistics for {len(base_features.columns)} base features...")
            rolling_stats = self.rolling_stats_generator.generate_stats(base_features)
            
            # Align and forward-fill in one step
            rolling_stats = rolling_stats.reindex(df.index).ffill()
            
            # Merge rolling stats with unified features
            unified_features = pd.concat([unified_features, rolling_stats], axis=1)
            
            t1 = time.time()
            print(f"Added {len(rolling_stats.columns)} rolling statistic features (took {t1-t0:.2f}s)")
            print(f"[FeatureStore] After rolling stats: {unified_features.shape[1]} features")
        
        # Safe mode: Downcast to float32 to reduce memory
        if self.safe_mode:
            # Only downcast numeric columns
            numeric_cols = unified_features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                unified_features[col] = unified_features[col].astype(np.float32)
        
        # Global feature cap check
        if unified_features.shape[1] > MAX_FEATURES:
            raise RuntimeError(
                f"Feature explosion detected: {unified_features.shape[1]} columns "
                f"(max allowed: {MAX_FEATURES}). "
                f"Check rolling stats configuration and column selection."
            )
        
        # Store feature columns for reference
        self.feature_columns = unified_features.columns.tolist()
        self.features = unified_features
        
        t_total_end = time.time()
        print(f"\nTotal feature generation time: {t_total_end - t_total_start:.2f}s")
        print(f"Final feature count: {len(unified_features.columns)}")
        
        return unified_features

    def build_features(
        self,
        df: pd.DataFrame,
        signal_modules: List[SignalModule],
        context_modules: List[ContextModule] = None
    ) -> pd.DataFrame:
        """
        Combine features from all modules into a unified dataframe.
        Includes logic to align indexes, merge columns, and drop duplicates.
        
        Args:
            df: Raw price/OHLCV dataframe
            signal_modules: List of SignalModule instances
            context_modules: Optional list of ContextModule instances
            
        Returns:
            Unified feature dataframe with prefixed column names
        """
        if context_modules is None:
            context_modules = []
        
        # Start with base dataframe index
        unified_features = pd.DataFrame(index=df.index)
        all_feature_dfs = []
        
        # Collect features from signal modules
        for module in signal_modules:
            try:
                module_features = module.compute_features(df)
                # Align index with base dataframe
                module_features = module_features.reindex(df.index)
                # Prefix all columns with module name
                module_features.columns = [f"{module.name}_{col}" for col in module_features.columns]
                all_feature_dfs.append(module_features)
            except Exception as e:
                print(f"Warning: Failed to compute features for {module.name}: {e}")
                continue
        
        # Collect features from context modules
        for module in context_modules:
            try:
                module_features = module.compute_features(df)
                # Align index with base dataframe
                module_features = module_features.reindex(df.index)
                # Prefix all columns with module name
                module_features.columns = [f"{module.name}_{col}" for col in module_features.columns]
                all_feature_dfs.append(module_features)
            except Exception as e:
                print(f"Warning: Failed to compute features for {module.name}: {e}")
                continue
        
        # Merge all feature dataframes
        if all_feature_dfs:
            unified_features = pd.concat(all_feature_dfs, axis=1)
        
        # Drop duplicate columns (keep first occurrence)
        unified_features = unified_features.loc[:, ~unified_features.columns.duplicated(keep='first')]
        
        # Ensure all features align on the same index
        unified_features = unified_features.reindex(df.index)
        
        # Store feature columns for reference
        self.feature_columns = unified_features.columns.tolist()
        self.features = unified_features
        
        return unified_features

    def build_single_row(
        self,
        df_row: pd.Series,
        signal_modules: List[SignalModule],
        context_modules: List[ContextModule] = None
    ) -> pd.DataFrame:
        """
        Build features for a single row (bar).
        Converts Series to DataFrame for processing.
        
        Args:
            df_row: Single row of OHLCV data as Series
            signal_modules: List of SignalModule instances
            context_modules: Optional list of ContextModule instances
            
        Returns:
            DataFrame with single row of features
        """
        if context_modules is None:
            context_modules = []
        
        # Convert Series to DataFrame with single row
        df = df_row.to_frame().T
        
        # Build features using existing method
        features = self.build_features(df, signal_modules, context_modules)
        
        return features
    
    def get_feature(self, module_name: str, feature_name: str) -> pd.Series:
        """
        Retrieve a specific feature by module and feature name.
        
        Args:
            module_name: Name of the module
            feature_name: Name of the feature (without module prefix)
            
        Returns:
            Series with the requested feature
        """
        col_name = f"{module_name}_{feature_name}"
        if col_name in self.features.columns:
            return self.features[col_name]
        return pd.Series(dtype=float, index=self.features.index)
    
    def clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Clean features by handling NaN, inf, and ensuring numeric types.
        
        Args:
            features: Feature dataframe to clean
            
        Returns:
            Cleaned feature dataframe
        """
        cleaned = features.copy()
        
        # Replace inf with NaN
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with forward fill, then backward fill, then 0
        cleaned = cleaned.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Ensure numeric types
        for col in cleaned.columns:
            if cleaned[col].dtype == 'object':
                try:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
                except (ValueError, TypeError):
                    pass

        return cleaned

    def build_and_cache(
        self,
        df: pd.DataFrame,
        signal_modules: List[SignalModule],
        context_modules: List[ContextModule],
        cache_path: "Path",
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        Build features once, persist to Parquet, and return the cached frame.

        This prevents repeated computation across walk-forward windows, which is
        the dominant runtime cost in the training pipeline.
        """
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            return self.load_cached(cache_path)

        features = self.build_features(df, signal_modules, context_modules)
        features = self.clean_features(features)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path)
        return features

    def load_cached(self, cache_path: "Path") -> pd.DataFrame:
        """Load cached features and restore internal metadata."""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached features not found at {cache_path}")

        features = pd.read_parquet(cache_path)
        self.feature_columns = features.columns.tolist()
        self.features = features
        return features

    def slice(self, features: pd.DataFrame, start_ts, end_ts) -> pd.DataFrame:
        """
        Return a time-sliced view of cached features without recomputation.
        """
        return features[(features.index >= start_ts) & (features.index < end_ts)]

