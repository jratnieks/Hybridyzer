# core/feature_store.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List
from modules.base import SignalModule, ContextModule
from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext


class RollingStatsGenerator:
    """
    Generates rolling statistics for base features.
    Produces rolling mean, std, max, min, slope, and zscore for specified windows.
    """
    
    def __init__(self, windows: List[int] = None):
        """
        Initialize rolling stats generator.
        
        Args:
            windows: List of window sizes (default: [3, 5, 10, 20])
        """
        if windows is None:
            windows = [3, 5, 10, 20]
        self.windows = windows
    
    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """
        Compute rolling linear regression slope.
        
        Args:
            series: Input series
            window: Window size
            
        Returns:
            Series of slopes
        """
        slopes = pd.Series(np.nan, index=series.index, dtype=float)
        
        for i in range(window - 1, len(series)):
            window_data = series.iloc[i - window + 1:i + 1]
            if window_data.notna().sum() >= 2:
                x = np.arange(len(window_data))
                y = window_data.values
                valid_mask = ~np.isnan(y)
                if valid_mask.sum() >= 2:
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    slope = np.polyfit(x_valid, y_valid, 1)[0]
                    slopes.iloc[i] = slope
        
        return slopes
    
    def _rolling_zscore(self, series: pd.Series, window: int) -> pd.Series:
        """
        Compute rolling z-score.
        
        Args:
            series: Input series
            window: Window size
            
        Returns:
            Series of z-scores
        """
        rolling_mean = series.rolling(window=window, min_periods=window).mean()
        rolling_std = series.rolling(window=window, min_periods=window).std()
        zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        return zscore
    
    def generate_stats(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rolling statistics for all features.
        
        Args:
            features: Base feature DataFrame
            
        Returns:
            DataFrame with rolling statistics (prefixed with stat name and window)
        """
        all_stats = []
        
        for col in features.columns:
            series = features[col]
            
            # Skip if all NaN or constant
            if series.isna().all() or series.nunique() <= 1:
                continue
            
            for window in self.windows:
                # Rolling mean
                rolling_mean = series.rolling(window=window, min_periods=window).mean()
                all_stats.append(rolling_mean.rename(f"{col}_mean_{window}"))
                
                # Rolling std
                rolling_std = series.rolling(window=window, min_periods=window).std()
                all_stats.append(rolling_std.rename(f"{col}_std_{window}"))
                
                # Rolling max
                rolling_max = series.rolling(window=window, min_periods=window).max()
                all_stats.append(rolling_max.rename(f"{col}_max_{window}"))
                
                # Rolling min
                rolling_min = series.rolling(window=window, min_periods=window).min()
                all_stats.append(rolling_min.rename(f"{col}_min_{window}"))
                
                # Rolling slope
                rolling_slope = self._rolling_slope(series, window)
                all_stats.append(rolling_slope.rename(f"{col}_slope_{window}"))
                
                # Rolling zscore
                rolling_zscore = self._rolling_zscore(series, window)
                all_stats.append(rolling_zscore.rename(f"{col}_zscore_{window}"))
        
        if not all_stats:
            return pd.DataFrame(index=features.index)
        
        # Combine all statistics
        stats_df = pd.concat(all_stats, axis=1)
        
        return stats_df


class RegimeContextFeatures:
    """
    Generates regime-specific context features:
    - Rolling ATR
    - Rolling volatility (std of returns)
    - Rolling kurtosis and skew
    - Volume surge ratios
    - Return bins (bucketized)
    """
    
    def __init__(self, atr_windows: List[int] = None, return_bins: int = 10):
        """
        Initialize regime context feature generator.
        
        Args:
            atr_windows: List of window sizes for ATR (default: [3, 6, 12, 24])
            return_bins: Number of bins for return bucketization (default: 10)
        """
        if atr_windows is None:
            atr_windows = [3, 6, 12, 24]
        self.atr_windows = atr_windows
        self.return_bins = return_bins
    
    def _wilder_atr(self, df: pd.DataFrame, n: int) -> pd.Series:
        """Compute Wilder's ATR."""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate regime context features from OHLCV data.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with regime context features
        """
        all_features = []
        close = df["close"]
        volume = df.get("volume", pd.Series(1.0, index=df.index))
        
        # Rolling ATR for different windows
        for window in self.atr_windows:
            atr = self._wilder_atr(df, window)
            all_features.append(atr.rename(f"atr_{window}"))
        
        # Rolling volatility (std of returns)
        returns = close.pct_change()
        for window in self.atr_windows:
            vol = returns.rolling(window=window, min_periods=window).std()
            all_features.append(vol.rename(f"volatility_{window}"))
        
        # Rolling kurtosis and skew
        for window in self.atr_windows:
            rolling_kurt = returns.rolling(window=window, min_periods=window).apply(
                lambda x: x.kurtosis() if len(x.dropna()) >= window else np.nan, raw=False
            )
            rolling_skew = returns.rolling(window=window, min_periods=window).apply(
                lambda x: x.skew() if len(x.dropna()) >= window else np.nan, raw=False
            )
            all_features.append(rolling_kurt.rename(f"kurtosis_{window}"))
            all_features.append(rolling_skew.rename(f"skew_{window}"))
        
        # Volume surge ratios
        # Compare current volume to rolling average
        for window in [6, 12, 24]:
            vol_ma = volume.rolling(window=window, min_periods=window).mean()
            vol_surge = (volume / vol_ma.replace(0, np.nan)).rename(f"volume_surge_{window}")
            all_features.append(vol_surge)
        
        # Return bins (bucketized)
        # Create quantile-based bins
        for window in [6, 12, 24]:
            rolling_returns = returns.rolling(window=window, min_periods=window)
            
            # Compute quantiles for binning
            q_low = rolling_returns.quantile(0.33)
            q_high = rolling_returns.quantile(0.67)
            
            # Bin returns: -1 (low), 0 (mid), 1 (high)
            return_bin = pd.Series(0, index=df.index, dtype=int)
            return_bin[returns < q_low] = -1
            return_bin[returns > q_high] = 1
            return_bin = return_bin.rename(f"return_bin_{window}")
            all_features.append(return_bin)
        
        # Combine all features
        if not all_features:
            return pd.DataFrame(index=df.index)
        
        features_df = pd.concat(all_features, axis=1)
        
        return features_df


class SignalDynamicsFeatures:
    """
    Generates signal dynamics features:
    - Signal streak length
    - Time since last long signal
    - Time since last short signal
    - Distance-to-previous-extreme for hull
    - Derivative of topvec and botvec
    """
    
    def __init__(self, signal_modules: List[SignalModule] = None):
        """
        Initialize signal dynamics feature generator.
        
        Args:
            signal_modules: List of SignalModule instances to compute signals from
        """
        self.signal_modules = signal_modules if signal_modules is not None else []
    
    def _compute_signal_streak(self, signal: pd.Series) -> pd.Series:
        """Compute consecutive signal streak length."""
        streak = pd.Series(0, index=signal.index, dtype=int)
        current_streak = 0
        last_sig = 0
        
        for i in range(len(signal)):
            sig_val = signal.iloc[i]
            if pd.isna(sig_val):
                sig_val = 0
            
            if sig_val == last_sig and sig_val != 0:
                current_streak += 1
            elif sig_val != 0:
                current_streak = 1
            else:
                current_streak = 0
            
            streak.iloc[i] = current_streak
            last_sig = sig_val
        
        return streak
    
    def _time_since_signal(self, signal: pd.Series, target: int) -> pd.Series:
        """Compute bars since last signal of target type (1 for long, -1 for short)."""
        time_since = pd.Series(np.nan, index=signal.index, dtype=float)
        last_idx = None
        
        for i in range(len(signal)):
            sig_val = signal.iloc[i]
            if pd.isna(sig_val):
                sig_val = 0
            
            if sig_val == target:
                last_idx = i
                time_since.iloc[i] = 0
            elif last_idx is not None:
                time_since.iloc[i] = i - last_idx
            else:
                time_since.iloc[i] = np.nan
        
        return time_since
    
    def _distance_to_extreme(self, series: pd.Series, lookback: int = 100) -> pd.Series:
        """Compute distance from current value to previous extreme (max/min) within lookback."""
        distance = pd.Series(np.nan, index=series.index, dtype=float)
        
        for i in range(len(series)):
            if i < lookback:
                continue
            
            window = series.iloc[i-lookback:i]
            if window.isna().all():
                continue
            
            current_val = series.iloc[i]
            if pd.isna(current_val):
                continue
            
            # Find previous max and min
            prev_max = window.max()
            prev_min = window.min()
            
            if pd.isna(prev_max) or pd.isna(prev_min):
                continue
            
            # Distance to extremes
            dist_to_max = abs(current_val - prev_max) / prev_max if prev_max != 0 else np.nan
            dist_to_min = abs(current_val - prev_min) / prev_min if prev_min != 0 else np.nan
            
            # Use minimum distance (closer extreme)
            distance.iloc[i] = min(dist_to_max, dist_to_min) if not (pd.isna(dist_to_max) or pd.isna(dist_to_min)) else np.nan
        
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
        all_features = []
        
        # Compute signals from all signal modules
        module_signals = {}
        for module in self.signal_modules:
            try:
                # Get module-specific features (remove prefix)
                module_feat_cols = [c for c in unified_features.columns if c.startswith(f"{module.name}_")]
                if not module_feat_cols:
                    continue
                
                module_features = unified_features[module_feat_cols].copy()
                # Remove prefix for module's compute_signal
                module_features.columns = [c.replace(f"{module.name}_", "") for c in module_features.columns]
                
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
        hull_modules = ["superma", "pvt", "trendmagic"]
        for module_name in hull_modules:
            hull_col = f"{module_name}_hull"
            if hull_col in unified_features.columns:
                hull = unified_features[hull_col]
                dist_extreme = self._distance_to_extreme(hull, lookback=100)
                all_features.append(dist_extreme.rename(f"{module_name}_hull_dist_extreme"))
        
        # Derivative of topvec and botvec
        for module_name in hull_modules:
            topvec_col = f"{module_name}_topvector01"
            botvec_col = f"{module_name}_botvector01"
            
            if topvec_col in unified_features.columns:
                topvec = unified_features[topvec_col]
                topvec_deriv = topvec.diff().rename(f"{module_name}_topvec_deriv")
                all_features.append(topvec_deriv)
            
            if botvec_col in unified_features.columns:
                botvec = unified_features[botvec_col]
                botvec_deriv = botvec.diff().rename(f"{module_name}_botvec_deriv")
                all_features.append(botvec_deriv)
        
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
        generate_regime_context: bool = True
    ):
        """
        Initialize feature store.
        
        Args:
            generate_rolling_stats: Whether to generate rolling statistics (default: True)
            rolling_windows: List of window sizes for rolling stats (default: [3, 5, 10, 20])
            generate_regime_context: Whether to generate regime context features (default: True)
        """
        self.features = pd.DataFrame()
        self.feature_columns = []
        # Default modules
        self.signal_modules = [
            SuperMA4hr(),
            TrendMagicV2(),
            PVTEliminator(),
        ]
        self.context_modules = [
            PivotRSIContext(),
            LinRegChannelContext(),
        ]
        # Rolling stats generator
        self.generate_rolling_stats = generate_rolling_stats
        self.rolling_stats_generator = RollingStatsGenerator(windows=rolling_windows) if generate_rolling_stats else None
        # Regime context features generator
        self.generate_regime_context = generate_regime_context
        self.regime_context_generator = RegimeContextFeatures() if generate_regime_context else None
        # Signal dynamics features generator
        self.generate_signal_dynamics = True
        self.signal_dynamics_generator = SignalDynamicsFeatures(signal_modules=self.signal_modules)

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
        # Start with base dataframe index
        unified_features = pd.DataFrame(index=df.index)
        all_feature_dfs = []
        
        # Collect features from signal modules
        for module in self.signal_modules:
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
        for module in self.context_modules:
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
        
        # Forward-fill NA values
        unified_features = unified_features.ffill()
        
        # Ensure all features align on the same index
        unified_features = unified_features.reindex(df.index)
        
        # Generate regime context features if enabled
        if self.generate_regime_context and self.regime_context_generator is not None:
            print("Generating regime context features...")
            regime_features = self.regime_context_generator.generate_features(df)
            
            # Align regime features to index
            regime_features = regime_features.reindex(df.index)
            
            # Forward-fill regime features
            regime_features = regime_features.ffill()
            
            # Merge regime features with base features
            unified_features = pd.concat([unified_features, regime_features], axis=1)
            
            print(f"Added {len(regime_features.columns)} regime context features")
        
        # Generate signal dynamics features if enabled
        if self.generate_signal_dynamics and self.signal_dynamics_generator is not None:
            print("Generating signal dynamics features...")
            signal_dynamics = self.signal_dynamics_generator.generate_features(df, unified_features)
            
            # Align signal dynamics features to index
            signal_dynamics = signal_dynamics.reindex(df.index)
            
            # Forward-fill signal dynamics features
            signal_dynamics = signal_dynamics.ffill()
            
            # Merge signal dynamics features with base features
            unified_features = pd.concat([unified_features, signal_dynamics], axis=1)
            
            print(f"Added {len(signal_dynamics.columns)} signal dynamics features")
        
        # Generate rolling statistics if enabled
        if self.generate_rolling_stats and self.rolling_stats_generator is not None:
            print(f"Generating rolling statistics for {len(unified_features.columns)} base features...")
            rolling_stats = self.rolling_stats_generator.generate_stats(unified_features)
            
            # Align rolling stats to index
            rolling_stats = rolling_stats.reindex(df.index)
            
            # Forward-fill rolling stats
            rolling_stats = rolling_stats.ffill()
            
            # Merge rolling stats with base features
            unified_features = pd.concat([unified_features, rolling_stats], axis=1)
            
            print(f"Added {len(rolling_stats.columns)} rolling statistic features")
        
        # Store feature columns for reference
        self.feature_columns = unified_features.columns.tolist()
        self.features = unified_features
        
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
        cleaned = cleaned.ffill().bfill().fillna(0)
        
        # Ensure numeric types
        for col in cleaned.columns:
            if cleaned[col].dtype == 'object':
                try:
                    cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
                except (ValueError, TypeError):
                    pass
        
        return cleaned

