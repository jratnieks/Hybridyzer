# core/pattern_memory.py
"""
Engram-inspired PatternMemory: frozen, pattern-indexed memory of market outcomes.

Stores statistical facts about pattern→outcome relationships and exposes them as:
1. mem_* features for ML models
2. MemoryGate for veto/penalty of low-confidence trades

Memory is built OFFLINE on TRAIN only, frozen for val/test/live.
Constant-time O(1) lookup via discretized pattern keys.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """
    Compute Wilson score interval lower bound for binomial proportion.
    
    Args:
        wins: Number of wins
        n: Total number of trials
        z: Z-score for confidence level (default 1.96 for 95%)
        
    Returns:
        Lower bound of confidence interval [0, 1]
    """
    if n == 0:
        return 0.0
    
    p = wins / n
    z2 = z * z
    denominator = n + z2
    
    # Wilson score formula
    numerator = p * n + z2 / 2
    sqrt_term = z * np.sqrt((p * (1 - p) * n + z2 / 4) / denominator)
    
    lower = (numerator / denominator) - (sqrt_term / denominator)
    return max(0.0, min(1.0, lower))


def rule_based_regime_simple(features: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """
    Simple rule-based regime for pattern key computation.
    Uses only features available before RegimeDetector.
    
    Falls back to make_regime_labels if linreg features not available.
    
    Args:
        features: Feature DataFrame
        df: OHLCV DataFrame
        
    Returns:
        Series of regime labels: 'trend_up', 'trend_down', 'chop'
    """
    # Try to use linreg features if available (from train.py rule_based_regime logic)
    lr_slope_col = "linreg_lr_slope"
    lr_mid_col = "linreg_lr_mid"
    
    if lr_slope_col in features.columns and lr_mid_col in features.columns:
        regimes = pd.Series("chop", index=features.index, dtype=object)
        lr_slope = features[lr_slope_col]
        lr_mid = features[lr_mid_col]
        price = df["close"].reindex(features.index)
        
        threshold = 0.001
        trend_up_mask = (lr_slope > threshold) & (price > lr_mid)
        trend_dn_mask = (lr_slope < -threshold) & (price < lr_mid)
        
        regimes[trend_up_mask] = "trend_up"
        regimes[trend_dn_mask] = "trend_down"
        return regimes
    else:
        # Fallback to make_regime_labels
        from core.labeling import make_regime_labels
        return make_regime_labels(df)


class PatternMemory:
    """
    Pattern-indexed memory that maps discretized feature states to statistical outcomes.
    
    Memory is built offline on TRAIN data only, then frozen for inference.
    """
    
    # Key fields (MLFeatures-only to avoid build_features vs build() mismatch)
    KEY_FIELDS = [
        "return_1",
        "return_5",
        "volatility_20",
        "rsi_14",
        "distance_sma_20",
        "atr_14_pct",
        "vol_compression_5_20",
        "volume_zscore_20",
    ]
    
    # Fixed bucket definitions for categorical fields
    FIXED_BUCKETS = {
        "rsi_14": [0, 30, 45, 55, 70, 100],  # 5 bins
        "vol_compression_5_20": [float('-inf'), 0.8, 1.0, 1.2, float('inf')],  # 4 bins
        "volume_zscore_20": [float('-inf'), -1.0, 1.0, 2.0, float('inf')],  # 4 bins
    }
    
    # Quantile-based fields (7 bins: [q05, q20, q40, q60, q80, q95])
    QUANTILE_FIELDS = [
        "return_1",
        "return_5",
        "volatility_20",
        "distance_sma_20",
        "atr_14_pct",
    ]
    
    def __init__(
        self,
        min_n_build: int = 30,
        min_n_feature: int = 20,
        k: int = 50,
        n0: int = 50,
        horizon_bars: int = 12,
    ):
        """
        Initialize PatternMemory.
        
        Args:
            min_n_build: Minimum samples to store a key (default: 30)
            min_n_feature: Minimum samples for mem_* to return non-defaults (default: 20)
            k: Prior strength for EV shrinkage (default: 50)
            n0: Sample size parameter for mem_conf (default: 50)
            horizon_bars: Forward horizon for outcomes (default: 12)
        """
        self.min_n_build = min_n_build
        self.min_n_feature = min_n_feature
        self.k = k
        self.n0 = n0
        self.horizon_bars = horizon_bars
        
        # Bucket edges (fitted on TRAIN)
        self.bucket_edges: Dict[str, np.ndarray] = {}
        
        # Memory table: {(regime, key_tuple): stats_dict}
        self.memory_table: Dict[Tuple[str, Tuple[int, ...]], Dict] = {}
        
        # Global prior for EV shrinkage
        self.mu_global: float = 0.0
        
        # Metadata
        self.train_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self.is_built = False
    
    def _compute_bucket(self, value: float, field: str, edges: np.ndarray) -> int:
        """Compute bucket index for a value given bucket edges."""
        if np.isnan(value):
            return len(edges) - 1  # NaN bucket (last bucket)
        
        # Find insertion point
        idx = np.searchsorted(edges, value, side='right')
        # Clamp to valid range
        return max(0, min(len(edges) - 1, idx - 1))
    
    def _fit_bucket_edges(self, train_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fit bucket edges on TRAIN data.
        
        Returns:
            Dict mapping field name to bucket edges array
        """
        edges = {}
        
        for field in self.KEY_FIELDS:
            if field not in train_features.columns:
                # Missing field: use default bins
                print(f"[PatternMemory] Warning: Field '{field}' not in features, using default bins")
                edges[field] = np.array([-np.inf, 0, np.inf])
                continue
            
            if field in self.FIXED_BUCKETS:
                # Fixed thresholds
                edges[field] = np.array(self.FIXED_BUCKETS[field])
            elif field in self.QUANTILE_FIELDS:
                # Quantile-based: 7 bins from [q05, q20, q40, q60, q80, q95]
                values = train_features[field].dropna()
                if len(values) == 0:
                    # Fallback: use fixed bins if no data
                    edges[field] = np.array([-np.inf, -0.01, -0.001, 0, 0.001, 0.01, np.inf])
                else:
                    quantiles = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
                    q_values = np.quantile(values, quantiles)
                    # Add -inf and +inf for tails
                    edges[field] = np.concatenate([[-np.inf], q_values, [np.inf]])
            else:
                # Default: quintiles
                values = train_features[field].dropna()
                if len(values) == 0:
                    edges[field] = np.array([-np.inf, 0, np.inf])
                else:
                    quantiles = [0.2, 0.4, 0.6, 0.8]
                    q_values = np.quantile(values, quantiles)
                    edges[field] = np.concatenate([[-np.inf], q_values, [np.inf]])
        
        return edges
    
    def _compute_key(self, row_features: pd.Series, regime: str) -> Tuple[str, Tuple[int, ...]]:
        """
        Compute pattern key for a single row.
        
        Args:
            row_features: Series with feature values
            regime: Regime label ('trend_up', 'trend_down', 'chop')
            
        Returns:
            Tuple of (regime, key_tuple) where key_tuple is bucket indices
        """
        key_buckets = []
        
        for field in self.KEY_FIELDS:
            if field not in row_features:
                # Missing field: use last bucket (NaN bucket)
                if field in self.bucket_edges:
                    bucket = len(self.bucket_edges[field]) - 1
                else:
                    bucket = 0
            else:
                value = row_features[field]
                edges = self.bucket_edges[field]
                bucket = self._compute_bucket(value, field, edges)
            
            key_buckets.append(bucket)
        
        key_tuple = tuple(key_buckets)
        return (regime, key_tuple)
    
    def build(
        self,
        train_features: pd.DataFrame,
        train_df: pd.DataFrame,
        label_series: Optional[pd.Series] = None,
    ) -> None:
        """
        Build memory table from TRAIN data.
        
        Args:
            train_features: Feature DataFrame (must include KEY_FIELDS)
            train_df: OHLCV DataFrame with 'close' column
            label_series: Optional direction labels (-1, 0, +1) for win_rate computation
        """
        print(f"[PatternMemory] Building memory table from {len(train_features)} rows...")
        
        # Validate KEY_FIELDS are present
        missing_fields = [f for f in self.KEY_FIELDS if f not in train_features.columns]
        if missing_fields:
            raise ValueError(f"PatternMemory requires KEY_FIELDS: {missing_fields} not found in features")
        
        # Store train date range
        self.train_date_range = (train_df.index[0], train_df.index[-1])
        
        # Fit bucket edges on TRAIN
        self.bucket_edges = self._fit_bucket_edges(train_features)
        print(f"[PatternMemory] Fitted bucket edges for {len(self.bucket_edges)} fields")
        
        # Compute rule-based regime for each row
        regimes = rule_based_regime_simple(train_features, train_df)
        
        # Compute forward returns
        close = train_df['close']
        future_return = (close.shift(-self.horizon_bars) / close.replace(0, np.nan)) - 1.0
        
        # Compute direction labels if not provided
        if label_series is None:
            # Simple: sign of smoothed return
            smoothed_return = future_return.rolling(window=12, min_periods=1).mean()
            direction = pd.Series(0, index=train_df.index, dtype=int)
            direction[smoothed_return > 0.0005] = 1
            direction[smoothed_return < -0.0005] = -1
        else:
            direction = label_series.reindex(train_df.index).fillna(0).astype(int)
        
        # Aggregate stats per (regime, key)
        aggregates: Dict[Tuple[str, Tuple[int, ...]], Dict] = {}
        
        for idx in train_features.index:
            if idx not in train_df.index:
                continue
            
            row_features = train_features.loc[idx]
            regime = regimes.loc[idx] if idx in regimes.index else "chop"
            key = self._compute_key(row_features, regime)
            
            if key not in aggregates:
                aggregates[key] = {
                    'n': 0,
                    'fwd_returns': [],
                    'directions': [],
                    'mae_values': [],
                    'mfe_values': [],
                }
            
            fwd_ret = future_return.loc[idx] if idx in future_return.index else np.nan
            if not np.isnan(fwd_ret):
                aggregates[key]['n'] += 1
                aggregates[key]['fwd_returns'].append(fwd_ret)
                aggregates[key]['directions'].append(direction.loc[idx] if idx in direction.index else 0)
                
                # Compute MAE/MFE for this sample
                if idx in train_df.index:
                    entry_close = close.loc[idx]
                    dir_val = direction.loc[idx] if idx in direction.index else 0
                    
                    # Get path over horizon
                    path_returns = []
                    idx_loc = train_df.index.get_loc(idx)
                    for h in range(1, self.horizon_bars + 1):
                        if idx_loc + h < len(train_df):
                            path_idx = train_df.index[idx_loc + h]
                            if path_idx in close.index:
                                path_close = close.loc[path_idx]
                                path_ret = (path_close / entry_close) - 1.0
                                path_returns.append(path_ret)
                    
                    if path_returns:
                        if dir_val == 1:  # Long
                            mae = min(path_returns)  # Worst drawdown
                            mfe = max(path_returns)  # Best profit
                        elif dir_val == -1:  # Short
                            mae = max(path_returns)  # Worst drawdown (price went up)
                            mfe = min(path_returns)  # Best profit (price went down)
                        else:
                            mae = 0.0
                            mfe = 0.0
                        
                        aggregates[key]['mae_values'].append(mae)
                        aggregates[key]['mfe_values'].append(mfe)
        
        # Compute global mean EV for shrinkage
        all_fwd_returns = []
        for agg in aggregates.values():
            all_fwd_returns.extend(agg['fwd_returns'])
        self.mu_global = np.mean(all_fwd_returns) if all_fwd_returns else 0.0
        
        # Build memory table (only keys with n >= min_n_build)
        self.memory_table = {}
        for key, agg in aggregates.items():
            if agg['n'] < self.min_n_build:
                continue
            
            fwd_returns = np.array(agg['fwd_returns'])
            directions = np.array(agg['directions'])
            n = agg['n']
            
            # Mean forward return
            mean_fwd_ret = np.mean(fwd_returns)
            
            # Win rate: fraction where (direction==1 and fwd>0) or (direction==-1 and fwd<0)
            wins = np.sum(
                ((directions == 1) & (fwd_returns > 0)) |
                ((directions == -1) & (fwd_returns < 0))
            )
            win_rate = wins / n if n > 0 else 0.5
            
            # Wilson lower bound
            win_rate_lb = wilson_lower_bound(wins, n)
            
            # MAE/MFE percentiles
            mae_values = np.array(agg['mae_values']) if agg['mae_values'] else np.array([0.0])
            mfe_values = np.array(agg['mfe_values']) if agg['mfe_values'] else np.array([0.0])
            
            mae_p50 = np.median(mae_values)
            mae_p90 = np.percentile(mae_values, 90) if len(mae_values) > 0 else 0.0
            mfe_p50 = np.median(mfe_values)
            mfe_p90 = np.percentile(mfe_values, 90) if len(mfe_values) > 0 else 0.0
            
            # EV shrinkage
            ev_shrunk = (n / (n + self.k)) * mean_fwd_ret + (self.k / (n + self.k)) * self.mu_global
            
            # mem_conf
            mem_conf = (1 - np.exp(-n / self.n0)) * max(0, win_rate_lb - 0.5) * 2
            
            self.memory_table[key] = {
                'n': n,
                'mean_fwd_ret': float(mean_fwd_ret),
                'win_rate': float(win_rate),
                'win_rate_lb': float(win_rate_lb),
                'mae_p50': float(mae_p50),
                'mae_p90': float(mae_p90),
                'mfe_p50': float(mfe_p50),
                'mfe_p90': float(mfe_p90),
                'ev_shrunk': float(ev_shrunk),
                'conf': float(mem_conf),
            }
        
        self.is_built = True
        print(f"[PatternMemory] Built memory table with {len(self.memory_table)} keys (min_n={self.min_n_build})")
    
    def transform(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features by adding mem_* columns.
        
        Args:
            features: Feature DataFrame (must include KEY_FIELDS)
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with mem_* columns appended
        """
        if not self.is_built:
            raise ValueError("PatternMemory not built. Call build() first.")
        
        # Validate KEY_FIELDS are present (warn if missing, use defaults)
        missing_fields = [f for f in self.KEY_FIELDS if f not in features.columns]
        if missing_fields:
            print(f"[PatternMemory] Warning: Missing KEY_FIELDS: {missing_fields}. Using default bucket values.")
        
        # Compute rule-based regime
        regimes = rule_based_regime_simple(features, df)
        
        # Initialize mem_* columns
        n_rows = len(features)
        mem_cols = {
            'mem_n': np.zeros(n_rows, dtype=np.float32),
            'mem_hit': np.zeros(n_rows, dtype=np.float32),
            'mem_win_rate': np.full(n_rows, 0.5, dtype=np.float32),
            'mem_win_rate_lb': np.full(n_rows, 0.5, dtype=np.float32),
            'mem_mean_fwd_ret': np.zeros(n_rows, dtype=np.float32),
            'mem_ev_shrunk': np.zeros(n_rows, dtype=np.float32),
            'mem_mae_p50': np.zeros(n_rows, dtype=np.float32),
            'mem_mae_p90': np.zeros(n_rows, dtype=np.float32),
            'mem_mfe_p50': np.zeros(n_rows, dtype=np.float32),
            'mem_mfe_p90': np.zeros(n_rows, dtype=np.float32),
            'mem_conf': np.zeros(n_rows, dtype=np.float32),
        }
        
        # Lookup for each row
        for i, idx in enumerate(features.index):
            row_features = features.loc[idx]
            regime = regimes.loc[idx] if idx in regimes.index else "chop"
            key = self._compute_key(row_features, regime)
            
            if key in self.memory_table:
                stats = self.memory_table[key]
                
                # Only use stats if n >= min_n_feature
                if stats['n'] >= self.min_n_feature:
                    mem_cols['mem_hit'][i] = 1.0
                    mem_cols['mem_n'][i] = float(stats['n'])
                    mem_cols['mem_win_rate'][i] = float(stats['win_rate'])
                    mem_cols['mem_win_rate_lb'][i] = float(stats['win_rate_lb'])
                    mem_cols['mem_mean_fwd_ret'][i] = float(stats['mean_fwd_ret'])
                    mem_cols['mem_ev_shrunk'][i] = float(stats['ev_shrunk'])
                    mem_cols['mem_mae_p50'][i] = float(stats['mae_p50'])
                    mem_cols['mem_mae_p90'][i] = float(stats['mae_p90'])
                    mem_cols['mem_mfe_p50'][i] = float(stats['mfe_p50'])
                    mem_cols['mem_mfe_p90'][i] = float(stats['mfe_p90'])
                    mem_cols['mem_conf'][i] = float(stats['conf'])
        
        # Create DataFrame with same index as features
        mem_df = pd.DataFrame(mem_cols, index=features.index)
        
        return mem_df
    
    def save(self, base_path: Path) -> None:
        """Save memory artifacts to disk."""
        base_path = Path(base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save bucket edges
        bins_path = base_path.parent / "pattern_memory_bins.json"
        bins_data = {
            field: edges.tolist() if isinstance(edges, np.ndarray) else edges
            for field, edges in self.bucket_edges.items()
        }
        with open(bins_path, 'w') as f:
            json.dump(bins_data, f, indent=2)
        
        # Save memory table
        # Convert tuple keys to lists for JSON-serializable format
        table_path = base_path.parent / "pattern_memory_table.pkl"
        table_to_save = {
            (regime, list(key_tuple) if isinstance(key_tuple, tuple) else key_tuple): stats
            for (regime, key_tuple), stats in self.memory_table.items()
        }
        with open(table_path, 'wb') as f:
            pickle.dump(table_to_save, f)
        
        # Save metadata
        meta_path = base_path.parent / "pattern_memory_meta.json"
        meta = {
            'horizon_bars': self.horizon_bars,
            'min_n_build': self.min_n_build,
            'min_n_feature': self.min_n_feature,
            'k': self.k,
            'n0': self.n0,
            'mu_global': self.mu_global,
            'key_fields': self.KEY_FIELDS,
            'train_date_range': [
                str(self.train_date_range[0]) if self.train_date_range else None,
                str(self.train_date_range[1]) if self.train_date_range else None,
            ],
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"[PatternMemory] Saved artifacts to {base_path.parent}")
    
    def load(self, base_path: Path) -> None:
        """Load memory artifacts from disk."""
        base_path = Path(base_path)
        
        # Load bucket edges
        bins_path = base_path.parent / "pattern_memory_bins.json"
        if not bins_path.exists():
            raise FileNotFoundError(f"Bucket edges not found: {bins_path}")
        
        with open(bins_path, 'r') as f:
            bins_data = json.load(f)
        
        self.bucket_edges = {
            field: np.array(edges) for field, edges in bins_data.items()
        }
        
        # Load memory table
        table_path = base_path.parent / "pattern_memory_table.pkl"
        if not table_path.exists():
            raise FileNotFoundError(f"Memory table not found: {table_path}")
        
        with open(table_path, 'rb') as f:
            self.memory_table = pickle.load(f)
        
        # Convert string keys back to tuples
        self.memory_table = {
            (regime, tuple(key_tuple) if isinstance(key_tuple, list) else key_tuple): stats
            for (regime, key_tuple), stats in self.memory_table.items()
        }
        
        # Load metadata
        meta_path = base_path.parent / "pattern_memory_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.horizon_bars = meta['horizon_bars']
        self.min_n_build = meta['min_n_build']
        self.min_n_feature = meta['min_n_feature']
        self.k = meta['k']
        self.n0 = meta['n0']
        self.mu_global = meta['mu_global']
        
        if meta['train_date_range'][0]:
            self.train_date_range = (
                pd.Timestamp(meta['train_date_range'][0]),
                pd.Timestamp(meta['train_date_range'][1]),
            )
        
        self.is_built = True
        print(f"[PatternMemory] Loaded memory table with {len(self.memory_table)} keys")


class MemoryGate:
    """
    Post-ML gate that vetoes or penalizes trades based on memory statistics.
    """
    
    def __init__(
        self,
        min_n_gate: int = 50,
        conf_min: float = 0.25,
        ev_veto: float = -0.0005,
        ev_floor: float = 0.0,
        ev_scale: float = 0.002,
        win_rate_lb_threshold: float = 0.45,
    ):
        """
        Initialize MemoryGate.
        
        Args:
            min_n_gate: Minimum n for gate to apply (below this, no veto)
            conf_min: Minimum conf for hard veto (default: 0.25)
            ev_veto: EV threshold for hard veto (default: -0.0005)
            ev_floor: EV floor for soft penalty (default: 0.0)
            ev_scale: EV scale for soft penalty (default: 0.002)
            win_rate_lb_threshold: Win rate lower bound threshold for direction-specific veto (default: 0.45)
        """
        self.min_n_gate = min_n_gate
        self.conf_min = conf_min
        self.ev_veto = ev_veto
        self.ev_floor = ev_floor
        self.ev_scale = ev_scale
        self.win_rate_lb_threshold = win_rate_lb_threshold
    
    def apply(
        self,
        signals: pd.Series,
        features_with_mem: pd.DataFrame,
        ml_proba: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Apply memory gate to signals.
        
        Args:
            signals: Raw ML signals (-1, 0, +1)
            features_with_mem: Feature DataFrame with mem_* columns
            ml_proba: Optional ML probability for soft penalty (if None, soft penalty disabled)
            
        Returns:
            Gated signals (-1, 0, +1)
        """
        gated = signals.copy()
        
        # Get mem_* columns
        mem_n = features_with_mem.get('mem_n', pd.Series(0, index=signals.index))
        mem_conf = features_with_mem.get('mem_conf', pd.Series(0, index=signals.index))
        mem_ev_shrunk = features_with_mem.get('mem_ev_shrunk', pd.Series(0, index=signals.index))
        mem_win_rate_lb = features_with_mem.get('mem_win_rate_lb', pd.Series(0.5, index=signals.index))
        
        # Hard veto conditions
        # Only apply if mem_n >= min_n_gate (miss = NO veto)
        has_enough_n = mem_n >= self.min_n_gate
        
        # Condition 1: conf >= conf_min AND ev <= ev_veto AND signal != 0
        hard_veto_1 = (
            has_enough_n &
            (mem_conf >= self.conf_min) &
            (mem_ev_shrunk <= self.ev_veto) &
            (signals != 0)
        )
        
        # Condition 2: Direction-specific veto (win_rate_lb < threshold)
        long_signals = signals == 1
        short_signals = signals == -1
        hard_veto_2 = (
            has_enough_n &
            (
                (long_signals & (mem_win_rate_lb < self.win_rate_lb_threshold)) |
                (short_signals & (mem_win_rate_lb < self.win_rate_lb_threshold))
            )
        )
        
        # Apply hard veto
        gated[hard_veto_1 | hard_veto_2] = 0
        
        # Soft penalty (if ml_proba provided)
        if ml_proba is not None:
            # Compute penalty
            penalty = np.clip(
                (self.ev_floor - mem_ev_shrunk) / self.ev_scale,
                0.0, 1.0
            ) * mem_conf
            
            # Adjust probability
            p_adj = ml_proba * (1 - penalty)
            
            # Apply soft penalty: if p_adj drops below threshold, set signal to 0
            # (This is a simplified version; full implementation would use p_adj in thresholding)
            # For now, we just reduce signal strength for low p_adj
            soft_penalty_mask = (
                has_enough_n &
                (mem_conf >= self.conf_min) &
                (mem_conf < 0.5) &  # Between conf_min and 0.5
                (signals != 0) &
                (p_adj < 0.5)  # Adjusted proba below threshold
            )
            gated[soft_penalty_mask] = 0
        
        return gated
