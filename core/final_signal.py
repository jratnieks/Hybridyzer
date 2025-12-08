# core/final_signal.py
"""
Final signal generator that combines RegimeDetector, DirectionBlender, and SignalBlender.
Produces clean, tradable signals with confidence thresholds and regime gating.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Optional, Dict, Literal, Any
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from core.regime_detector import RegimeDetector
from core.signal_blender import SignalBlender, DirectionBlender
from core.scalers import Scaler
from core.feature_store import FeatureStore
from modules.superma import SuperMA4hr
from modules.trendmagic import TrendMagicV2
from modules.pvt_eliminator import PVTEliminator
from modules.pivots_rsi import PivotRSIContext
from modules.linreg_channel import LinRegChannelContext


# Cache for regime-specific models
_REGIME_SIGNAL_CACHE = {}
_REGIME_DIR_CACHE = {}


def _regime_to_slug(name: str) -> str:
    """Convert regime name to slug for filenames."""
    return name.lower().replace(" ", "_")


def _load_regime_signal(regime: str, use_gpu: bool = True) -> Optional[SignalBlender]:
    """
    Load regime-specific SignalBlender model.
    Prefers calibrated model if available.
    
    Args:
        regime: Regime name (e.g., 'trend_up', 'trend_down', 'chop')
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        SignalBlender instance if model exists, None otherwise
    """
    slug = _regime_to_slug(regime)
    # Try calibrated model first
    path_calibrated = Path("models") / f"blender_{slug}_calibrated.pkl"
    path = Path("models") / f"blender_{slug}.pkl"
    
    # Prefer calibrated model if available
    model_path = path_calibrated if path_calibrated.exists() else path
    
    if model_path.exists():
        if regime not in _REGIME_SIGNAL_CACHE:
            try:
                model = SignalBlender(use_gpu=use_gpu)
                model.load(str(model_path))
                _REGIME_SIGNAL_CACHE[regime] = model
            except Exception:
                return None
        return _REGIME_SIGNAL_CACHE[regime]
    return None


def _load_regime_direction(regime: str, use_gpu: bool = True) -> Optional[DirectionBlender]:
    """
    Load regime-specific DirectionBlender model.
    Prefers calibrated model if available.
    
    Args:
        regime: Regime name (e.g., 'trend_up', 'trend_down', 'chop')
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        DirectionBlender instance if model exists, None otherwise
    """
    slug = _regime_to_slug(regime)
    # Try calibrated model first
    path_calibrated = Path("models") / f"blender_direction_{slug}_calibrated.pkl"
    path = Path("models") / f"blender_direction_{slug}.pkl"
    
    # Prefer calibrated model if available
    model_path = path_calibrated if path_calibrated.exists() else path
    
    if model_path.exists():
        if regime not in _REGIME_DIR_CACHE:
            try:
                model = DirectionBlender(use_gpu=use_gpu)
                model.load(str(model_path))
                _REGIME_DIR_CACHE[regime] = model
            except Exception:
                return None
        return _REGIME_DIR_CACHE[regime]
    return None


class FinalSignalGenerator:
    """
    Unified signal generator combining RegimeDetector, DirectionBlender, and SignalBlender.
    
    Logic:
    1. RegimeDetector gates trades: if regime == 'chop' → signal = 0
    2. If regime != 'chop', use DirectionBlender with confidence threshold
    3. Optionally require agreement with 3-class SignalBlender
    """
    
    def __init__(
        self,
        probability_threshold: float = 0.60,
        p_long: Optional[float] = None,
        p_short: Optional[float] = None,
        require_blender_agreement: bool = False,
        allow_low_confidence: bool = True,
        use_gpu: bool = True,
        regime_detector: Optional[RegimeDetector] = None,
        signal_blender: Optional[SignalBlender] = None,
        direction_blender: Optional[DirectionBlender] = None,
        scaler: Optional[Scaler] = None,
        regime_side_policy: Optional[Dict[str, Dict[str, bool]]] = None,
        disable_regime: Optional[Dict[str, bool]] = None,
        disable_shorts: bool = False,
        min_ev: Optional[float] = None,
        calibration_csv_path: Optional[str] = None
    ):
        """
        Initialize final signal generator.
        
        Args:
            probability_threshold: Minimum probability for DirectionBlender to take a trade (default: 0.60)
            require_blender_agreement: If True, require 3-class SignalBlender to agree with DirectionBlender (default: False)
            allow_low_confidence: If True, allow trades with proba >= 0.25 even if below main threshold (default: True)
            use_gpu: Whether to use GPU acceleration (default: True)
            regime_detector: Optional pre-loaded RegimeDetector instance
            signal_blender: Optional pre-loaded SignalBlender instance
            direction_blender: Optional pre-loaded DirectionBlender instance
            scaler: Optional pre-loaded Scaler instance
            regime_side_policy: Optional dict mapping regime names to {'long': bool, 'short': bool} (default: trend_up allows both, others disabled)
            disable_regime: Optional dict mapping regime names to bool to completely suppress regimes (default: trend_down and chop disabled)
            p_long: Optional separate threshold for long trades (overrides probability_threshold for longs)
            p_short: Optional separate threshold for short trades (overrides probability_threshold for shorts)
            disable_shorts: If True, force all short signals to be rejected (default: False)
            min_ev: Minimum expected value threshold for EV filtering (default: None, disabled)
            calibration_csv_path: Path to calibration CSV file for EV lookup (default: None)
        """
        # Handle separate long/short thresholds
        if p_long is not None or p_short is not None:
            self.p_long = p_long if p_long is not None else probability_threshold
            self.p_short = p_short if p_short is not None else probability_threshold
        else:
            self.p_long = probability_threshold
            self.p_short = probability_threshold
        
        self.probability_threshold = probability_threshold  # Keep for backward compatibility
        self.require_blender_agreement = require_blender_agreement
        self.allow_low_confidence = allow_low_confidence
        self.use_gpu = use_gpu
        
        # Models (can be provided directly or loaded via load_models())
        self.regime_detector = regime_detector
        self.direction_blender = direction_blender
        self.signal_blender = signal_blender
        self.scaler = scaler
        
        # Regime/side policy: defines which trades are allowed per regime
        default_regime_side_policy = {
            'trend_up':   {'long': True,  'short': True},
            'trend_down': {'long': False, 'short': False},
            'chop':       {'long': False, 'short': False},
        }
        self.regime_side_policy = regime_side_policy or default_regime_side_policy
        
        # Disable regime toggle: completely suppress entire regimes
        default_disable_regime = {
            'trend_up': False,
            'trend_down': True,
            'chop': True,
        }
        self.disable_regime = disable_regime if disable_regime is not None else default_disable_regime
        
        # Global flag to disable shorts completely
        self.disable_shorts_flag = disable_shorts
        
        # EV filtering parameters
        self.min_ev = min_ev
        self.calibration_csv_path = calibration_csv_path
        self.ev_lookup = None  # Will be populated if calibration CSV is provided (list of records)
        self.ev_loaded = False  # Flag indicating if EV lookup is loaded and ready
        self.ev_filtered_count = 0  # Track how many trades were filtered by EV
        
        # Load calibration CSV if provided (for EV computation and optional filtering)
        # Always load if calibration_csv_path is provided, regardless of min_ev
        if self.calibration_csv_path is not None:
            self._load_calibration_for_ev()
        elif self.min_ev is not None:
            print("Warning: min_ev specified but calibration_csv_path not provided. EV filtering disabled.")
            self.min_ev = None
        
        # Initialize feature_store if models are provided (needed for _prepare_blend_features)
        # If models are loaded via load_models(), feature_store will be initialized there
        if regime_detector is not None or direction_blender is not None or signal_blender is not None:
            self.feature_store = FeatureStore(use_gpu=self.use_gpu)
            self.feature_store.signal_modules = [
                SuperMA4hr(),
                TrendMagicV2(),
                PVTEliminator(),
            ]
            self.feature_store.context_modules = [
                PivotRSIContext(),
                LinRegChannelContext(),
            ]
        else:
            self.feature_store: Optional[FeatureStore] = None
        
        print(f"[FinalSignalGenerator] Initialized with:")
        print(f"  Probability threshold (general): {probability_threshold}")
        print(f"  Effective p_long: {self.p_long}")
        print(f"  Effective p_short: {self.p_short}")
        print(f"  Require blender agreement: {require_blender_agreement}")
        print(f"  Allow low confidence: {allow_low_confidence}")
        print(f"  GPU mode: {use_gpu}")
        print(f"  Regime/side policy: {self.regime_side_policy}")
        print(f"  Disable regime: {self.disable_regime}")
        print(f"  Disable shorts: {self.disable_shorts_flag}")
        if self.min_ev is not None:
            print(f"  Min EV threshold: {self.min_ev}")
            print(f"  Calibration CSV: {self.calibration_csv_path}")
            if self.ev_lookup is not None:
                print(f"  ✓ Calibration data loaded for EV filtering")
        if regime_detector is not None:
            print(f"  ✓ RegimeDetector provided")
        if direction_blender is not None:
            print(f"  ✓ DirectionBlender provided")
        if signal_blender is not None:
            print(f"  ✓ SignalBlender provided")
        if scaler is not None:
            print(f"  ✓ Scaler provided")
        if self.feature_store is not None:
            print(f"  ✓ FeatureStore initialized")
    
    def load_models(
        self,
        regime_model_path: str = "models/regime_model.pkl",
        direction_model_path: str = "models/blender_direction_model.pkl",
        blender_model_path: str = "models/blender_model.pkl",
        scaler_path: str = "models/feature_scaler.pkl"
    ) -> None:
        """
        Load all required models and scaler.
        Only loads models that haven't been provided via constructor.
        
        Args:
            regime_model_path: Path to RegimeDetector model
            direction_model_path: Path to DirectionBlender model
            blender_model_path: Path to SignalBlender (3-class) model
            scaler_path: Path to feature scaler
        """
        print("\n[FinalSignalGenerator] Loading models...")
        
        # Load scaler if not provided
        if self.scaler is None:
            print(f"  Loading scaler from {scaler_path}...")
            self.scaler = Scaler.load(scaler_path)
            print("  ✓ Scaler loaded")
        else:
            print("  ✓ Scaler already provided")
        
        # Load RegimeDetector if not provided
        if self.regime_detector is None:
            print(f"  Loading RegimeDetector from {regime_model_path}...")
            self.regime_detector = RegimeDetector(use_gpu=self.use_gpu)
            self.regime_detector.load(regime_model_path)
            print("  ✓ RegimeDetector loaded")
        else:
            print("  ✓ RegimeDetector already provided")
        
        # Load DirectionBlender if not provided
        if self.direction_blender is None:
            print(f"  Loading DirectionBlender from {direction_model_path}...")
            self.direction_blender = DirectionBlender(use_gpu=self.use_gpu)
            self.direction_blender.load(direction_model_path)
            print("  ✓ DirectionBlender loaded")
        else:
            print("  ✓ DirectionBlender already provided")
        
        # Load SignalBlender (3-class) if not provided
        if self.signal_blender is None:
            print(f"  Loading SignalBlender from {blender_model_path}...")
            self.signal_blender = SignalBlender(use_gpu=self.use_gpu)
            self.signal_blender.load(blender_model_path)
            print("  ✓ SignalBlender loaded")
        else:
            print("  ✓ SignalBlender already provided")
        
        # Initialize feature store (always needed for _prepare_blend_features)
        if self.feature_store is None:
            self.feature_store = FeatureStore(use_gpu=self.use_gpu)
            self.feature_store.signal_modules = [
                SuperMA4hr(),
                TrendMagicV2(),
                PVTEliminator(),
            ]
            self.feature_store.context_modules = [
                PivotRSIContext(),
                LinRegChannelContext(),
            ]
            print("  ✓ FeatureStore initialized")
        else:
            print("  ✓ FeatureStore already provided")
        
        print("\n[FinalSignalGenerator] All models ready!")
    
    def _load_calibration_for_ev(self) -> None:
        """
        Load calibration CSV and build a hierarchical EV lookup table.
        
        Expected columns (some optional depending on which CSV is passed):
          - prob_min (float)
          - prob_max (float)
          - avg_gross_return (float)
          - prob_bin (optional label, kept only for logging)
          - side_str (optional: 'long'/'short')
          - regime (optional: 'trend_up'/'trend_down'/'chop')
        
        We build a list of rows we can query by:
          (regime, side_str) + prob range
          (None, side_str)   + prob range
          (regime, None)     + prob range
          (None, None)       + prob range
        """
        import math
        
        if not self.calibration_csv_path:
            self.ev_lookup = None
            self.ev_loaded = False
            print("[FinalSignalGenerator] No calibration CSV provided for EV filtering")
            return
        
        cal_path = Path(self.calibration_csv_path)
        if not cal_path.exists():
            print(f"[FinalSignalGenerator] Calibration CSV not found: {cal_path}. EV filtering will be disabled.")
            self.ev_lookup = None
            self.ev_loaded = False
            return
        
        try:
            df = pd.read_csv(cal_path)
        except Exception as e:
            print(f"[FinalSignalGenerator] Failed to load calibration CSV '{cal_path}': {e}")
            self.ev_lookup = None
            self.ev_loaded = False
            return
        
        required_cols = {'avg_gross_return'}
        if not required_cols.issubset(df.columns):
            print(f"[FinalSignalGenerator] Calibration CSV '{cal_path}' missing required columns; EV filtering disabled")
            self.ev_lookup = None
            self.ev_loaded = False
            return
        
        # Ensure prob_min/prob_max exist
        if 'prob_min' not in df.columns or 'prob_max' not in df.columns:
            print(f"[FinalSignalGenerator] Calibration CSV '{cal_path}' has no prob_min/prob_max; EV filtering disabled")
            self.ev_lookup = None
            self.ev_loaded = False
            return
        
        # Normalize side/regime cols if present
        # IMPORTANT: Handle NaN values properly - don't convert them to 'nan' strings
        if 'side_str' in df.columns:
            # Convert to string, lowercase, but preserve NaN as None
            df['side_str'] = df['side_str'].apply(
                lambda x: str(x).lower() if pd.notna(x) and x is not None else None
            )
        else:
            df['side_str'] = None
        
        regime_col = None
        for cand in ['regime', 'regime_name', 'regime_str']:
            if cand in df.columns:
                regime_col = cand
                break
        
        if regime_col is not None:
            # Convert to string but preserve NaN as None
            df['regime_key'] = df[regime_col].apply(
                lambda x: str(x) if pd.notna(x) and x is not None else None
            )
        else:
            df['regime_key'] = None
        
        # Drop rows with NaN EV or broken ranges
        df = df[~df['avg_gross_return'].isna()].copy()
        df = df[~df['prob_min'].isna() & ~df['prob_max'].isna()].copy()
        
        if df.empty:
            print(f"[FinalSignalGenerator] Calibration CSV '{cal_path}' produced no valid EV rows")
            self.ev_lookup = None
            self.ev_loaded = False
            return
        
        # Build a list of dicts for fast-ish manual lookup
        records = []
        for _, row in df.iterrows():
            # Extract side_str and regime, converting 'nan' strings to None
            side_val = row['side_str'] if isinstance(row['side_str'], str) else None
            regime_val = row['regime_key'] if isinstance(row['regime_key'], str) else None
            
            # Filter out literal 'nan' strings that may have slipped through
            if side_val is not None and side_val.lower() == 'nan':
                side_val = None
            if regime_val is not None and regime_val.lower() == 'nan':
                regime_val = None
            
            rec = {
                'prob_min': float(row['prob_min']),
                'prob_max': float(row['prob_max']),
                'ev': float(row['avg_gross_return']),
                'prob_bin': row['prob_bin'] if 'prob_bin' in row else None,
                'side_str': side_val,
                'regime': regime_val,
            }
            # Skip degenerate ranges
            if math.isnan(rec['prob_min']) or math.isnan(rec['prob_max']):
                continue
            if rec['prob_max'] <= rec['prob_min']:
                continue
            records.append(rec)
        
        if not records:
            print(f"[FinalSignalGenerator] No usable EV records after cleaning '{cal_path}'")
            self.ev_lookup = None
            self.ev_loaded = False
            return
        
        self.ev_lookup = records
        self.ev_loaded = True
        
        global_ev = np.mean([r['ev'] for r in records])
        print(f"[FinalSignalGenerator] Loaded {len(records)} EV records from '{cal_path}', global EV={global_ev:.6f}")
    
    def _get_prob_bin(self, prob: float) -> tuple:
        """
        Backward-compatible stub. We now use numeric ranges directly.
        Return a trivial (prob, prob) tuple so calls don't explode if any remain.
        """
        return (float(prob), float(prob))
    
    def _lookup_ev(self, prob: float, side_str: str | None, regime: str | None) -> float | None:
        """
        Hierarchical EV lookup with nearest-bin fallback.
        
        Search order:
          1) (regime, side_str, prob range)
          2) (None,  side_str, prob range)
          3) (regime, None,    prob range)
          4) (None,  None,     prob range)
        If no range contains prob, use the range whose center is closest.
        
        Args:
            prob: Probability value (0.0 to 1.0)
            side_str: Side string ('long' or 'short', or None)
            regime: Regime name (e.g., 'trend_up', 'trend_down', 'chop', or None)
            
        Returns:
            EV as float, or None if absolutely nothing usable.
        """
        if not getattr(self, 'ev_loaded', False) or not getattr(self, 'ev_lookup', None):
            return None
        
        prob = float(prob)
        side_str = side_str.lower() if isinstance(side_str, str) else None
        regime = str(regime) if isinstance(regime, str) else None
        
        # Filter out 'nan' strings from inputs
        if side_str is not None and side_str == 'nan':
            side_str = None
        if regime is not None and regime.lower() == 'nan':
            regime = None
        
        records = self.ev_lookup
        
        # Helper: filter records by regime/side and containing prob
        def matching_records(match_regime, match_side):
            res = []
            for r in records:
                if match_regime is not None and r['regime'] != match_regime:
                    continue
                if match_side is not None and r['side_str'] != match_side:
                    continue
                if r['prob_min'] <= prob <= r['prob_max']:
                    res.append(r)
            return res
        
        # Helper: nearest bin if none contain prob
        def nearest_record(match_regime, match_side):
            best = None
            best_dist = None
            for r in records:
                if match_regime is not None and r['regime'] != match_regime:
                    continue
                if match_side is not None and r['side_str'] != match_side:
                    continue
                center = 0.5 * (r['prob_min'] + r['prob_max'])
                dist = abs(center - prob)
                if best is None or dist < best_dist:
                    best = r
                    best_dist = dist
            return best
        
        search_orders = [
            (regime, side_str),
            (None,  side_str),
            (regime, None),
            (None,  None),
        ]
        
        # 1) Try ranges that actually contain prob
        for reg_key, side_key in search_orders:
            candidates = matching_records(reg_key, side_key)
            if candidates:
                # Use average EV of candidates (usually 1)
                ev_vals = [c['ev'] for c in candidates if not np.isnan(c['ev'])]
                if ev_vals:
                    return float(np.mean(ev_vals))
        
        # 2) No containing bins? Use nearest-bin strategy
        for reg_key, side_key in search_orders:
            best = nearest_record(reg_key, side_key)
            if best is not None and not np.isnan(best['ev']):
                return float(best['ev'])
        
        # 3) Nothing at all
        return None
    
    def _expected_value_filter(self, prob: float, side_str: str | None, regime: str | None, min_ev: float) -> bool:
        """
        True = keep trade, False = drop trade.
        
        Uses hierarchical EV lookup with quantile bins and nearest-bin fallback.
        If lookup fails completely, we treat EV=0 (i.e., drop if min_ev > 0).
        
        Args:
            prob: Probability value (0.0 to 1.0)
            side_str: Side string ('long' or 'short', or None)
            regime: Regime name (e.g., 'trend_up', 'trend_down', 'chop', or None)
            min_ev: Minimum required EV threshold
        """
        if min_ev is None or min_ev <= 0 or not getattr(self, 'ev_loaded', False):
            return True  # no EV threshold, keep
        
        ev = self._lookup_ev(prob, side_str, regime)
        
        if ev is None or np.isnan(ev):
            # No usable EV -> treat as 0 to be conservative
            ev = 0.0
        
        return ev >= float(min_ev)
    
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build and scale features from raw OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Scaled feature DataFrame ready for model inference
        """
        if self.feature_store is None:
            raise ValueError("FeatureStore not initialized. Call load_models() first.")
        
        # Build raw features
        features = self.feature_store.build(df)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def _prepare_blend_features(
        self,
        features_scaled: pd.DataFrame,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for SignalBlender and DirectionBlender.
        Includes module signals and regime encoding.
        
        Args:
            features_scaled: Scaled feature DataFrame
            df: Raw OHLCV DataFrame (for module signal computation)
            
        Returns:
            DataFrame with features + signals + regime
        """
        # Compute module signals
        module_signals = {}
        for module in self.feature_store.signal_modules:
            try:
                module_base_feats = module.compute_features(df)
                module_base_feats = module_base_feats.reindex(features_scaled.index)
                module_signals[module.name] = module.compute_signal(module_base_feats)
            except Exception as e:
                print(f"Warning: Failed to compute module signals for {module.name}: {e}")
                continue
        
        # Build X_blend with features + signals + regime
        X_blend = features_scaled.copy()
        for module_name, signal in module_signals.items():
            aligned_signal = signal.reindex(X_blend.index).fillna(0)
            X_blend[f"{module_name}_signal"] = aligned_signal
        
        return X_blend
    
    def predict(
        self,
        features: pd.DataFrame,
        df: Optional[pd.DataFrame] = None
    ) -> tuple[pd.Series, pd.Series]:
        """
        Generate final signals from features.
        
        Args:
            features: Scaled feature DataFrame (if None, will build from df)
            df: Raw OHLCV DataFrame (required if features is None)
            
        Returns:
            Tuple of (final_signal, ev_series):
            - final_signal: Series of final signals (-1 for short, 0 for flat, 1 for long)
            - ev_series: Series of expected values for each trade (aligned with final_signal index)
        """
        if features is None:
            if df is None:
                raise ValueError("Either features or df must be provided")
            features = self._build_features(df)
        
        # Ensure models are loaded
        if self.regime_detector is None or self.direction_blender is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Step 1: Regime predictions
        regime = self.regime_detector.predict(features)
        
        # Step 2: Start all signals at zero (chop stays 0)
        final_signal = pd.Series(0, index=features.index, dtype=int, name='final_signal')
        
        # Initialize direction confidence storage
        self._direction_confidence_temp = pd.Series(0.0, index=features.index, name='direction_confidence')
        
        # Initialize EV series storage
        ev_series = pd.Series(np.nan, index=features.index, name='ev', dtype=float)
        
        # Step 3: Build X_blend once - ensure index alignment
        if df is None:
            X_blend = features.copy()
        else:
            X_blend = self._prepare_blend_features(features, df)
        
        # Ensure X_blend.index == features.index (alignment invariant)
        if not X_blend.index.equals(features.index):
            print(f"WARNING: X_blend.index != features.index, reindexing...")
            X_blend = X_blend.reindex(features.index)
        
        # Add regime column (mapped to numeric for model compatibility)
        regime_map = {'trend_up': 0, 'trend_down': 1, 'chop': 2}
        X_blend["regime"] = regime.map(regime_map).fillna(2)
        
        # Step 4: Process trend_up and trend_down only
        unique_regimes = [r for r in regime.unique() if r != 'chop']
        
        for regime_name in unique_regimes:
            # HARD profile suppression (no trades in this regime)
            if self.disable_regime.get(regime_name, False):
                print(f"[{regime_name}] DISABLED (entire regime suppressed)")
                continue
            
            regime_mask = (regime == regime_name).reindex(features.index, fill_value=False)
            if not regime_mask.any():
                continue
            
            print(f"[{regime_name}] samples: {regime_mask.sum()}")
            
            # Load regime-specific models if available
            regime_signal_model = _load_regime_signal(regime_name, use_gpu=self.use_gpu)
            regime_dir_model = _load_regime_direction(regime_name, use_gpu=self.use_gpu)
            
            signal_model = regime_signal_model if regime_signal_model is not None else self.signal_blender
            dir_model = regime_dir_model if regime_dir_model is not None else self.direction_blender
            
            # Log which models are being used
            if regime_signal_model is not None and regime_dir_model is not None:
                print(f"[{regime_name}] Using regime-specific models (signal+direction)")
            elif regime_signal_model is not None or regime_dir_model is not None:
                print(f"[{regime_name}] Using mixed models (some regime-specific, some global)")
            else:
                print(f"[{regime_name}] Falling back to global models (no regime-specific available)")
            
            # Slice rows for this regime
            X_blend_regime = X_blend.loc[regime_mask]
            
            # Convert predictions to Series/DataFrame with index
            direction_pred = pd.Series(
                dir_model.predict(X_blend_regime),
                index=X_blend_regime.index,
                dtype=int
            )
            
            direction_proba = pd.DataFrame(
                dir_model.predict_proba(X_blend_regime),
                index=X_blend_regime.index
            )
            max_proba = direction_proba.max(axis=1)
            
            # Apply separate thresholds for long vs short
            # Create masks for long and short predictions
            long_pred_mask = (direction_pred == 1)
            short_pred_mask = (direction_pred == -1)
            
            # Count before thresholding for logging
            long_count_pre = long_pred_mask.sum()
            short_count_pre = short_pred_mask.sum()
            
            # Fix probability threshold logic: use p_long/p_short correctly based on direction
            # Create threshold series based on prediction direction
            p_threshold_series = pd.Series(0.0, index=direction_pred.index)
            p_threshold_series[long_pred_mask] = self.p_long if self.p_long is not None else self.probability_threshold
            p_threshold_series[short_pred_mask] = self.p_short if self.p_short is not None else self.probability_threshold
            
            # Compute regime-specific effective threshold multipliers
            if regime_name == 'trend_up' or regime_name == 'trend_down':
                regime_multiplier = 0.8
            else:
                regime_multiplier = 1.0
            
            # Apply thresholds separately for longs and shorts with correct p_long/p_short
            eff_threshold_long = (self.p_long if self.p_long is not None else self.probability_threshold) * regime_multiplier
            eff_threshold_short = (self.p_short if self.p_short is not None else self.probability_threshold) * regime_multiplier
            
            long_confident = (
                (max_proba >= eff_threshold_long) |
                (self.allow_low_confidence & (max_proba >= 0.25))
            )
            short_confident = (
                (max_proba >= eff_threshold_short) |
                (self.allow_low_confidence & (max_proba >= 0.25))
            )
            
            # Combine into soft confidence mask based on prediction direction
            soft_confidence_mask = (long_pred_mask & long_confident) | (short_pred_mask & short_confident)
            
            # Reject low-impact trades BEFORE applying costs
            # Use the correct threshold based on direction
            # The valid mask is already computed via long_confident and short_confident above
            # Count how many would be filtered by threshold
            filtered_by_threshold_mask = ~(long_confident | short_confident)
            filtered_by_threshold_count = filtered_by_threshold_mask.sum()
            # Reindex to features.index to ensure alignment
            soft_confidence_mask = soft_confidence_mask.reindex(features.index, fill_value=False)
            # Intersect with regime_mask to get valid trades for this regime
            confident_mask = soft_confidence_mask & regime_mask
            trade_mask = confident_mask
            print(f"[{regime_name}] confident: {confident_mask.sum()}")
            print(f"[{regime_name}] filtered_by_threshold: {filtered_by_threshold_count}")
            
            # Optional: Agreement filter
            if self.require_blender_agreement and signal_model is not None:
                blend_pred = pd.Series(
                    signal_model.predict(X_blend_regime),
                    index=X_blend_regime.index,
                    dtype=int
                )
                # Compute blend probabilities and get max probability
                blend_proba_df = pd.DataFrame(
                    signal_model.predict_proba(X_blend_regime),
                    index=X_blend_regime.index
                )
                blend_proba = blend_proba_df.max(axis=1)
                
                # Agreement filter logic:
                # - For long signals (direction_pred == 1): skip agreement filter entirely
                # - For short signals (direction_pred == -1): apply agreement filter as normal
                long_mask = (direction_pred == 1)
                short_agreement_mask = (
                    (direction_pred == -1) &
                    (blend_pred == -1) &
                    (blend_proba >= 0.25)
                )
                # Long signals pass without agreement, short signals require agreement
                agreement_mask = long_mask | short_agreement_mask
                
                # Reindex to features.index to ensure alignment
                agreement_mask = agreement_mask.reindex(features.index, fill_value=False)
                # Intersect with regime_mask to get valid trades for this regime
                agreement_mask = agreement_mask & regime_mask
                trade_mask = trade_mask & agreement_mask
            else:
                # No agreement filter, so agreement_mask is not computed
                agreement_mask = None
            
            if agreement_mask is not None:
                print(f"[{regime_name}] agreement: {agreement_mask.sum()}")
            else:
                print(f"[{regime_name}] agreement: {0} (not required)")
            
            # Compute EV for each trade and apply EV filtering (after thresholding but before regime/side-policy filtering)
            # Get current trade indices before EV filtering
            trade_indices_before_ev = features.index[trade_mask]
            ev_filtered_count_regime = 0
            
            # Compute EV for each potential trade (ALWAYS compute, even without min_ev filtering)
            for idx in trade_indices_before_ev:
                prob_val = max_proba.loc[idx]
                side_val = direction_pred.loc[idx]
                regime_val = regime_name
                
                # Map side to string
                side_str = 'long' if side_val > 0 else 'short' if side_val < 0 else None
                
                # Compute EV using hierarchical lookup (if calibration data is loaded)
                trade_ev = self._lookup_ev(prob_val, side_str, regime_val)
                if trade_ev is not None:
                    ev_series.loc[idx] = trade_ev
                # If no calibration data, leave as NaN (don't default to 0.0)
                
                # Apply EV filter if enabled
                if self.min_ev is not None:
                    if not self._expected_value_filter(prob_val, side_str, regime_val, self.min_ev):
                        # Filter out this trade by removing it from trade_mask
                        trade_mask.loc[idx] = False
                        ev_filtered_count_regime += 1
                        self.ev_filtered_count += 1
            
            if self.min_ev is not None and ev_filtered_count_regime > 0:
                print(f"[{regime_name}] trades after EV filtering: {trade_mask.sum()} (filtered {ev_filtered_count_regime} by EV)")
            
            # Apply regime/side policy - correctly load side rules
            side_rules = self.regime_side_policy.get(regime_name, {"long": True, "short": True})
            allow_long = side_rules.get("long", True)
            allow_short = side_rules.get("short", True)
            
            long_count_post = long_pred_mask.sum()
            short_count_post = short_pred_mask.sum()
            print(f"[{regime_name}] side rules: long={allow_long}, short={allow_short}")
            print(f"[{regime_name}] Before thresholding: long={long_count_pre}, short={short_count_pre}")
            print(f"[{regime_name}] After thresholding: long={long_count_post}, short={short_count_post}")
            
            # Zero-out disallowed directions directly in direction_pred
            # This happens AFTER confidence, agreement, and EV filtering checks
            if not allow_long:
                direction_pred.loc[direction_pred == 1] = 0
            if not allow_short:
                direction_pred.loc[direction_pred == -1] = 0
            
            # Update trade_mask to exclude zeroed-out predictions
            # Create mask for valid (non-zero) directions after policy filtering
            valid_direction_mask = (direction_pred != 0)
            # Reindex to features.index to ensure alignment
            valid_direction_mask = valid_direction_mask.reindex(features.index, fill_value=False)
            # Intersect with existing trade_mask
            trade_mask = trade_mask & valid_direction_mask
            
            print(f"[{regime_name}] trades after regime-policy: {trade_mask.sum()}")
            
            # Assign final signals
            # Get indices where trade_mask is True (already aligned to features.index and intersected with regime_mask)
            trade_indices = features.index[trade_mask]
            # These indices are guaranteed to be in X_blend_regime.index (subset of regime_mask)
            final_signal.loc[trade_indices] = direction_pred.loc[trade_indices]
            
            # Store direction confidence for calibration (max probability for the chosen direction)
            # Store max_proba for trades that were actually taken
            self._direction_confidence_temp.loc[trade_indices] = max_proba.loc[trade_indices]
        
        # Hard-filter trades when using 'trend_up_only' or similar profiles
        if self.disable_regime.get("trend_down", False):
            final_signal[regime == "trend_down"] = 0
        
        if self.regime_side_policy.get("trend_up", {}).get("short", True) is False:
            final_signal[(regime == "trend_up") & (final_signal < 0)] = 0
        
        # Apply disable_shorts flag globally
        if self.disable_shorts_flag:
            final_signal[final_signal < 0] = 0
        
        # Post-processing: Prevent trade explosion - cap short ratio
        long_count = (final_signal == 1).sum()
        short_count = (final_signal == -1).sum()
        total_trades = long_count + short_count
        
        max_short_ratio = 0.70  # configurable
        if total_trades > 0 and short_count > 0:
            actual_short_ratio = short_count / total_trades
            if actual_short_ratio > max_short_ratio:
                num_to_zero = int(short_count - (max_short_ratio * total_trades))
                short_indices = final_signal.index[final_signal == -1]
                if len(short_indices) > 0 and num_to_zero > 0:
                    indices_to_zero = random.sample(list(short_indices), min(num_to_zero, len(short_indices)))
                    final_signal.loc[indices_to_zero] = 0
                    print(f"Post-processing: Zeroed out {len(indices_to_zero)} short signals to cap short ratio to <{max_short_ratio:.0%}")
                    print(f"  Before: {long_count} longs, {short_count} shorts (ratio: {actual_short_ratio:.1%})")
                    print(f"  After: {(final_signal == 1).sum()} longs, {(final_signal == -1).sum()} shorts")
        
        # Validate that short-side is fully purged if disable_shorts_flag is set
        if self.disable_shorts_flag:
            short_count_final = (final_signal < 0).sum()
            assert short_count_final == 0, f"disable_shorts_flag is True but {short_count_final} short signals remain"
        
        # Store direction confidence as instance variable for access after prediction
        self.direction_confidence = self._direction_confidence_temp.copy()
        delattr(self, '_direction_confidence_temp')
        
        # Compute EV statistics for trades only
        trade_mask_final = final_signal != 0
        if trade_mask_final.any():
            trade_evs = ev_series[trade_mask_final]
            valid_evs = trade_evs.dropna()
            
            if len(valid_evs) > 0:
                mean_ev = valid_evs.mean()
                median_ev = valid_evs.median()
                ev_negative_count = (valid_evs < 0).sum()
                ev_below_threshold_count = 0
                if self.min_ev is not None:
                    ev_below_threshold_count = (valid_evs < self.min_ev).sum()
                
                print(f"\n[EV Statistics] (trades only):")
                print(f"  Trades with EV data: {len(valid_evs)}/{trade_mask_final.sum()}")
                print(f"  Mean EV: {mean_ev:.6f}")
                print(f"  Median EV: {median_ev:.6f}")
                print(f"  EV < 0 count: {ev_negative_count}")
                if self.min_ev is not None:
                    print(f"  EV < min_ev ({self.min_ev:.6f}) count: {ev_below_threshold_count}")
            else:
                print(f"\n[EV Statistics] No calibration data loaded - EV values not available")
                print(f"  To enable EV computation, provide --calibration-csv argument")
        
        return final_signal, ev_series
    
    def predict_proba(
        self,
        features: pd.DataFrame,
        df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate final signals with probability information.
        
        Args:
            features: Scaled feature DataFrame (if None, will build from df)
            df: Raw OHLCV DataFrame (required if features is None)
            
        Returns:
            Dictionary with keys: 'signal', 'direction_proba', 'regime_proba', 'confidence'
        """
        if features is None:
            if df is None:
                raise ValueError("Either features or df must be provided")
            features = self._build_features(df)
        
        # Ensure models are loaded
        if self.regime_detector is None or self.direction_blender is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Predict regime and probabilities
        regime = self.regime_detector.predict(features)
        regime_proba = self.regime_detector.predict_proba(features)
        
        # Prepare blend features
        if df is None:
            X_blend = features.copy()
            regime_map = {'trend_up': 0, 'trend_down': 1, 'chop': 2}
            X_blend["regime"] = regime.map(regime_map).fillna(2)
        else:
            X_blend = self._prepare_blend_features(features, df)
            regime_map = {'trend_up': 0, 'trend_down': 1, 'chop': 2}
            X_blend["regime"] = regime.map(regime_map).fillna(2)
        
        # Get DirectionBlender probabilities
        direction_proba = self.direction_blender.predict_proba(X_blend)
        max_direction_proba = direction_proba.max(axis=1)
        
        # Generate final signal (ignore ev_series for predict_proba)
        final_signal, _ = self.predict(features, df)
        
        # Compute overall confidence (regime confidence * direction confidence)
        regime_confidence = regime_proba.max(axis=1)
        confidence = regime_confidence * max_direction_proba
        
        return {
            'signal': final_signal,
            'direction_proba': max_direction_proba,
            'regime_proba': regime_confidence,
            'confidence': confidence,
            'regime': regime
        }
    
    def generate_signal(
        self,
        df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Generate final signal from raw OHLCV data.
        Convenience method that handles feature building internally.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Dictionary with keys: 'signal', 'regime', 'confidence', 'direction_proba'
            For single-row input, returns scalar values.
            For multi-row input, returns Series.
        """
        # Build and scale features
        features = self._build_features(df)
        
        # Generate predictions with probabilities
        results = self.predict_proba(features, df)
        
        # Return results (will be Series for multi-row, scalar for single-row)
        return {
            'signal': results['signal'],
            'regime': results['regime'],
            'confidence': results['confidence'],
            'direction_proba': results['direction_proba']
        }
    
    def compute_ev_diagnostics(
        self,
        trade_df: pd.DataFrame,
        output_dir: Path = Path("results"),
        min_ev_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive EV diagnostics for trade analysis.
        
        Args:
            trade_df: DataFrame with columns: future_return, ev, signal, prob (direction_confidence)
            output_dir: Directory to save plots and CSVs (default: results/)
            min_ev_threshold: Optional min_ev threshold for statistics (default: self.min_ev)
            
        Returns:
            Dictionary with EV statistics and diagnostics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided threshold or fall back to instance min_ev
        if min_ev_threshold is None:
            min_ev_threshold = self.min_ev if self.min_ev is not None else 0.0
        
        # Filter to trades only (signal != 0) and valid EV data
        trade_mask = trade_df['signal'] != 0
        trades = trade_df[trade_mask].copy()
        
        # Get valid EV entries (not NaN)
        valid_ev_mask = ~trades['ev'].isna()
        trades_with_ev = trades[valid_ev_mask].copy()
        
        if len(trades_with_ev) == 0:
            print("[EV Analysis] No trades with valid EV data. Skipping diagnostics.")
            return {'error': 'No valid EV data'}
        
        ev_values = trades_with_ev['ev']
        future_returns = trades_with_ev['future_return']
        
        # =====================================================================
        # 1. Compute EV Summary Statistics
        # =====================================================================
        stats = {
            'n_trades_total': len(trades),
            'n_trades_with_ev': len(trades_with_ev),
            'ev_mean': float(ev_values.mean()),
            'ev_median': float(ev_values.median()),
            'ev_std': float(ev_values.std()),
            'ev_min': float(ev_values.min()),
            'ev_max': float(ev_values.max()),
            'ev_q25': float(ev_values.quantile(0.25)),
            'ev_q50': float(ev_values.quantile(0.50)),
            'ev_q75': float(ev_values.quantile(0.75)),
            'ev_positive_count': int((ev_values > 0).sum()),
            'ev_negative_count': int((ev_values < 0).sum()),
            'ev_above_threshold_count': int((ev_values >= min_ev_threshold).sum()) if min_ev_threshold else 0,
            'min_ev_threshold': min_ev_threshold
        }
        
        # =====================================================================
        # 2. Compute Correlations
        # =====================================================================
        # Correlation with future return
        valid_both = ~(ev_values.isna() | future_returns.isna())
        if valid_both.sum() > 2:
            stats['corr_ev_future_return'] = float(ev_values[valid_both].corr(future_returns[valid_both]))
            stats['corr_ev_abs_future_return'] = float(ev_values[valid_both].corr(future_returns[valid_both].abs()))
        else:
            stats['corr_ev_future_return'] = np.nan
            stats['corr_ev_abs_future_return'] = np.nan
        
        # =====================================================================
        # 3. Compute EV Bin Table (10 equal-width bins)
        # =====================================================================
        try:
            # Create 10 equal-width bins
            ev_min, ev_max = ev_values.min(), ev_values.max()
            if ev_min == ev_max:
                # All same value - create single bin
                trades_with_ev['ev_bin'] = pd.Interval(ev_min, ev_max, closed='both')
            else:
                trades_with_ev['ev_bin'] = pd.cut(ev_values, bins=10, duplicates='drop')
            
            # Compute metrics per bin
            def bin_metrics(group):
                fr = group['future_return']
                sig = group['signal']
                hits = ((fr > 0) & (sig > 0)) | ((fr < 0) & (sig < 0))
                return pd.Series({
                    'count': len(group),
                    'hit_rate': hits.mean() if len(group) > 0 else 0.0,
                    'avg_return': fr.mean() if len(group) > 0 else 0.0,
                    'median_return': fr.median() if len(group) > 0 else 0.0,
                    'avg_abs_return': fr.abs().mean() if len(group) > 0 else 0.0,
                    'ev_mean': group['ev'].mean() if len(group) > 0 else 0.0
                })
            
            ev_bins_df = trades_with_ev.groupby('ev_bin', observed=False).apply(bin_metrics).reset_index()
            
            # Add bin edges as separate columns for CSV compatibility
            ev_bins_df['bin_left'] = ev_bins_df['ev_bin'].apply(lambda x: x.left if hasattr(x, 'left') else np.nan)
            ev_bins_df['bin_right'] = ev_bins_df['ev_bin'].apply(lambda x: x.right if hasattr(x, 'right') else np.nan)
            ev_bins_df['ev_bin'] = ev_bins_df['ev_bin'].astype(str)  # Convert to string for CSV
            
            # Save bins CSV
            ev_bins_path = output_dir / "ev_bins.csv"
            ev_bins_df.to_csv(ev_bins_path, index=False)
            stats['ev_bins_path'] = str(ev_bins_path)
        except Exception as e:
            print(f"[EV Analysis] Warning: Could not compute EV bins: {e}")
            ev_bins_df = pd.DataFrame()
            stats['ev_bins_path'] = None
        
        # =====================================================================
        # 4. Generate EV Histogram
        # =====================================================================
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(ev_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='EV = 0')
            if min_ev_threshold and min_ev_threshold > 0:
                ax.axvline(x=min_ev_threshold, color='green', linestyle='--', linewidth=1.5, 
                          label=f'min_ev = {min_ev_threshold:.6f}')
            ax.axvline(x=stats['ev_mean'], color='orange', linestyle='-', linewidth=1.5,
                      label=f'Mean = {stats["ev_mean"]:.6f}')
            ax.set_xlabel('Expected Value (EV)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'EV Distribution (n={len(trades_with_ev)} trades)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            hist_path = output_dir / "ev_hist.png"
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            stats['ev_hist_path'] = str(hist_path)
        except Exception as e:
            print(f"[EV Analysis] Warning: Could not generate histogram: {e}")
            stats['ev_hist_path'] = None
        
        # =====================================================================
        # 5. Generate EV vs Future Return Scatter Plot
        # =====================================================================
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Color by trade direction
            long_mask = trades_with_ev['signal'] > 0
            short_mask = trades_with_ev['signal'] < 0
            
            if long_mask.any():
                ax.scatter(ev_values[long_mask], future_returns[long_mask], 
                          alpha=0.5, s=10, c='green', label='Long')
            if short_mask.any():
                ax.scatter(ev_values[short_mask], future_returns[short_mask], 
                          alpha=0.5, s=10, c='red', label='Short')
            
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            
            # Add correlation text
            corr_text = f"Corr(EV, Return): {stats['corr_ev_future_return']:.3f}"
            ax.text(0.02, 0.98, corr_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Expected Value (EV)')
            ax.set_ylabel('Future Return')
            ax.set_title(f'EV vs Future Return (n={len(trades_with_ev)} trades)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            scatter_path = output_dir / "ev_scatter.png"
            plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            stats['ev_scatter_path'] = str(scatter_path)
        except Exception as e:
            print(f"[EV Analysis] Warning: Could not generate scatter plot: {e}")
            stats['ev_scatter_path'] = None
        
        # =====================================================================
        # 6. Save Summary Statistics CSV
        # =====================================================================
        try:
            summary_df = pd.DataFrame([stats])
            summary_path = output_dir / "ev_analysis_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            stats['summary_path'] = str(summary_path)
        except Exception as e:
            print(f"[EV Analysis] Warning: Could not save summary CSV: {e}")
            stats['summary_path'] = None
        
        # =====================================================================
        # 7. Print Compact Text Report
        # =====================================================================
        self._print_ev_analysis_report(stats)
        
        return stats
    
    def _print_ev_analysis_report(self, stats: Dict[str, Any]) -> None:
        """Print compact EV analysis report to terminal."""
        print("\n" + "="*60)
        print("EV ANALYSIS (DEBUG MODE)")
        print("="*60)
        print(f"Trades with EV data: {stats['n_trades_with_ev']}/{stats['n_trades_total']}")
        print(f"Mean EV: {stats['ev_mean']:.6f}")
        print(f"Median EV: {stats['ev_median']:.6f}")
        print(f"EV Std: {stats['ev_std']:.6f}")
        print(f"EV Range: [{stats['ev_min']:.6f}, {stats['ev_max']:.6f}]")
        print(f"EV Quartiles: Q25={stats['ev_q25']:.6f}, Q50={stats['ev_q50']:.6f}, Q75={stats['ev_q75']:.6f}")
        
        corr_ev_ret = stats.get('corr_ev_future_return', np.nan)
        corr_ev_abs = stats.get('corr_ev_abs_future_return', np.nan)
        if not np.isnan(corr_ev_ret):
            print(f"Correlation(EV, future_return): {corr_ev_ret:.3f}")
        if not np.isnan(corr_ev_abs):
            print(f"Correlation(EV, abs_future_return): {corr_ev_abs:.3f}")
        
        print(f"EV > 0 count: {stats['ev_positive_count']}")
        print(f"EV < 0 count: {stats['ev_negative_count']}")
        
        if stats.get('min_ev_threshold') and stats['min_ev_threshold'] > 0:
            print(f"EV >= threshold count (min_ev={stats['min_ev_threshold']:.2e}): {stats['ev_above_threshold_count']}")
        
        if stats.get('ev_bins_path'):
            print(f"Bins saved to {stats['ev_bins_path']}")
        if stats.get('summary_path'):
            print(f"Summary saved to {stats['summary_path']}")
        if stats.get('ev_hist_path') or stats.get('ev_scatter_path'):
            print(f"Plots saved to results/")
        
        print("="*60 + "\n")

