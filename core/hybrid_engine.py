# core/hybrid_engine.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Optional
from modules.base import SignalModule, ContextModule
from core.feature_store import FeatureStore
from core.regime_detector import RegimeDetector
from core.signal_blender import SignalBlender
from core.risk_layer import RiskLayer


class HybridEngine:
    """
    Central engine that orchestrates all modules and produces final signals.
    """

    def __init__(
        self,
        signal_modules: List[SignalModule],
        context_modules: Optional[List[ContextModule]] = None,
        regime_model_path: str = "models/regime_detector.pkl",
        blender_model_path: str = "models/signal_blender.pkl"
    ):
        """
        Initialize the hybrid engine with modules and load ML models.
        
        Args:
            signal_modules: List of SignalModule instances
            context_modules: Optional list of ContextModule instances
            regime_model_path: Path to regime detector model
            blender_model_path: Path to signal blender model
        """
        self.signal_modules = signal_modules
        self.context_modules = context_modules if context_modules else []
        
        self.feature_store = FeatureStore()
        self.regime_detector = RegimeDetector()
        self.signal_blender = SignalBlender()
        self.risk_layer = RiskLayer()
        
        # Load ML models
        regime_path = Path(regime_model_path)
        blender_path = Path(blender_model_path)
        
        if regime_path.exists():
            self.regime_detector.load(str(regime_path))
            print(f"Loaded regime detector from {regime_path}")
        else:
            print(f"Warning: Regime model not found at {regime_path}")
        
        if blender_path.exists():
            self.signal_blender.load(str(blender_path))
            print(f"Loaded signal blender from {blender_path}")
        else:
            print(f"Warning: Blender model not found at {blender_path}")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main processing pipeline:
        1. Build feature store from all modules
        2. Detect regime
        3. Blend module signals using regime
        4. Apply risk layer
        5. Return final long/short/flat
        
        Args:
            df: Raw OHLCV dataframe with columns: open, high, low, close, volume
            
        Returns:
            DataFrame with columns:
            - signal: Final actionable signals (-1, 0, +1)
            - regime: Detected regime per bar
            - {module_name}_signal: Individual module signals
        """
        # Step 1: Build unified feature store from all modules
        features = self.feature_store.build_features(
            df, self.signal_modules, self.context_modules
        )
        
        # Step 2: Detect regime (pass price_df for ATR calculation)
        regime = self.regime_detector.detect_regime(features, price_df=df)
        
        # Step 3: Compute signals from each signal module
        module_signals = {}
        module_confidences = {}
        
        for module in self.signal_modules:
            # Extract module-specific features (prefixed with module name)
            module_features = self._get_module_features(features, module.name)
            
            # Compute signal and confidence
            module_signals[module.name] = module.compute_signal(module_features)
            module_confidences[module.name] = module.compute_confidence(module_features)
        
        # Step 4: Blend signals based on regime
        blended_signal = self.signal_blender.blend_signals(
            module_signals, regime, module_confidences
        )
        
        # Step 5: Apply risk layer
        final_signal = self.risk_layer.apply_risk(blended_signal, df)
        
        # Step 6: Build output dataframe
        result = pd.DataFrame(index=df.index)
        result['signal'] = final_signal
        result['regime'] = regime
        
        # Include module-level signals for analysis
        for module_name, signal in module_signals.items():
            result[f'{module_name}_signal'] = signal
        
        return result

    def run_bar(self, df_row: pd.Series) -> int:
        """
        Process a single bar and return action signal.
        
        Args:
            df_row: Single row of OHLCV data as Series
            
        Returns:
            Action signal (-1, 0, or +1)
        """
        # Build features for single row
        features = self.feature_store.build_single_row(
            df_row, self.signal_modules, self.context_modules
        )
        features = self.feature_store.clean_features(features)
        
        # Predict regime
        regime = self.regime_detector.predict(features)
        regime_value = regime.iloc[0] if len(regime) > 0 else "neutral"
        
        # Encode regime for blender
        regime_map = {
            'trend_up': 0, 'trend_down': 1, 'chop': 2,
            'high_vol': 3, 'low_vol': 4, 'neutral': 5
        }
        regime_encoded = regime_map.get(regime_value, 5)
        
        # Compute module signals for blender input
        module_signals = {}
        for module in self.signal_modules:
            module_features = self._get_module_features(features, module.name)
            module_signals[module.name] = module.compute_signal(module_features)
        
        # Prepare features for signal blender (features + signals + regime)
        # Use vectorized operations - align all signals to features index
        features_signals = features.copy()
        for module_name, signal in module_signals.items():
            # Align signal to features index (vectorized)
            aligned_signal = signal.reindex(features.index).fillna(0)
            features_signals[f"{module_name}_signal"] = aligned_signal
        # Add regime as constant (vectorized)
        features_signals["regime"] = regime_encoded
        
        # Predict action
        action_series = self.signal_blender.predict(features_signals)
        action = int(action_series.iloc[0]) if len(action_series) > 0 else 0
        
        # Apply risk layer clipping
        action = self.risk_layer.clip(action)
        
        return action

    def _get_module_features(self, features: pd.DataFrame, module_name: str) -> pd.DataFrame:
        """
        Extract features for a specific module from unified feature store.
        
        Args:
            features: Unified feature dataframe with prefixed columns
            module_name: Name of the module
            
        Returns:
            DataFrame with features for the specified module (columns without prefix)
        """
        # Filter features by module prefix
        module_features = features.filter(regex=f'^{module_name}_', axis=1).copy()
        
        # Remove prefix from column names for module consumption
        if not module_features.empty:
            module_features.columns = [col.replace(f'{module_name}_', '') for col in module_features.columns]
        
        return module_features

