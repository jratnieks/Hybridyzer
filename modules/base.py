# modules/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class SignalModule(ABC):
    """Produces signals in {-1,0,1} plus module-specific features."""
    name: str = "signal"

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def compute_signal(self, feats: pd.DataFrame) -> pd.Series:
        """Return -1, 0, +1 aligned to feats.index."""
        pass

    def compute_confidence(self, feats: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=feats.index, name=f"{self.name}_conf")


class ContextModule(ABC):
    """Produces context/regime features only."""
    name: str = "context"

    @abstractmethod
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
