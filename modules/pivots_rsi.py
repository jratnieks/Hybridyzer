# modules/pivots_rsi.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import ContextModule

def sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    roll_dn = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_dn
    return 100 - (100 / (1 + rs))

class PivotRSIContext(ContextModule):
    name = "pivots_rsi"

    def __init__(self, fast=13, slow=21, signal_len=8,
                 rsi_period=14, overbought=70, oversold=30):
        self.fast = fast
        self.slow = slow
        self.signal_len = signal_len
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        fast_ma = ema(close, self.fast)
        slow_ma = ema(close, self.slow)
        macd = (fast_ma - slow_ma).rename("macd")
        signal = sma(macd, self.signal_len).rename("macd_signal")
        hist = (macd - signal).rename("macd_hist")

        # Daily pivots (approx: resample to 1D on your index)
        daily = df.resample("1D").agg({"high":"max","low":"min","close":"last"}).dropna()
        pivot = ((daily["high"] + daily["low"] + daily["close"]) / 3.0).rename("pivot")
        pivot = pivot.reindex(df.index, method="ffill")

        r = rsi(close, self.rsi_period).rename("rsi")
        rsi_ob = (r > self.overbought).astype(int).rename("rsi_overbought")
        rsi_os = (r < self.oversold).astype(int).rename("rsi_oversold")

        feats = pd.concat([macd, signal, hist, pivot, r, rsi_ob, rsi_os], axis=1)
        return feats
