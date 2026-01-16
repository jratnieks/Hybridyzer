# modules/superma.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import SignalModule

def sma(s, n):
    return s.rolling(n, min_periods=n).mean()

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def vwma(price, vol, n):
    pv = price * vol
    return pv.rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum()

def wilder_atr(df, n):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # Wilder's RMA
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return atr

def crossover(a, b):
    return (a.shift(1) <= b.shift(1)) & (a > b)

def crossunder(a, b):
    return (a.shift(1) >= b.shift(1)) & (a < b)

class SuperMA4hr(SignalModule):
    name = "superma"

    def __init__(
        self,
        hull_len=1,
        topdecay=17, topstage=35, topmalength=20,
        botdecay=43, botstage=30, botmalength=18,
        onlylong=False,
    ):
        self.hull_len = hull_len
        self.topdecay = topdecay
        self.topstage = topstage
        self.topmalength = topmalength
        self.botdecay = botdecay
        self.botstage = botstage
        self.botmalength = botmalength
        self.onlylong = onlylong

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        vol = df.get("volume", pd.Series(1.0, index=df.index))

        # Hull (approx like Pine formula)
        half = max(int(round(self.hull_len/2)), 1)
        sqrtp = max(int(round(np.sqrt(self.hull_len))), 1)
        hull_raw = 2*sma(close, half) - sma(close, self.hull_len)
        hull = vwma(hull_raw, vol, sqrtp)

        decayt = wilder_atr(df, self.topdecay) * 0.005
        decayb = wilder_atr(df, self.botdecay) * 0.005

        # Stateful decaying vectors (loop required)
        topvec = np.zeros(len(df), dtype=float)
        top_final_counter = np.zeros(len(df), dtype=float)

        botvec = np.zeros(len(df), dtype=float)
        bot_final_counter = np.zeros(len(df), dtype=float)

        c = close.values
        dt = decayt.values
        db = decayb.values
        
        # Handle NaN in decay values (warmup period) - use 0 decay
        dt = np.nan_to_num(dt, nan=0.0)
        db = np.nan_to_num(db, nan=0.0)

        last_top_touch = -1
        last_bot_touch = -1

        for i in range(len(df)):
            prev_top = topvec[i-1] if i > 0 else c[i]
            prev_bot = botvec[i-1] if i > 0 else c[i]
            
            # Handle NaN propagation - reset to current price if prev is NaN
            if np.isnan(prev_top):
                prev_top = c[i]
            if np.isnan(prev_bot):
                prev_bot = c[i]

            prev_tfc = top_final_counter[i-1] if i > 0 else self.topstage
            prev_bfc = bot_final_counter[i-1] if i > 0 else self.botstage

            # update top vector
            if c[i] >= prev_top:
                topvec[i] = c[i]
                last_top_touch = i
            else:
                topvec[i] = prev_top - (dt[i] * prev_tfc)

            # update bot vector
            if c[i] <= prev_bot:
                botvec[i] = c[i]
                last_bot_touch = i
            else:
                botvec[i] = prev_bot + (db[i] * prev_bfc)

            # counters based on bars since last touch (Pine uses n/valuewhen; 87 hardcoded)
            top_count = i - last_top_touch if last_top_touch >= 0 else 0
            bot_count = i - last_bot_touch if last_bot_touch >= 0 else 0

            top_final_counter[i] = self.topstage if top_count <= 87 else (15 if top_count <= 87*4 else 12)
            bot_final_counter[i] = self.botstage if bot_count <= 87 else (15 if bot_count <= 87*4 else 12)

        topvec = pd.Series(topvec, index=df.index, name="topvector01")
        botvec = pd.Series(botvec, index=df.index, name="botvector01")

        topvec_ma = sma(topvec, self.topmalength).rename("topvecMA")
        botvec_ma = sma(botvec, self.botmalength).rename("botvecMA")

        feats = pd.concat([
            hull.rename("hull"),
            topvec,
            botvec,
            topvec_ma,
            botvec_ma,
            decayt.rename("decayt"),
            decayb.rename("decayb"),
        ], axis=1)
        return feats

    def compute_signal(self, feats: pd.DataFrame) -> pd.Series:
        hull = feats["hull"]
        topvec_ma = feats["topvecMA"]
        botvec_ma = feats["botvecMA"]

        long_sig = crossover(hull, topvec_ma)
        short_sig = crossunder(hull, botvec_ma) & (~self.onlylong)

        sig = pd.Series(0, index=feats.index, dtype=int, name="superma_sig")
        sig[long_sig] = 1
        sig[short_sig] = -1
        return sig
