# modules/pvt_eliminator.py
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
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def crossover(a, b):
    return (a.shift(1) <= b.shift(1)) & (a > b)

def crossunder(a, b):
    return (a.shift(1) >= b.shift(1)) & (a < b)

class PVTEliminator(SignalModule):
    name = "pvt"

    def __init__(
        self,
        length=23,
        forex_strength=True,
        factor=520,
        factor1=-520,
        hull_len=1,
        topstage=35, topmalength=20,
        botstage=28, botmalength=18,
        topdecay=17, botdecay=43,
    ):
        self.length = length
        self.forexyes = 10000 if forex_strength else 1
        self.factor = factor
        self.factor1 = factor1
        self.hull_len = hull_len
        self.topstage = topstage
        self.topmalength = topmalength
        self.botstage = botstage
        self.botmalength = botmalength
        self.topdecay = topdecay
        self.botdecay = botdecay

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        open_ = df["open"]
        vol = df.get("volume", pd.Series(1.0, index=df.index))

        # PVT residual
        xsma = sma(close, self.length)
        nRes = (close - xsma) * self.forexyes

        is_up = nRes > 0
        is_dn = nRes < 0

        # candle direction
        candledir = np.where(open_.values < close.values, 1, np.where(open_.values > close.values, -1, 0))

        # build sig state machine (uparrowcond/downarrowcond logic approximated)
        sig = np.zeros(len(df), dtype=int)
        prev_up = False
        prev_dn = False

        for i in range(len(df)):
            up_raw = (is_up.iloc[i] and nRes.iloc[i] > self.factor and candledir[i] == 1)
            dn_raw = (is_dn.iloc[i] and nRes.iloc[i] < self.factor1 and candledir[i] == -1)

            up = (not prev_up) and up_raw
            dn = (not prev_dn) and dn_raw

            if up:
                sig[i] = 1
                prev_up, prev_dn = True, False
            elif dn:
                sig[i] = -1
                prev_dn, prev_up = True, False
            else:
                sig[i] = sig[i-1] if i > 0 else 0

        sig = pd.Series(sig, index=df.index, name="sig_pvt")

        # Hull + vectors (same style as SuperMA)
        half = max(int(round(self.hull_len/2)), 1)
        sqrtp = max(int(round(np.sqrt(self.hull_len))), 1)
        hull_raw = 2*sma(close, half) - sma(close, self.hull_len)
        hull = vwma(hull_raw, vol, sqrtp).rename("hull")

        decayt = wilder_atr(df, self.topdecay) * 0.005
        decayb = wilder_atr(df, self.botdecay) * 0.005

        topvec = np.zeros(len(df), dtype=float)
        botvec = np.zeros(len(df), dtype=float)
        top_fc = np.zeros(len(df), dtype=float)
        bot_fc = np.zeros(len(df), dtype=float)

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
                
            prev_tfc = top_fc[i-1] if i > 0 else self.topstage
            prev_bfc = bot_fc[i-1] if i > 0 else self.botstage

            if c[i] >= prev_top:
                topvec[i] = c[i]; last_top_touch = i
            else:
                topvec[i] = prev_top - dt[i]*prev_tfc

            if c[i] <= prev_bot:
                botvec[i] = c[i]; last_bot_touch = i
            else:
                botvec[i] = prev_bot + db[i]*prev_bfc

            top_count = i - last_top_touch if last_top_touch >= 0 else 0
            bot_count = i - last_bot_touch if last_bot_touch >= 0 else 0
            top_fc[i] = self.topstage if top_count <= 87 else (15 if top_count <= 174 else 12)
            bot_fc[i] = self.botstage if bot_count <= 87 else (9 if bot_count <= 174 else 25)

        topvec = pd.Series(topvec, index=df.index, name="topvector01")
        botvec = pd.Series(botvec, index=df.index, name="botvector01")

        topvec_ma = sma(topvec, self.topmalength).rename("topvecMA")
        botvec_ma = sma(botvec, self.botmalength).rename("botvecMA")
        vecavg = ((topvec_ma + botvec_ma) / 2.0).rename("vecavg")

        # breakout conditions
        long_break = crossover(hull, topvec_ma)
        short_break = crossunder(hull, botvec_ma)

        # position state (1-layer only, like Pine)
        in_long = pd.Series(False, index=df.index)
        in_short = pd.Series(False, index=df.index)
        last_long_t = 0
        last_short_t = 0
        for i, t in enumerate(df.index):
            if long_break.iloc[i]:
                last_long_t = i
            if short_break.iloc[i]:
                last_short_t = i
            in_long.iloc[i] = last_long_t > last_short_t
            in_short.iloc[i] = last_short_t > last_long_t

        finallong = (~in_long) & (sig.shift(1) != 1) & (sig == 1) & (hull > vecavg)
        finalshort = (~in_short) & (sig.shift(1) != -1) & (sig == -1) & (hull < vecavg)

        feats = pd.concat([
            nRes.rename("nRes"),
            sig,
            hull, topvec, botvec, topvec_ma, botvec_ma, vecavg,
            pd.Series(finallong.astype(int), index=df.index, name="finallongcond"),
            pd.Series(finalshort.astype(int), index=df.index, name="finalshortcond"),
        ], axis=1)
        return feats

    def compute_signal(self, feats: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=feats.index, dtype=int, name="pvt_sig")
        sig[feats["finallongcond"] == 1] = 1
        sig[feats["finalshortcond"] == 1] = -1
        return sig
