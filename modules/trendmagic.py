# modules/trendmagic.py
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

def triple_ema(src, n):
    a = ema(src, n)
    b = ema(a, int(round(n*0.628)))
    c = ema(b, int(round(n*0.314)))
    return c

def compute_zigzag(src, length=10, extreme=3, lookback=1):
    """
    Pine f_zz:
      _hls = tripleEMA(src, length, 0.628, 0.314)
      rising = _hls >= _hls[1]
      zigzag = rising & ~rising[lookback] ? lowest(extreme) :
               ~rising & rising[lookback] ? highest(extreme) :
               na
    We'll output a series with pivot values, ffilled to make it usable.
    """
    hls = triple_ema(src, length)
    rising = hls >= hls.shift(1)

    piv = pd.Series(np.nan, index=src.index)

    low_roll = src.rolling(extreme, min_periods=extreme).min()
    high_roll = src.rolling(extreme, min_periods=extreme).max()

    # Shift and fill NaN - use where() to avoid downcasting warning
    rising_shifted = rising.shift(lookback)
    rising_shifted_filled = rising_shifted.where(rising_shifted.notna(), False).astype(bool)
    turn_up = rising & (~rising_shifted_filled)
    turn_dn = (~rising) & rising_shifted_filled

    piv[turn_up] = low_roll[turn_up]
    piv[turn_dn] = high_roll[turn_dn]

    # Pine leaves NA between pivots. For signals we want last pivot value.
    return piv.ffill().rename("zigzag")

class TrendMagicV2(SignalModule):
    name = "trendmagic"

    def __init__(
        self,
        zz_length=10, zz_extreme=3, zz_lookback=1,
        flength=8,
        vecavglength=9,
        trange=0.01, brange=0.055,
        topstagelength=87, topstage1=18, topstage2=15, topstage3=12,
        botstagelength=87, botstage1=24, botstage2=15, botstage3=12,
        pyramiding_less_than=1, pyramiding_equal_to=0, pyramiding_greater_than=1_000_000,
        use_tp=False, tp=1.50,  # % (1.50 = 1.5%)
        use_sl=False, sl=1.50,
        use_ts=False, tsi=13.00, ts=4.00,
    ):
        self.zz_length = zz_length
        self.zz_extreme = zz_extreme
        self.zz_lookback = zz_lookback
        self.flength = flength
        self.vecavglength = vecavglength
        self.trange = trange
        self.brange = brange
        self.topstagelength = topstagelength
        self.topstage1 = topstage1
        self.topstage2 = topstage2
        self.topstage3 = topstage3
        self.botstagelength = botstagelength
        self.botstage1 = botstage1
        self.botstage2 = botstage2
        self.botstage3 = botstage3
        self.pyrl = pyramiding_less_than
        self.pyre = pyramiding_equal_to
        self.pyrg = pyramiding_greater_than
        self.use_tp = use_tp
        self.tp = tp
        self.use_sl = use_sl
        self.sl = sl
        self.use_ts = use_ts
        self.tsi = tsi
        self.ts = ts

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        vol = df.get("volume", pd.Series(1.0, index=df.index))

        src = close  # Pine security(t, period, close) -> same TF here

        zigzag = compute_zigzag(src, self.zz_length, self.zz_extreme, self.zz_lookback)

        # Vectors (stateful envelope)
        decayt = wilder_atr(df, 10) * 0.005
        decayb = wilder_atr(df, 10) * 0.005

        topvec = np.zeros(len(df), dtype=float)
        botvec = np.zeros(len(df), dtype=float)
        top_fc = np.zeros(len(df), dtype=float)
        bot_fc = np.zeros(len(df), dtype=float)

        s = src.values
        dt = decayt.values
        db = decayb.values

        last_top_touch = -1
        last_bot_touch = -1

        for i in range(len(df)):
            prev_top = topvec[i-1] if i > 0 else s[i]
            prev_bot = botvec[i-1] if i > 0 else s[i]
            prev_tfc = top_fc[i-1] if i > 0 else self.topstage1
            prev_bfc = bot_fc[i-1] if i > 0 else self.botstage1

            if s[i] >= prev_top:
                topvec[i] = s[i]; last_top_touch = i
            else:
                topvec[i] = prev_top - dt[i]*prev_tfc

            if s[i] <= prev_bot:
                botvec[i] = s[i]; last_bot_touch = i
            else:
                botvec[i] = prev_bot + db[i]*prev_bfc

            top_count = i - last_top_touch if last_top_touch >= 0 else 0
            bot_count = i - last_bot_touch if last_bot_touch >= 0 else 0

            top_fc[i] = self.topstage1 if top_count <= self.topstagelength else (
                self.topstage2 if top_count <= self.topstagelength else self.topstage3
            )
            bot_fc[i] = self.botstage1 if bot_count <= self.botstagelength else (
                self.botstage2 if bot_count <= self.botstagelength else self.botstage3
            )

        topvec = pd.Series(topvec, index=df.index, name="topvector01")
        botvec = pd.Series(botvec, index=df.index, name="botvector01")

        price = ema(src, 1).rename("price")
        hls = triple_ema(src, self.flength).rename("hls")

        vecavg = (topvec + botvec) / 2.0
        vecavgt = (topvec + price) / 2.0
        vecavgb = (botvec + price) / 2.0

        avg_vec = ema(vecavg, self.vecavglength).rename("avg_vec")
        topavg = vwma(vecavgt, vol, self.vecavglength).rename("topavg")
        botavg = vwma(vecavgb, vol, self.vecavglength).rename("botavg")

        take = ((avg_vec + price) / 2.0).rename("take")
        tperc = (topavg * self.trange / 100.0).rename("tperc")
        bperc = (botavg * self.brange / 100.0).rename("bperc")

        topsig = (topavg + tperc).rename("topsig")
        botsig = (botavg - bperc).rename("botsig")

        width = ((topsig - botsig) / take).rename("width")

        feats = pd.concat([
            zigzag, topvec, botvec,
            price, hls, avg_vec, topavg, botavg,
            topsig, botsig, width,
        ], axis=1)

        # Base triggers
        feats["long_raw"] = crossunder(zigzag, botsig).astype(int)
        feats["short_raw"] = crossover(zigzag, topsig).astype(int)

        # Pyramiding counts (stateful)
        section_l = np.zeros(len(df), dtype=int)
        section_s = np.zeros(len(df), dtype=int)
        long_cond = np.zeros(len(df), dtype=int)
        short_cond = np.zeros(len(df), dtype=int)

        for i in range(len(df)):
            if feats["long_raw"].iloc[i]:
                section_l[i] = (section_l[i-1] + 1) if i > 0 else 1
                section_s[i] = 0
            else:
                section_l[i] = section_l[i-1] if i > 0 else 0
                section_s[i] = section_s[i-1] if i > 0 else 0

            if feats["short_raw"].iloc[i]:
                section_s[i] = (section_s[i-1] + 1) if i > 0 else 1
                section_l[i] = 0

            allow_long = (section_l[i] <= self.pyrl) or (section_l[i] >= self.pyrg) or (section_l[i] == self.pyre)
            allow_short = (section_s[i] <= self.pyrl) or (section_s[i] >= self.pyrg) or (section_s[i] == self.pyre)

            long_cond[i] = 1 if feats["long_raw"].iloc[i] and allow_long else 0
            short_cond[i] = 1 if feats["short_raw"].iloc[i] and allow_short else 0

        feats["sectionLongs"] = section_l
        feats["sectionShorts"] = section_s
        feats["longCondition"] = long_cond
        feats["shortCondition"] = short_cond

        return feats

    def compute_signal(self, feats: pd.DataFrame) -> pd.Series:
        sig = pd.Series(0, index=feats.index, dtype=int, name="trendmagic_sig")
        sig[feats["longCondition"] == 1] = 1
        sig[feats["shortCondition"] == 1] = -1
        return sig
