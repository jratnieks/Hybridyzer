# modules/ohlcv_context.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import ContextModule


class OHLCVContext(ContextModule):
    name = "ohlcv"

    def __init__(
        self,
        windows: list[int] | None = None,
        vr_lag: int = 2,
        memory_atr_period: int = 14,
        memory_alpha_base: float = 0.20,
        memory_alpha_min: float = 0.02,
        memory_alpha_max: float = 0.60,
    ):
        self.windows = windows if windows is not None else [3, 6, 12, 24, 48]
        self.vr_lag = max(2, int(vr_lag))
        self.memory_atr_period = max(2, int(memory_atr_period))
        self.memory_alpha_base = float(memory_alpha_base)
        self.memory_alpha_min = float(memory_alpha_min)
        self.memory_alpha_max = float(memory_alpha_max)

    def _vol_adaptive_memory(self, series: pd.Series, alpha: pd.Series) -> pd.Series:
        values = series.to_numpy()
        alphas = alpha.to_numpy()
        out = np.empty(len(values), dtype=float)
        out[:] = np.nan
        last = np.nan
        for i in range(len(values)):
            v = values[i]
            a = alphas[i]
            if np.isnan(v):
                out[i] = last
                continue
            if np.isnan(last):
                last = v
                out[i] = last
                continue
            if np.isnan(a):
                out[i] = last
                continue
            last = last + a * (v - last)
            out[i] = last
        return pd.Series(out, index=series.index)

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].astype(float)
        open_price = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(1.0, index=df.index)

        returns = close.pct_change()
        abs_returns = returns.abs()
        vol_safe = volume.replace(0, np.nan)
        log_close = np.log(close.replace(0, np.nan))

        log_high_low = np.log(high / low.replace(0, np.nan))
        log_close_open = np.log(close / open_price.replace(0, np.nan))
        log_open_prev_close = np.log(open_price / close.shift(1).replace(0, np.nan))
        log_high_open = np.log(high / open_price.replace(0, np.nan))
        log_low_open = np.log(low / open_price.replace(0, np.nan))
        log_high_close = np.log(high / close.replace(0, np.nan))
        log_low_close = np.log(low / close.replace(0, np.nan))

        ln2 = np.log(2.0)
        park_var = (log_high_low ** 2) / (4.0 * ln2)
        gk_var = 0.5 * (log_high_low ** 2) - (2.0 * ln2 - 1.0) * (log_close_open ** 2)
        rs_var = (log_high_open * log_high_close) + (log_low_open * log_low_close)

        signed_volume = np.sign(returns) * volume

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(
            alpha=1.0 / self.memory_atr_period,
            adjust=False,
            min_periods=self.memory_atr_period,
        ).mean()
        vol_pct = (atr / close.replace(0, np.nan)).replace(0, np.nan)
        alpha_raw = self.memory_alpha_base / (vol_pct + 1e-9)
        alpha_t = alpha_raw.clip(self.memory_alpha_min, self.memory_alpha_max)
        memory = self._vol_adaptive_memory(log_close, alpha_t).rename("mem_vol")
        mem_resid = (log_close - memory).rename("mem_vol_resid")
        mem_resid_norm = (mem_resid / vol_pct.replace(0, np.nan)).rename("mem_vol_resid_norm")
        mem_slope = memory.diff().rename("mem_vol_slope")

        cummax = close.cummax()
        drawdown = (close / cummax.replace(0, np.nan)) - 1.0
        is_new_high = close == cummax
        drawdown_duration = close.groupby(is_new_high.cumsum()).cumcount().astype(float)

        features = [
            memory,
            mem_resid,
            mem_resid_norm,
            mem_slope,
            drawdown.rename("drawdown_depth"),
            drawdown_duration.rename("drawdown_duration"),
        ]

        r_k = close.pct_change(self.vr_lag)

        for window in self.windows:
            min_periods = window

            vol_mean = volume.rolling(window=window, min_periods=min_periods).mean()
            vol_sum = volume.rolling(window=window, min_periods=min_periods).sum()
            signed_sum = signed_volume.rolling(window=window, min_periods=min_periods).sum()

            amihud = (abs_returns / vol_safe).rolling(window=window, min_periods=min_periods).mean()
            impact = (close.diff().abs() / vol_safe).rolling(window=window, min_periods=min_periods).mean()
            vol_ratio = volume / vol_mean.replace(0, np.nan)
            signed_imbalance = signed_sum / vol_sum.replace(0, np.nan)
            up_vol = volume.where(returns > 0, 0.0)
            up_ratio = up_vol.rolling(window=window, min_periods=min_periods).sum() / vol_sum.replace(0, np.nan)

            park_vol = np.sqrt(park_var.rolling(window=window, min_periods=min_periods).mean())
            gk_vol = np.sqrt(gk_var.clip(lower=0).rolling(window=window, min_periods=min_periods).mean())

            if window > 1:
                k = 0.34 / (1.34 + (window + 1) / (window - 1))
                sigma_o = log_open_prev_close.rolling(window=window, min_periods=min_periods).var()
                sigma_c = log_close_open.rolling(window=window, min_periods=min_periods).var()
                sigma_rs = rs_var.rolling(window=window, min_periods=min_periods).mean()
                yz_var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs
                yz_vol = np.sqrt(yz_var.clip(lower=0))
            else:
                yz_vol = pd.Series(np.nan, index=df.index)

            autocorr = returns.rolling(window=window, min_periods=min_periods).corr(returns.shift(1))
            var_1 = returns.rolling(window=window, min_periods=min_periods).var()
            var_k = r_k.rolling(window=window, min_periods=min_periods).var()
            variance_ratio = var_k / (self.vr_lag * var_1.replace(0, np.nan))

            drawdown_min = drawdown.rolling(window=window, min_periods=min_periods).min()
            ret_min = returns.rolling(window=window, min_periods=min_periods).min()

            features.extend([
                amihud.rename(f"amihud_{window}"),
                impact.rename(f"impact_per_vol_{window}"),
                vol_ratio.rename(f"volume_ratio_{window}"),
                signed_imbalance.rename(f"signed_vol_imbalance_{window}"),
                up_ratio.rename(f"up_vol_ratio_{window}"),
                park_vol.rename(f"parkinson_vol_{window}"),
                gk_vol.rename(f"gk_vol_{window}"),
                yz_vol.rename(f"yz_vol_{window}"),
                autocorr.rename(f"return_autocorr_1_{window}"),
                variance_ratio.rename(f"variance_ratio_{window}_k{self.vr_lag}"),
                drawdown_min.rename(f"drawdown_min_{window}"),
                ret_min.rename(f"return_min_{window}"),
            ])

        return pd.concat(features, axis=1)
