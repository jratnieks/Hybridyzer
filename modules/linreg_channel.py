# modules/linreg_channel.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import ContextModule

class LinRegChannelContext(ContextModule):
    name = "linreg"

    def __init__(self, length=100, deviations=2.0):
        self.length = length
        self.deviations = deviations

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        src = df["close"].astype(float)
        n = self.length

        # rolling linear regression with slope/intercept
        idx = np.arange(n)
        idx_mean = idx.mean()
        denom = ((idx - idx_mean) ** 2).sum()

        def reg_window(y):
            y = y.values
            y_mean = y.mean()
            slope = ((idx - idx_mean) * (y - y_mean)).sum() / denom
            intercept = y_mean - slope * idx_mean
            fitted_last = intercept + slope * idx[-1]
            return fitted_last, slope, intercept

        fitted = []
        slope_list = []
        intercept_list = []
        dev_list = []

        for i in range(len(src)):
            if i < n-1:
                fitted.append(np.nan); slope_list.append(np.nan); intercept_list.append(np.nan); dev_list.append(np.nan)
                continue
            window = src.iloc[i-n+1:i+1]
            f_last, sl, ic = reg_window(window)
            fitted.append(f_last); slope_list.append(sl); intercept_list.append(ic)

            # deviation around regression line
            y = window.values
            pred = ic + sl * idx
            dev = np.sqrt(((y - pred) ** 2).mean())
            dev_list.append(dev)

        lrc = pd.Series(fitted, index=df.index, name="lr_mid")
        lr_slope = pd.Series(slope_list, index=df.index, name="lr_slope")
        lr_int = pd.Series(intercept_list, index=df.index, name="lr_intercept")
        lr_dev = pd.Series(dev_list, index=df.index, name="lr_dev")

        upper = (lrc + lr_dev * self.deviations).rename("lr_upper")
        lower = (lrc - lr_dev * self.deviations).rename("lr_lower")
        width = (upper - lower).rename("lr_width")

        # price position within channel (0=lower, 1=upper)
        pos = ((src - lower) / (width.replace(0, np.nan))).clip(0, 1).rename("lr_pos")

        feats = pd.concat([lrc, lr_slope, lr_int, lr_dev, upper, lower, width, pos], axis=1)
        return feats
