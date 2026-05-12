from __future__ import annotations

import pandas as pd
import numpy as np

from strategies.fvg_detector import detect_fvg
from strategies.market_structure import detect_swings, detect_bos
from strategies.liquidity_sweep import detect_liquidity_sweep


class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["return"] = out["close"].pct_change()
        out["range"] = out["high"] - out["low"]
        out["body"] = (out["close"] - out["open"]).abs()
        out["direction"] = np.where(out["close"] > out["open"], 1, -1)
        out["atr"] = self._atr(out, self.config["strategy"].get("atr_period", 14))

        out = detect_fvg(out, self.config)
        out = detect_swings(out, self.config)
        out = detect_bos(out)
        out = detect_liquidity_sweep(out)

        return out.dropna().reset_index(drop=True)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
