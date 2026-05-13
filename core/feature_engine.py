from __future__ import annotations
from strategies.regime_filters import add_advanced_regime_filters
from strategies.ict_quality import add_ict_quality_filters

import pandas as pd
import numpy as np

from strategies.fvg_detector import detect_fvg
from strategies.market_structure import detect_swings, detect_bos
from strategies.liquidity_sweep import detect_liquidity_sweep
from strategies.filters import (
    add_session_filter,
    add_regime_filter,
    add_fvg_retest,
)


class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        out["return"] = out["close"].pct_change()
        out["range"] = out["high"] - out["low"]
        out["body"] = (out["close"] - out["open"]).abs()
        out["direction"] = np.where(out["close"] > out["open"], 1, -1)

        out["atr"] = self._atr(
            out,
            int(self.config["strategy"].get("atr_period", 14)),
        )

        out = detect_fvg(out, self.config)
        out = detect_swings(out, self.config)
        out = detect_bos(out)
        out = detect_liquidity_sweep(out)

        out = add_fvg_retest(
            out,
            lookback=int(self.config["strategy"].get("fvg_retest_lookback", 20)),
        )

        out = add_session_filter(out)

        out = add_regime_filter(
            out,
            min_atr_pct=float(self.config["strategy"].get("min_atr_pct", 0.0004)),
            max_atr_pct=float(self.config["strategy"].get("max_atr_pct", 0.004)),
        )
        out = add_advanced_regime_filters(out)
        out = add_ict_quality_filters(out, self.config)

        out["fvg_low"] = out["fvg_low"].fillna(0)
        out["fvg_high"] = out["fvg_high"].fillna(0)

        return out.dropna(subset=["return", "atr"]).reset_index(drop=True)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat(
            [high_low, high_close, low_close],
            axis=1,
        ).max(axis=1)

        return tr.rolling(period).mean()