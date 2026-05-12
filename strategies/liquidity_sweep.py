from __future__ import annotations

import pandas as pd


def detect_liquidity_sweep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bullish_sweep"] = False
    out["bearish_sweep"] = False

    prev_high = out["high"].shift(1)
    prev_low = out["low"].shift(1)

    out["bearish_sweep"] = (out["high"] > prev_high) & (out["close"] < prev_high)
    out["bullish_sweep"] = (out["low"] < prev_low) & (out["close"] > prev_low)

    return out
