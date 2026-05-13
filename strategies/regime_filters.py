from __future__ import annotations

import pandas as pd


def add_advanced_regime_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()

    out["ema_slope"] = out["ema_20"] - out["ema_20"].shift(10)
    out["ema_trend_up"] = out["ema_20"] > out["ema_50"]
    out["ema_trend_down"] = out["ema_20"] < out["ema_50"]

    out["atr_mean_50"] = out["atr"].rolling(50).mean()
    out["atr_expansion"] = out["atr"] > out["atr_mean_50"]

    out["bos_failure_bullish"] = (
        out["bullish_bos"].shift(1)
        & (out["close"] < out["open"])
    )

    out["bos_failure_bearish"] = (
        out["bearish_bos"].shift(1)
        & (out["close"] > out["open"])
    )

    out["sweep_count_5"] = (
        out["bullish_sweep"].astype(int)
        + out["bearish_sweep"].astype(int)
    ).rolling(5).sum()

    out["too_many_sweeps"] = out["sweep_count_5"] >= 3

    out["long_regime_ok"] = (
        out["ema_trend_up"]
        & (out["ema_slope"] > 0)
        & out["atr_expansion"]
        & ~out["bos_failure_bullish"]
        & ~out["too_many_sweeps"]
    )

    out["short_regime_ok"] = (
        out["ema_trend_down"]
        & (out["ema_slope"] < 0)
        & out["atr_expansion"]
        & ~out["bos_failure_bearish"]
        & ~out["too_many_sweeps"]
    )

    return out