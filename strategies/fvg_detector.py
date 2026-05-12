from __future__ import annotations

import pandas as pd


def detect_fvg(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    out["bullish_fvg"] = False
    out["bearish_fvg"] = False
    out["fvg_low"] = None
    out["fvg_high"] = None

    min_points = float(config["strategy"].get("fvg_min_points", 0))

    for i in range(2, len(out)):
        a = out.iloc[i - 2]
        c = out.iloc[i]

        bullish_gap = c["low"] - a["high"]
        bearish_gap = a["low"] - c["high"]

        if bullish_gap > min_points:
            out.at[out.index[i], "bullish_fvg"] = True
            out.at[out.index[i], "fvg_low"] = a["high"]
            out.at[out.index[i], "fvg_high"] = c["low"]

        if bearish_gap > min_points:
            out.at[out.index[i], "bearish_fvg"] = True
            out.at[out.index[i], "fvg_low"] = c["high"]
            out.at[out.index[i], "fvg_high"] = a["low"]

    return out
