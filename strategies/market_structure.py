from __future__ import annotations

import pandas as pd


def detect_swings(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    lookback = int(config["strategy"].get("swing_lookback", 3))
    out["swing_high"] = False
    out["swing_low"] = False

    for i in range(lookback, len(out) - lookback):
        window = out.iloc[i - lookback:i + lookback + 1]
        if out.iloc[i]["high"] == window["high"].max():
            out.at[out.index[i], "swing_high"] = True
        if out.iloc[i]["low"] == window["low"].min():
            out.at[out.index[i], "swing_low"] = True

    return out


def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bullish_bos"] = False
    out["bearish_bos"] = False

    last_swing_high = None
    last_swing_low = None

    for i, row in out.iterrows():
        if row.get("swing_high"):
            last_swing_high = row["high"]
        if row.get("swing_low"):
            last_swing_low = row["low"]

        if last_swing_high is not None and row["close"] > last_swing_high:
            out.at[i, "bullish_bos"] = True
        if last_swing_low is not None and row["close"] < last_swing_low:
            out.at[i, "bearish_bos"] = True

    return out
