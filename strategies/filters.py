from __future__ import annotations

import pandas as pd


def add_session_filter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    time = pd.to_datetime(out["time"])
    hour = time.dt.hour

    out["session"] = "other"

    # ใช้เวลา server/broker ก่อน ยังไม่แปลง timezone
    out.loc[(hour >= 7) & (hour < 11), "session"] = "london"
    out.loc[(hour >= 12) & (hour < 17), "session"] = "new_york"

    out["session_allowed"] = out["session"].isin(["london", "new_york"])

    return out


def add_regime_filter(df: pd.DataFrame, min_atr_pct: float, max_atr_pct: float) -> pd.DataFrame:
    out = df.copy()

    out["atr_pct"] = out["atr"] / out["close"]

    out["regime_allowed"] = (
        (out["atr_pct"] >= min_atr_pct)
        & (out["atr_pct"] <= max_atr_pct)
    )

    return out


def add_fvg_retest(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    out = df.copy()

    out["bullish_fvg_retest"] = False
    out["bearish_fvg_retest"] = False

    bullish_zones = []
    bearish_zones = []

    for i, row in out.iterrows():
        # เก็บ FVG zone
        if bool(row.get("bullish_fvg")):
            bullish_zones.append({
                "created_index": i,
                "low": float(row.get("fvg_low", 0)),
                "high": float(row.get("fvg_high", 0)),
            })

        if bool(row.get("bearish_fvg")):
            bearish_zones.append({
                "created_index": i,
                "low": float(row.get("fvg_low", 0)),
                "high": float(row.get("fvg_high", 0)),
            })

        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        open_ = float(row["open"])

        # bullish FVG retest:
        # price กลับมาแตะ zone แล้วปิด bullish rejection
        for zone in bullish_zones[-lookback:]:
            if i <= zone["created_index"]:
                continue

            touched = low <= zone["high"] and high >= zone["low"]
            rejected = close > open_

            if touched and rejected:
                out.at[i, "bullish_fvg_retest"] = True
                break

        # bearish FVG retest:
        # price กลับมาแตะ zone แล้วปิด bearish rejection
        for zone in bearish_zones[-lookback:]:
            if i <= zone["created_index"]:
                continue

            touched = high >= zone["low"] and low <= zone["high"]
            rejected = close < open_

            if touched and rejected:
                out.at[i, "bearish_fvg_retest"] = True
                break

    return out