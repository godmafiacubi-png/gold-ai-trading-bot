from __future__ import annotations

import pandas as pd


def add_session_filter(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    out = df.copy()

    time = pd.to_datetime(out["time"])
    hour = time.dt.hour

    out["session"] = "other"

    # Broker/server-time session map. Calibrate this later if broker time differs from UTC.
    london = (hour >= 7) & (hour < 13)
    new_york = (hour >= 13) & (hour < 22)
    overlap = (hour >= 13) & (hour < 17)

    out.loc[london, "session"] = "london"
    out.loc[new_york, "session"] = "new_york"
    out.loc[overlap, "session"] = "overlap"

    if config is not None:
        allowed_sessions = config.get("strategy", {}).get("allowed_sessions", [])
        session_switches = config.get("sessions", {})

        if not allowed_sessions:
            allowed_sessions = []
            if session_switches.get("london", True):
                allowed_sessions.append("london")
            if session_switches.get("new_york", True):
                allowed_sessions.append("new_york")
            if session_switches.get("overlap", True):
                allowed_sessions.append("overlap")
    else:
        allowed_sessions = ["london", "new_york", "overlap"]

    out["session_allowed"] = out["session"].isin(allowed_sessions)

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
        # Store FVG zones.
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

        # Bullish FVG retest: price touches zone and closes with bullish rejection.
        for zone in bullish_zones[-lookback:]:
            if i <= zone["created_index"]:
                continue

            touched = low <= zone["high"] and high >= zone["low"]
            rejected = close > open_

            if touched and rejected:
                out.at[i, "bullish_fvg_retest"] = True
                break

        # Bearish FVG retest: price touches zone and closes with bearish rejection.
        for zone in bearish_zones[-lookback:]:
            if i <= zone["created_index"]:
                continue

            touched = high >= zone["low"] and low <= zone["high"]
            rejected = close < open_

            if touched and rejected:
                out.at[i, "bearish_fvg_retest"] = True
                break

    return out
