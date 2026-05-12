from __future__ import annotations

import pandas as pd


def detect_regime(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return "UNKNOWN"

    recent_vol = df["return"].tail(20).std()
    base_vol = df["return"].tail(100).std()

    if recent_vol > base_vol * 1.5:
        return "HIGH_VOL"
    if recent_vol < base_vol * 0.7:
        return "LOW_VOL"
    return "NORMAL"
