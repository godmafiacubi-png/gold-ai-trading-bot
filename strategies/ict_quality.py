from __future__ import annotations

import pandas as pd


def add_ict_quality_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()

    s = config["strategy"]

    body_avg_period = int(s.get("body_avg_period", 20))
    min_body_atr_ratio = float(s.get("min_body_atr_ratio", 0.35))
    min_rejection_wick_ratio = float(s.get("min_rejection_wick_ratio", 0.35))
    min_close_strength_sell = float(s.get("min_close_strength_sell", 0.45))
    min_close_strength_buy = float(s.get("min_close_strength_buy", 0.55))

    out["candle_range"] = out["high"] - out["low"]
    out["candle_body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    out["body_avg"] = out["candle_body"].rolling(body_avg_period).mean()

    out["body_atr_ratio"] = out["candle_body"] / out["atr"]
    out["body_expansion"] = out["candle_body"] > out["body_avg"]

    # close position inside candle range: 0 = low, 1 = high
    out["close_position"] = (
        (out["close"] - out["low"]) / out["candle_range"].replace(0, 1)
    )

    out["upper_wick_ratio"] = out["upper_wick"] / out["candle_range"].replace(0, 1)
    out["lower_wick_ratio"] = out["lower_wick"] / out["candle_range"].replace(0, 1)

    # BUY quality:
    # - price rejects lower area
    # - close in upper part
    # - body meaningful vs ATR
    # - candle has expansion
    out["bullish_rejection_quality"] = (
        (out["lower_wick_ratio"] >= min_rejection_wick_ratio)
        & (out["close_position"] >= min_close_strength_buy)
        & (out["body_atr_ratio"] >= min_body_atr_ratio)
        & out["body_expansion"]
        & (out["close"] > out["open"])
    )

    # SELL quality:
    # - price rejects upper area
    # - close in lower part
    # - body meaningful vs ATR
    # - candle has expansion
    out["bearish_rejection_quality"] = (
        (out["upper_wick_ratio"] >= min_rejection_wick_ratio)
        & (out["close_position"] <= min_close_strength_sell)
        & (out["body_atr_ratio"] >= min_body_atr_ratio)
        & out["body_expansion"]
        & (out["close"] < out["open"])
    )

    # Session-specific tightening
    # NY gold is often violent/fakeout-heavy, so require stronger candle quality.
    out["ny_quality_ok_buy"] = True
    out["ny_quality_ok_sell"] = True

    is_ny = out.get("session", "other") == "new_york"

    out.loc[is_ny, "ny_quality_ok_buy"] = (
        out.loc[is_ny, "body_atr_ratio"] >= min_body_atr_ratio * 1.05
    )

    out.loc[is_ny, "ny_quality_ok_sell"] = (
        out.loc[is_ny, "body_atr_ratio"] >= min_body_atr_ratio * 1.05
    )

    out["long_quality_ok"] = (
        out["bullish_rejection_quality"]
        & out["ny_quality_ok_buy"]
    )

    out["short_quality_ok"] = (
        out["bearish_rejection_quality"]
        & out["ny_quality_ok_sell"]
    )

    return out