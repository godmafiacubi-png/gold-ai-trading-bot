from __future__ import annotations

import pandas as pd


class HTFContextEngine:
    def __init__(self, config: dict):
        self.config = config

    def build(self, htf: pd.DataFrame) -> pd.DataFrame:
        out = htf.copy()

        out["time"] = pd.to_datetime(out["time"])

        out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
        out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
        out["htf_trend"] = "neutral"

        out.loc[out["ema_20"] > out["ema_50"], "htf_trend"] = "bullish"
        out.loc[out["ema_20"] < out["ema_50"], "htf_trend"] = "bearish"

        lookback = int(self.config["htf"].get("premium_discount_lookback", 100))

        out["htf_range_high"] = out["high"].rolling(lookback).max()
        out["htf_range_low"] = out["low"].rolling(lookback).min()
        out["htf_equilibrium"] = (out["htf_range_high"] + out["htf_range_low"]) / 2

        out["htf_premium"] = out["close"] > out["htf_equilibrium"]
        out["htf_discount"] = out["close"] < out["htf_equilibrium"]

        out["prev_high"] = out["high"].shift(1)
        out["prev_low"] = out["low"].shift(1)

        out["htf_bullish_sweep"] = (
            (out["low"] < out["prev_low"])
            & (out["close"] > out["prev_low"])
        )

        out["htf_bearish_sweep"] = (
            (out["high"] > out["prev_high"])
            & (out["close"] < out["prev_high"])
        )

        out["recent_htf_bullish_sweep"] = (
            out["htf_bullish_sweep"].rolling(5).max().fillna(0).astype(bool)
        )

        out["recent_htf_bearish_sweep"] = (
            out["htf_bearish_sweep"].rolling(5).max().fillna(0).astype(bool)
        )

        out["htf_bullish_bos"] = out["close"] > out["high"].rolling(20).max().shift(1)
        out["htf_bearish_bos"] = out["close"] < out["low"].rolling(20).min().shift(1)

        out["recent_htf_bullish_bos"] = (
            out["htf_bullish_bos"].rolling(5).max().fillna(0).astype(bool)
        )

        out["recent_htf_bearish_bos"] = (
            out["htf_bearish_bos"].rolling(5).max().fillna(0).astype(bool)
        )

        keep = [
            "time",
            "htf_trend",
            "htf_premium",
            "htf_discount",
            "recent_htf_bullish_sweep",
            "recent_htf_bearish_sweep",
            "recent_htf_bullish_bos",
            "recent_htf_bearish_bos",
        ]

        return out[keep].dropna().reset_index(drop=True)

    def merge_to_ltf(self, ltf: pd.DataFrame, htf_context: pd.DataFrame) -> pd.DataFrame:
        ltf_out = ltf.copy()
        ltf_out["time"] = pd.to_datetime(ltf_out["time"])

        htf_context = htf_context.copy()
        htf_context["time"] = pd.to_datetime(htf_context["time"])

        merged = pd.merge_asof(
            ltf_out.sort_values("time"),
            htf_context.sort_values("time"),
            on="time",
            direction="backward",
        )

        merged["htf_trend"] = merged["htf_trend"].fillna("neutral")

        bool_cols = [
            "htf_premium",
            "htf_discount",
            "recent_htf_bullish_sweep",
            "recent_htf_bearish_sweep",
            "recent_htf_bullish_bos",
            "recent_htf_bearish_bos",
        ]

        for col in bool_cols:
            merged[col] = merged[col].fillna(False).astype(bool)

        return merged.reset_index(drop=True)