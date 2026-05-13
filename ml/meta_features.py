from __future__ import annotations

import pandas as pd


# Single source of truth for the meta-filter model input columns.
# Keep HTF context here as ML features only; do not use them as hard filters.
META_FEATURE_COLUMNS = [
    # price / volatility
    "return",
    "range",
    "body",
    "atr",
    "atr_pct",
    "ema_slope",
    "sweep_count_5",
    "spread",

    # LTF structure
    "bullish_sweep",
    "bearish_sweep",
    "bullish_bos",
    "bearish_bos",
    "bullish_fvg_retest",
    "bearish_fvg_retest",
    "long_regime_ok",
    "short_regime_ok",
    "long_quality_ok",
    "short_quality_ok",

    # HTF context as ML features, not hard filters
    "htf_trend",
    "htf_premium",
    "htf_discount",
    "recent_htf_bullish_sweep",
    "recent_htf_bearish_sweep",
    "recent_htf_bullish_bos",
    "recent_htf_bearish_bos",

    # trade direction
    "side",
]

DROP_COLUMNS = ["target", "pnl", "r_multiple"]
MIN_META_DATASET_ROWS = 100


def encode_htf_trend(value: object) -> int:
    return {
        "bullish": 1,
        "bearish": -1,
        "neutral": 0,
    }.get(str(value), 0)


def encode_side(value: object) -> int:
    if isinstance(value, str):
        return {
            "BUY": 1,
            "SELL": -1,
        }.get(value.upper(), 0)

    return int(value) if pd.notna(value) else 0


def normalize_meta_record(record: dict) -> dict:
    normalized = dict(record)

    for key, value in list(normalized.items()):
        if isinstance(value, bool):
            normalized[key] = int(value)

    normalized["htf_trend"] = encode_htf_trend(
        normalized.get("htf_trend", "neutral")
    )
    normalized["side"] = encode_side(normalized.get("side", 0))

    for column in META_FEATURE_COLUMNS:
        normalized.setdefault(column, 0)

    return normalized


def build_meta_feature_record(row: pd.Series, signal: str) -> dict:
    record = {col: row.get(col, 0) for col in META_FEATURE_COLUMNS}
    record["side"] = 1 if signal == "BUY" else -1
    return normalize_meta_record(record)
