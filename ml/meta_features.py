from __future__ import annotations

import pandas as pd


# Features used by the meta-filter before adding the trade side.
# Keep this list as the single source of truth for both dataset creation
# and live/backtest inference.
FEATURE_COLUMNS = [
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
]

MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["side"]
DROP_COLUMNS = ["target", "pnl", "r_multiple"]
MIN_META_DATASET_ROWS = 100


def encode_htf_trend(value: object) -> int:
    return {
        "bullish": 1,
        "bearish": -1,
        "neutral": 0,
    }.get(str(value), 0)


def normalize_meta_record(record: dict) -> dict:
    normalized = dict(record)

    for key, value in list(normalized.items()):
        if isinstance(value, bool):
            normalized[key] = int(value)

    normalized["htf_trend"] = encode_htf_trend(
        normalized.get("htf_trend", "neutral")
    )

    return normalized


def build_meta_feature_record(row: pd.Series, signal: str) -> dict:
    record = {col: row.get(col, 0) for col in FEATURE_COLUMNS}
    record["side"] = 1 if signal == "BUY" else -1
    return normalize_meta_record(record)
