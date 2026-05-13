import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from core.htf_context import HTFContextEngine


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

raw = pd.read_csv("data/history.csv")
features = FeatureEngine(config).build(raw)

htf_raw = pd.read_csv("data/history_h1.csv")
htf_context = HTFContextEngine(config).build(htf_raw)
features = HTFContextEngine(config).merge_to_ltf(features, htf_context)

checks = {
    "total_rows": len(features),
    "session_allowed": features["session_allowed"].sum(),
    "regime_allowed": features["regime_allowed"].sum(),
    "bullish_fvg_retest": features["bullish_fvg_retest"].sum(),
    "bearish_fvg_retest": features["bearish_fvg_retest"].sum(),
    "bullish_bos": features["bullish_bos"].sum(),
    "bearish_bos": features["bearish_bos"].sum(),
    "long_regime_ok": features["long_regime_ok"].sum(),
    "short_regime_ok": features["short_regime_ok"].sum(),
    "long_quality_ok": features["long_quality_ok"].sum(),
    "short_quality_ok": features["short_quality_ok"].sum(),
    "htf_discount": features["htf_discount"].sum(),
    "htf_premium": features["htf_premium"].sum(),
    "htf_bullish": (features["htf_trend"] == "bullish").sum(),
    "htf_bearish": (features["htf_trend"] == "bearish").sum(),
    "htf_neutral": (features["htf_trend"] == "neutral").sum(),
}

print("\n===== FILTER COUNTS =====")
for k, v in checks.items():
    print(k, ":", v)

base_long = (
    features["session_allowed"]
    & features["regime_allowed"]
    & features["bullish_fvg_retest"]
    & features["bullish_bos"]
    & features["long_regime_ok"]
    & features["long_quality_ok"]
)

base_short = (
    features["session_allowed"]
    & features["regime_allowed"]
    & features["bearish_fvg_retest"]
    & features["bearish_bos"]
    & features["short_regime_ok"]
    & features["short_quality_ok"]
)

print("\n===== BASE SETUPS BEFORE HTF =====")
print("base_long:", base_long.sum())
print("base_short:", base_short.sum())
print("base_total:", (base_long | base_short).sum())

htf_long = base_long & features["htf_discount"]
htf_short = base_short & features["htf_premium"]

print("\n===== AFTER PREMIUM/DISCOUNT =====")
print("htf_long:", htf_long.sum())
print("htf_short:", htf_short.sum())
print("htf_total:", (htf_long | htf_short).sum())

trend_long = htf_long & features["htf_trend"].isin(["bullish", "neutral"])
trend_short = htf_short & features["htf_trend"].isin(["bearish", "neutral"])

print("\n===== AFTER TREND =====")
print("trend_long:", trend_long.sum())
print("trend_short:", trend_short.sum())
print("trend_total:", (trend_long | trend_short).sum())