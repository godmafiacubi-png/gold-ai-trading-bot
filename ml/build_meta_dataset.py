from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from core.htf_context import HTFContextEngine
from backtest.backtester import Backtester


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


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features(config: dict) -> pd.DataFrame:
    raw = pd.read_csv("data/history.csv")
    features = FeatureEngine(config).build(raw)

    if config.get("htf", {}).get("enabled", False):
        htf_path = Path("data/history_h1.csv")

        if not htf_path.exists():
            raise FileNotFoundError(
                "Missing data/history_h1.csv. Run: python export_mt5_htf_data.py"
            )

        htf_raw = pd.read_csv(htf_path)
        htf_engine = HTFContextEngine(config)
        htf_context = htf_engine.build(htf_raw)
        features = htf_engine.merge_to_ltf(features, htf_context)

    return features


def run_backtest_for_labels(config: dict, features: pd.DataFrame) -> pd.DataFrame:
    bt = Backtester(
        initial_balance=float(config["risk"].get("account_balance", 1000)),
        risk_per_trade_pct=float(config["risk"].get("risk_per_trade_pct", 1.0)),
        rr=1.5,
        sl_atr_mult=1.5,
        max_holding_bars=24,
        commission_per_lot=7.0,
        slippage_points_min=5,
        slippage_points_max=30,
        point_value=0.001,
        contract_size=100.0,
        use_meta_filter=False,
    )

    bt.run(features)

    trades_path = Path("logs/backtest_trades.csv")
    if not trades_path.exists():
        raise FileNotFoundError("Missing logs/backtest_trades.csv")

    return pd.read_csv(trades_path)


def normalize_record(record: dict) -> dict:
    # Convert bools to ints
    for key, value in list(record.items()):
        if isinstance(value, bool):
            record[key] = int(value)

    # Encode HTF trend
    trend = record.get("htf_trend", "neutral")
    record["htf_trend"] = {
        "bullish": 1,
        "bearish": -1,
        "neutral": 0,
    }.get(str(trend), 0)

    return record


def main() -> None:
    config = load_config()
    features = build_features(config)
    trades = run_backtest_for_labels(config, features)

    rows = []

    for _, trade in trades.iterrows():
        entry_time = str(trade["entry_time"])

        matched = features[features["time"].astype(str) == entry_time]

        if matched.empty:
            continue

        row = matched.iloc[0].copy()

        record = {}

        for col in FEATURE_COLUMNS:
            record[col] = row.get(col, 0)

        record["side"] = trade["side"]
        record["pnl"] = trade["pnl"]
        record["r_multiple"] = trade["r_multiple"]
        record["target"] = 1 if trade["pnl"] > 0 else 0

        record = normalize_record(record)
        rows.append(record)

    dataset = pd.DataFrame(rows)

    if dataset.empty:
        raise RuntimeError(
            "Meta dataset is empty. Check signal logic/backtest trades."
        )

    dataset["side"] = dataset["side"].map({
        "BUY": 1,
        "SELL": -1,
    }).fillna(0)

    for col in dataset.columns:
        if dataset[col].dtype == "bool":
            dataset[col] = dataset[col].astype(int)

    dataset = dataset.fillna(0)

    Path("data").mkdir(exist_ok=True)
    dataset.to_csv("data/meta_dataset.csv", index=False)

    print("Saved data/meta_dataset.csv")
    print(dataset.head())
    print("\nRows:", len(dataset))
    print("Win rate:", round(dataset["target"].mean() * 100, 2), "%")
    print("\nColumns:")
    print(list(dataset.columns))


if __name__ == "__main__":
    main()