from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from core.htf_context import HTFContextEngine
from backtest.backtester import Backtester


THRESHOLD = 0.60
RR = 1.5
OUT_DIR = Path("logs/diagnostics")


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features(config: dict) -> pd.DataFrame:
    raw = pd.read_csv("data/history.csv")
    features = FeatureEngine(config).build(raw)

    if config.get("htf", {}).get("enabled", False):
        htf_raw = pd.read_csv("data/history_h1.csv")
        htf_engine = HTFContextEngine(config)
        htf_context = htf_engine.build(htf_raw)
        features = htf_engine.merge_to_ltf(features, htf_context)

    return features


def run_ml_backtest(config: dict, features: pd.DataFrame) -> pd.DataFrame:
    bt = Backtester(
        initial_balance=float(config["risk"].get("account_balance", 1000)),
        risk_per_trade_pct=float(config["risk"].get("risk_per_trade_pct", 1.0)),
        rr=RR,
        sl_atr_mult=1.5,
        max_holding_bars=24,
        commission_per_lot=7.0,
        slippage_points_min=5,
        slippage_points_max=30,
        point_value=0.001,
        contract_size=100.0,
        use_meta_filter=True,
        meta_model_path="models/meta_filter.pkl",
        meta_threshold=THRESHOLD,
        config=config,
    )

    bt.run(features)
    trades = pd.read_csv("logs/backtest_trades.csv")
    return trades


def probability_bucket_report(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    out = trades.copy()
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    labels = [
        "<0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.80",
        "0.80+",
    ]
    out["probability_bucket"] = pd.cut(
        out["meta_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    out["target"] = (out["pnl"] > 0).astype(int)

    rows = []
    for bucket, group in out.groupby("probability_bucket", observed=False):
        if group.empty:
            continue

        wins = group[group["pnl"] > 0]
        losses = group[group["pnl"] < 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
        avg_predicted = group["meta_probability"].mean()
        actual_winrate = group["target"].mean()
        brier = ((group["meta_probability"] - group["target"]) ** 2).mean()

        rows.append({
            "probability_bucket": str(bucket),
            "trades": int(len(group)),
            "avg_predicted_probability": round(avg_predicted, 4),
            "actual_winrate": round(actual_winrate, 4),
            "calibration_error": round(avg_predicted - actual_winrate, 4),
            "brier_score": round(brier, 4),
            "profit_factor": round(pf, 2),
            "net_profit": round(group["pnl"].sum(), 2),
            "expectancy": round(group["pnl"].mean(), 2),
            "avg_r": round(group["r_multiple"].mean(), 3),
        })

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    features = build_features(config)
    trades = run_ml_backtest(config, features)

    report = probability_bucket_report(trades)

    print("\n===== PROBABILITY CALIBRATION REPORT =====")
    if report.empty:
        print("No trades")
        return

    print(report.to_string(index=False))
    report.to_csv(OUT_DIR / "probability_calibration.csv", index=False)

    overall_brier = ((trades["meta_probability"] - (trades["pnl"] > 0).astype(int)) ** 2).mean()
    print("\nOverall Brier score:", round(overall_brier, 4))
    print("Exported:", OUT_DIR / "probability_calibration.csv")


if __name__ == "__main__":
    main()
