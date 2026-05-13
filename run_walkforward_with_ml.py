import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from core.htf_context import HTFContextEngine
from backtest.backtester import Backtester


THRESHOLD = 0.60


def run_fold(config, fold_df, fold_name):
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
        use_meta_filter=True,
        meta_model_path="models/meta_filter.pkl",
        meta_threshold=THRESHOLD,
    )

    result = bt.run(fold_df)
    result["fold"] = fold_name
    return result


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw = pd.read_csv("data/history.csv")
    features = FeatureEngine(config).build(raw)

    if config.get("htf", {}).get("enabled", False):
        htf_raw = pd.read_csv("data/history_h1.csv")
        htf_context = HTFContextEngine(config).build(htf_raw)
        features = HTFContextEngine(config).merge_to_ltf(features, htf_context)

    folds = 3
    chunk_size = len(features) // folds

    results = []

    for i in range(folds):
        start = i * chunk_size

        if i < folds - 1:
            end = (i + 1) * chunk_size
        else:
            end = len(features)

        fold_df = features.iloc[start:end].copy()

        result = run_fold(
            config=config,
            fold_df=fold_df,
            fold_name=f"fold_{i+1}",
        )

        results.append(result)

    df = pd.DataFrame(results)

    cols = [
        "fold",
        "trades",
        "winrate",
        "profit_factor",
        "net_profit",
        "max_drawdown_pct",
        "expectancy_per_trade",
        "avg_r",
        "skipped_by_meta",
        "avg_meta_probability",
    ]

    print("\n===== WALK FORWARD WITH ML =====")
    print(df[cols].to_string(index=False))

    print("\n===== SUMMARY =====")

    print("Average PF:", round(df["profit_factor"].mean(), 2))
    print("Average DD:", round(df["max_drawdown_pct"].mean(), 2))
    print("Total Net:", round(df["net_profit"].sum(), 2))
    print("Total Trades:", int(df["trades"].sum()))

    profitable = (df["net_profit"] > 0).sum()

    print(
        "Profitable folds:",
        profitable,
        "/",
        len(df),
    )


if __name__ == "__main__":
    main()