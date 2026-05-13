import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from backtest.backtester import Backtester
from core.htf_context import HTFContextEngine


def run_case(features, config, name, params):
    bt = Backtester(
        initial_balance=float(config["risk"].get("account_balance", 1000)),
        risk_per_trade_pct=float(config["risk"].get("risk_per_trade_pct", 1.0)),
        rr=params.get("rr", 2.0),
        sl_atr_mult=params.get("sl_atr_mult", 1.5),
        max_holding_bars=params.get("max_holding_bars", 24),
        commission_per_lot=params.get("commission_per_lot", 7.0),
        slippage_points_min=params.get("slippage_points_min", 5),
        slippage_points_max=params.get("slippage_points_max", 30),
        point_value=params.get("point_value", 0.001),
        contract_size=params.get("contract_size", 100.0),
        seed=params.get("seed", 42),
    )

    result = bt.run(features)
    result["case"] = name
    return result


def apply_spread_multiplier(df, multiplier):
    out = df.copy()
    if "spread" in out.columns:
        out["spread"] = out["spread"] * multiplier
    return out


def stress_test(features, config):
    cases = [
        {
            "name": "base",
            "spread_mult": 1.0,
            "params": {
                "rr": 2.0,
                "slippage_points_min": 5,
                "slippage_points_max": 30,
            },
        },
        {
            "name": "spread_x1_5",
            "spread_mult": 1.5,
            "params": {
                "rr": 2.0,
                "slippage_points_min": 5,
                "slippage_points_max": 30,
            },
        },
        {
            "name": "spread_x2",
            "spread_mult": 2.0,
            "params": {
                "rr": 2.0,
                "slippage_points_min": 5,
                "slippage_points_max": 30,
            },
        },
        {
            "name": "slippage_30_80",
            "spread_mult": 1.0,
            "params": {
                "rr": 2.0,
                "slippage_points_min": 30,
                "slippage_points_max": 80,
            },
        },
        {
            "name": "harsh_cost",
            "spread_mult": 2.0,
            "params": {
                "rr": 2.0,
                "slippage_points_min": 30,
                "slippage_points_max": 80,
            },
        },
        {
            "name": "rr_1_5",
            "spread_mult": 1.0,
            "params": {
                "rr": 1.5,
                "slippage_points_min": 5,
                "slippage_points_max": 30,
            },
        },
        {
            "name": "rr_2_5",
            "spread_mult": 1.0,
            "params": {
                "rr": 2.5,
                "slippage_points_min": 5,
                "slippage_points_max": 30,
            },
        },
    ]

    results = []

    for case in cases:
        case_features = apply_spread_multiplier(features, case["spread_mult"])
        result = run_case(
            case_features,
            config,
            case["name"],
            case["params"],
        )
        results.append(result)

    return pd.DataFrame(results)


def walk_forward_test(features, config, folds=5):
    results = []

    chunk_size = len(features) // folds

    for i in range(folds):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < folds - 1 else len(features)

        fold_df = features.iloc[start:end].copy()

        if len(fold_df) < 50:
            continue

        result = run_case(
            fold_df,
            config,
            f"walk_forward_fold_{i + 1}",
            {
                "rr": 2.0,
                "slippage_points_min": 5,
                "slippage_points_max": 30,
            },
        )

        result["start_time"] = fold_df.iloc[0].get("time", start)
        result["end_time"] = fold_df.iloc[-1].get("time", end)
        results.append(result)

    return pd.DataFrame(results)


def print_table(title, df):
    print(f"\n===== {title} =====")

    if df.empty:
        print("No results")
        return

    cols = [
        "case",
        "trades",
        "winrate",
        "profit_factor",
        "net_profit",
        "max_drawdown_pct",
        "expectancy_per_trade",
        "avg_r",
    ]

    existing_cols = [c for c in cols if c in df.columns]
    print(df[existing_cols].to_string(index=False))


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw = pd.read_csv("data/history.csv")
    features = FeatureEngine(config).build(raw)

    if config.get("htf", {}).get("enabled", False):
         htf_raw = pd.read_csv("data/history_h1.csv")
         htf_context = HTFContextEngine(config).build(htf_raw)
         features = HTFContextEngine(config).merge_to_ltf(features, htf_context)

    stress_df = stress_test(features, config)
    walk_df = walk_forward_test(features, config, folds=3)

    stress_df.to_csv("logs/stress_test_results.csv", index=False)
    walk_df.to_csv("logs/walk_forward_results.csv", index=False)

    print_table("STRESS TEST", stress_df)
    print_table("WALK FORWARD", walk_df)

    print("\nExported:")
    print("logs/stress_test_results.csv")
    print("logs/walk_forward_results.csv")


if __name__ == "__main__":
    main()