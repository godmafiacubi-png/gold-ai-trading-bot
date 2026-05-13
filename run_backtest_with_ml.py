import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from backtest.backtester import Backtester


def run(threshold):
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw = pd.read_csv("data/history.csv")
    features = FeatureEngine(config).build(raw)

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
        meta_threshold=threshold,
    )

    result = bt.run(features)
    result["threshold"] = threshold
    return result


if __name__ == "__main__":
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n===== ML META FILTER BACKTEST =====")

    for t in thresholds:
        result = run(t)

        print(
            f"threshold={t} | "
            f"trades={result.get('trades')} | "
            f"winrate={result.get('winrate')} | "
            f"PF={result.get('profit_factor')} | "
            f"net={result.get('net_profit')} | "
            f"DD={result.get('max_drawdown_pct')} | "
            f"exp={result.get('expectancy_per_trade')}"
        )