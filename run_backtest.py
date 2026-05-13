import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from backtest.backtester import Backtester


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

df = pd.read_csv("data/history.csv")

features = FeatureEngine(config).build(df)

bt = Backtester(
    initial_balance=float(config["risk"].get("account_balance", 1000)),
    risk_per_trade_pct=float(config["risk"].get("risk_per_trade_pct", 1.0)),
    rr=2.0,
    sl_atr_mult=1.5,
    max_holding_bars=24,
    commission_per_lot=7.0,
    slippage_points_min=5,
    slippage_points_max=30,
    point_value=0.001,
    contract_size=100.0,
)

results = bt.run(features)

print("\n===== REALISTIC BACKTEST RESULTS =====")
for k, v in results.items():
    print(f"{k}: {v}")

print("\nExported:")
print("logs/backtest_trades.csv")
print("logs/equity_curve.csv")