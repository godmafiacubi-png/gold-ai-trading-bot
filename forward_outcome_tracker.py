from pathlib import Path
import yaml
import pandas as pd

from core.mt5_connector import MT5Connector
from core.feature_engine import FeatureEngine
from backtest.backtester import Backtester
from utils.logger import setup_logger


SIGNAL_LOG = Path("logs/forward_demo_signals.csv")
OUTCOME_LOG = Path("logs/forward_demo_outcomes.csv")


def load_existing_outcomes():
    if OUTCOME_LOG.exists():
        return pd.read_csv(OUTCOME_LOG)
    return pd.DataFrame()


def main():
    if not SIGNAL_LOG.exists():
        print("No forward signal log found.")
        return

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config["app"].get("log_level", "INFO"))

    signals = pd.read_csv(SIGNAL_LOG)

    # track เฉพาะ signal ที่ ML accept
    signals = signals[signals["accepted"] == True].copy()

    if signals.empty:
        print("No accepted forward signals yet.")
        return

    existing = load_existing_outcomes()
    tracked_times = set(existing["signal_time"].astype(str)) if not existing.empty else set()

    mt5 = MT5Connector(config, logger)
    raw = mt5.get_rates()

    if raw.empty:
        print("No MT5 data.")
        return

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
        use_meta_filter=False,
    )

    results = []

    for _, sig in signals.iterrows():
        signal_time = str(sig["signal_time"])

        if signal_time in tracked_times:
            continue

        matched = features[features["time"].astype(str) == signal_time]

        if matched.empty:
            print(f"Signal candle not found yet: {signal_time}")
            continue

        idx = matched.index[0]

        # ต้องมีแท่งอนาคตพอสำหรับ max_holding_bars
        if idx + 25 >= len(features):
            print(f"Not enough future bars yet: {signal_time}")
            continue

        row = features.iloc[idx]
        side = sig["signal"]

        next_row = features.iloc[idx + 1]
        atr = float(row.get("atr", 0))

        if atr <= 0:
            continue

        spread = bt._get_spread_price(next_row)
        slippage = 0.0
        raw_open_bid = float(next_row["open"])

        if side == "BUY":
            entry = raw_open_bid + spread + slippage
            sl = raw_open_bid - atr * 1.5
            risk_distance = abs(entry - sl)
            tp = entry + risk_distance * 1.5
        else:
            entry = raw_open_bid - slippage
            sl = raw_open_bid + spread + atr * 1.5
            risk_distance = abs(sl - entry)
            tp = entry - risk_distance * 1.5

        lot = bt._calculate_lot(10, risk_distance)

        future = features.iloc[idx + 1 : idx + 25]

        trade = bt._simulate_trade(
            side=side,
            future=future,
            entry=entry,
            sl=sl,
            tp=tp,
            lot=lot,
            risk_amount=10,
            entry_spread=spread,
            entry_slippage=slippage,
            balance_before=1000,
            meta_probability=float(sig["meta_probability"]),
        )

        results.append({
            "signal_time": signal_time,
            "signal": side,
            "meta_probability": sig["meta_probability"],
            "entry": trade.entry,
            "exit": trade.exit,
            "sl": trade.sl,
            "tp": trade.tp,
            "result": trade.result,
            "pnl": trade.pnl,
            "r_multiple": trade.r_multiple,
            "bars_held": trade.bars_held,
            "mae": trade.mae,
            "mfe": trade.mfe,
            "exit_time": trade.exit_time,
        })

    if not results:
        print("No completed new outcomes.")
        return

    new_df = pd.DataFrame(results)

    if OUTCOME_LOG.exists():
        old = pd.read_csv(OUTCOME_LOG)
        final = pd.concat([old, new_df], ignore_index=True)
    else:
        final = new_df

    OUTCOME_LOG.parent.mkdir(exist_ok=True)
    final.to_csv(OUTCOME_LOG, index=False)

    print("Saved:", OUTCOME_LOG)
    print(new_df)


if __name__ == "__main__":
    main()