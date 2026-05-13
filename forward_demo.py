import time
from pathlib import Path
from datetime import datetime

import yaml
import joblib
import pandas as pd

from core.mt5_connector import MT5Connector
from core.feature_engine import FeatureEngine
from backtest.backtester import Backtester
from utils.logger import setup_logger


LOG_PATH = Path("logs/forward_demo_signals.csv")


def append_log(row: dict):
    LOG_PATH.parent.mkdir(exist_ok=True)

    df = pd.DataFrame([row])

    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger = setup_logger(config["app"].get("log_level", "INFO"))

    mt5 = MT5Connector(config, logger)
    feature_engine = FeatureEngine(config)

    meta_model = joblib.load("models/meta_filter.pkl")

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
        meta_threshold=0.60,
    )

    logger.info("Starting FORWARD DEMO LOGGER. No real orders will be sent.")

    last_signal_time = None

    while True:
        try:
            raw = mt5.get_rates()

            if raw.empty or len(raw) < 100:
                logger.warning("Not enough market data")
                time.sleep(60)
                continue

            features = feature_engine.build(raw)

            if features.empty:
                logger.warning("No features generated")
                time.sleep(60)
                continue

            row = features.iloc[-1]
            signal = bt._generate_signal(row)

            if signal == "NONE":
                logger.info("No signal | time=%s", row.get("time"))
                time.sleep(60)
                continue

            signal_time = str(row.get("time"))

            if signal_time == last_signal_time:
                logger.info("Signal already logged | time=%s", signal_time)
                time.sleep(60)
                continue

            probability = bt._meta_probability(row, signal)

            accepted = probability >= 0.60

            log_row = {
                "logged_at": datetime.utcnow().isoformat(),
                "signal_time": signal_time,
                "symbol": config["market"]["symbol"],
                "timeframe": config["market"]["timeframe"],
                "signal": signal,
                "meta_probability": round(probability, 4),
                "accepted": accepted,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "spread": row.get("spread"),
                "atr": row.get("atr"),
                "atr_pct": row.get("atr_pct"),
                "ema_slope": row.get("ema_slope"),
                "session": row.get("session"),
                "bullish_bos": row.get("bullish_bos"),
                "bearish_bos": row.get("bearish_bos"),
                "bullish_fvg_retest": row.get("bullish_fvg_retest"),
                "bearish_fvg_retest": row.get("bearish_fvg_retest"),
                "long_regime_ok": row.get("long_regime_ok"),
                "short_regime_ok": row.get("short_regime_ok"),
            }

            append_log(log_row)

            logger.info(
                "FORWARD SIGNAL | %s | prob=%.4f | accepted=%s",
                signal,
                probability,
                accepted,
            )

            last_signal_time = signal_time

            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Forward demo stopped by user")
            break

        except Exception as e:
            logger.exception("Forward demo error: %s", e)
            time.sleep(60)


if __name__ == "__main__":
    main()