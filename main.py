import argparse
import yaml

from utils.logger import setup_logger
from core.mt5_connector import MT5Connector
from core.feature_engine import FeatureEngine
from strategies.ict_smc_strategy import ICTSMCStrategy
from core.risk_manager import RiskManager
from core.execution_engine import ExecutionEngine


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger(config["app"].get("log_level", "INFO"))

    logger.info("Starting Gold AI Trading Bot in %s mode", config["app"]["mode"])

    mt5 = MT5Connector(config, logger)
    data = mt5.get_rates()

    if data.empty:
        logger.warning("No market data loaded. Check MT5 terminal, symbol, and broker suffix.")
        return

    features = FeatureEngine(config).build(data)
    signal = ICTSMCStrategy(config).generate_signal(features)

    risk = RiskManager(config)
    order = risk.prepare_order(signal, features.iloc[-1])

    executor = ExecutionEngine(config, logger)
    result = executor.execute(order)

    logger.info("Final result: %s", result)


if __name__ == "__main__":
    main()
