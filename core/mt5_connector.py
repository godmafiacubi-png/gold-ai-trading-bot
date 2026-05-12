from __future__ import annotations

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None


TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}


class MT5Connector:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.symbol = config["market"]["symbol"]
        self.timeframe = config["market"].get("timeframe", "M5")
        self.bars = int(config["market"].get("bars", 1000))

    def get_rates(self) -> pd.DataFrame:
        if mt5 is None:
            self.logger.warning("MetaTrader5 package not installed. Returning empty dataframe.")
            return pd.DataFrame()

        if not mt5.initialize():
            self.logger.error("MT5 initialize failed: %s", mt5.last_error())
            return pd.DataFrame()

        tf = TIMEFRAME_MAP.get(self.timeframe, TIMEFRAME_MAP["M5"])
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, self.bars)

        if rates is None:
            self.logger.error("MT5 data request failed: %s", mt5.last_error())
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        if df.empty:
            return df

        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
