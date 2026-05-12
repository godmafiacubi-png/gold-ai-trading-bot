from __future__ import annotations

import pandas as pd


class Backtester:
    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance

    def run(self, df: pd.DataFrame) -> dict:
        return {
            "initial_balance": self.initial_balance,
            "trades": 0,
            "net_profit": 0.0,
            "max_drawdown": 0.0,
            "note": "Backtest engine placeholder. Next step: add trade simulation loop.",
        }
