from __future__ import annotations

import pandas as pd


class ICTSMCStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.min_confidence = float(config["strategy"].get("min_confidence", 0.65))

    def generate_signal(self, df: pd.DataFrame) -> dict:
        row = df.iloc[-1]

        buy_score = 0.0
        sell_score = 0.0
        reasons = []

        if row.get("bullish_fvg"):
            buy_score += 0.3
            reasons.append("bullish_fvg")
        if row.get("bullish_bos"):
            buy_score += 0.3
            reasons.append("bullish_bos")
        if row.get("bullish_sweep"):
            buy_score += 0.2
            reasons.append("bullish_liquidity_sweep")

        if row.get("bearish_fvg"):
            sell_score += 0.3
            reasons.append("bearish_fvg")
        if row.get("bearish_bos"):
            sell_score += 0.3
            reasons.append("bearish_bos")
        if row.get("bearish_sweep"):
            sell_score += 0.2
            reasons.append("bearish_liquidity_sweep")

        if buy_score >= self.min_confidence and buy_score > sell_score:
            return {"side": "BUY", "confidence": buy_score, "reasons": reasons}

        if sell_score >= self.min_confidence and sell_score > buy_score:
            return {"side": "SELL", "confidence": sell_score, "reasons": reasons}

        return {"side": "NONE", "confidence": max(buy_score, sell_score), "reasons": reasons}
