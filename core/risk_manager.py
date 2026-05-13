from __future__ import annotations


class RiskManager:
    def __init__(self, config: dict):
        self.config = config

    def prepare_order(self, signal: dict, latest_row) -> dict:
        if signal["side"] == "NONE":
            return {"action": "NO_TRADE", "reason": "No valid signal", "signal": signal}

        atr = float(latest_row.get("atr", 0))
        price = float(latest_row.get("close"))
        sl_atr_mult = float(signal.get("stop_loss_atr_mult", 1.5))
        tp_atr_mult = float(signal.get("take_profit_atr_mult", 3.0))
        risk_multiplier = float(signal.get("risk_multiplier", 1.0))
        max_lot = signal.get("max_lot")
        lot = self._calculate_lot(atr) * max(risk_multiplier, 0.0)
        if max_lot is not None:
            lot = min(lot, float(max_lot))
        lot = max(round(lot, 2), 0.01)

        if signal["side"] == "BUY":
            sl = price - atr * sl_atr_mult
            tp = price + atr * tp_atr_mult
        else:
            sl = price + atr * sl_atr_mult
            tp = price - atr * tp_atr_mult

        return {
            "action": "OPEN",
            "side": signal["side"],
            "symbol": self.config["market"]["symbol"],
            "lot": lot,
            "price": price,
            "sl": round(sl, 3),
            "tp": round(tp, 3),
            "confidence": signal["confidence"],
            "reasons": signal["reasons"],
            "risk_multiplier": risk_multiplier,
            "execution_plan": signal.get("execution_plan"),
            "hybrid_ai": signal.get("hybrid_ai"),
        }

    def _calculate_lot(self, atr: float) -> float:
        risk_cfg = self.config["risk"]
        if not risk_cfg.get("use_auto_lot", True):
            return float(risk_cfg.get("fixed_lot", 0.01))

        balance = float(risk_cfg.get("account_balance", 1000))
        risk_pct = float(risk_cfg.get("risk_per_trade_pct", 1.0)) / 100

        risk_amount = balance * risk_pct
        rough_lot = risk_amount / max(atr * 100, 1)
        return max(round(rough_lot, 2), 0.01)
