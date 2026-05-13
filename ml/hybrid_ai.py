from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class RegimeState:
    label: str
    confidence: float
    volatility_state: str
    trend_state: str
    reasons: list[str]


@dataclass
class AnomalyState:
    is_anomaly: bool
    score: float
    reasons: list[str]


@dataclass
class VolatilityForecast:
    current_atr: float
    forecast_atr: float
    atr_pct: float
    stop_loss_atr_mult: float
    take_profit_atr_mult: float
    position_size_multiplier: float
    reasons: list[str]


@dataclass
class RiskDecision:
    approved: bool
    risk_multiplier: float
    max_lot: float | None
    reasons: list[str]


@dataclass
class ExecutionPlan:
    policy: str
    aggressiveness: float
    max_slippage_points: float
    order_type: str
    reasons: list[str]


class RegimeClassifier:
    """Hybrid market-regime classifier using optional model inference plus stable rules."""

    def __init__(self, config: dict):
        hybrid_cfg = config.get("hybrid_ai", {})
        self.enabled = bool(hybrid_cfg.get("enabled", False))
        self.model_path = hybrid_cfg.get("regime_model_path")
        self.model = self._load_model(self.model_path)

    def classify(self, df: pd.DataFrame) -> RegimeState:
        if len(df) < 30:
            return RegimeState("UNKNOWN", 0.0, "UNKNOWN", "UNKNOWN", ["not_enough_bars"])

        row = df.iloc[-1]
        recent = df.tail(50)
        returns = recent["return"].replace([np.inf, -np.inf], np.nan).dropna()
        atr_pct = float(row.get("atr_pct", row.get("atr", 0) / max(row.get("close", 1), 1)))
        vol_now = float(returns.tail(20).std() or 0)
        vol_base = float(returns.std() or 0)
        ema_slope = float(row.get("ema_slope", recent["close"].ewm(span=20).mean().diff().iloc[-1]))

        volatility_state = "NORMAL_VOL"
        if vol_base > 0 and vol_now > vol_base * 1.5:
            volatility_state = "HIGH_VOL"
        elif vol_base > 0 and vol_now < vol_base * 0.7:
            volatility_state = "LOW_VOL"

        trend_threshold = max(float(row.get("atr", 0)) * 0.03, abs(float(row.get("close", 0))) * 0.00002)
        if ema_slope > trend_threshold:
            trend_state = "TREND_UP"
        elif ema_slope < -trend_threshold:
            trend_state = "TREND_DOWN"
        else:
            trend_state = "RANGE"

        label = f"{trend_state}_{volatility_state}"
        confidence = 0.55
        reasons = [f"volatility={volatility_state}", f"trend={trend_state}", f"atr_pct={atr_pct:.6f}"]

        model_label, model_confidence = self._predict_with_model(row)
        if model_label:
            label = model_label
            confidence = max(confidence, model_confidence)
            reasons.append("model_regime_override")
        else:
            if trend_state != "RANGE":
                confidence += 0.2
            if volatility_state != "NORMAL_VOL":
                confidence += 0.1

        return RegimeState(label, round(min(confidence, 0.99), 4), volatility_state, trend_state, reasons)

    def _predict_with_model(self, row: pd.Series) -> tuple[str | None, float]:
        if self.model is None:
            return None, 0.0

        feature_row = self._feature_frame(row)
        if feature_row.empty:
            return None, 0.0

        prediction = self.model.predict(feature_row)[0]
        confidence = 0.65
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(feature_row)[0]
            confidence = float(np.max(probabilities))
        return str(prediction), confidence

    def _feature_frame(self, row: pd.Series) -> pd.DataFrame:
        cols = ["return", "range", "body", "atr", "atr_pct", "ema_slope", "sweep_count_5"]
        return pd.DataFrame([{col: row.get(col, 0) for col in cols}]).fillna(0)

    @staticmethod
    def _load_model(model_path: str | None) -> Any | None:
        if not model_path:
            return None
        path = Path(model_path)
        if not path.exists():
            return None
        return joblib.load(path)


class BoostingSignalScorer:
    """Scores a base ICT/SMC setup with an optional boosting model and deterministic fallback."""

    def __init__(self, config: dict):
        hybrid_cfg = config.get("hybrid_ai", {})
        self.model_path = hybrid_cfg.get("signal_model_path")
        self.threshold = float(hybrid_cfg.get("min_signal_score", config.get("strategy", {}).get("min_confidence", 0.65)))
        self.model = RegimeClassifier._load_model(self.model_path)

    def score(self, df: pd.DataFrame, base_signal: dict, regime: RegimeState) -> tuple[float, list[str]]:
        if base_signal.get("side") == "NONE" or df.empty:
            return float(base_signal.get("confidence", 0.0)), ["base_signal_none"]

        row = df.iloc[-1]
        reasons = ["base_confidence_weighted"]
        fallback_score = self._fallback_score(row, base_signal, regime)

        if self.model is None:
            return round(fallback_score, 4), reasons + ["boosting_model_missing_fallback_rules"]

        feature_row = self._feature_frame(row, base_signal, regime)
        if hasattr(self.model, "predict_proba"):
            model_score = float(self.model.predict_proba(feature_row)[0][1])
        else:
            prediction = self.model.predict(feature_row)[0]
            model_score = float(prediction)

        blended = (model_score * 0.65) + (fallback_score * 0.35)
        return round(blended, 4), reasons + ["boosting_model_score"]

    def _fallback_score(self, row: pd.Series, base_signal: dict, regime: RegimeState) -> float:
        score = float(base_signal.get("confidence", 0.0))
        side = base_signal.get("side")

        if side == "BUY" and regime.trend_state == "TREND_UP":
            score += 0.12
        if side == "SELL" and regime.trend_state == "TREND_DOWN":
            score += 0.12
        if regime.volatility_state == "HIGH_VOL":
            score -= 0.08
        if bool(row.get("session_allowed", True)):
            score += 0.04
        if side == "BUY" and bool(row.get("long_quality_ok", False)):
            score += 0.08
        if side == "SELL" and bool(row.get("short_quality_ok", False)):
            score += 0.08
        if bool(row.get("bullish_sweep", False)) or bool(row.get("bearish_sweep", False)):
            score += 0.04

        return min(max(score, 0.0), 0.99)

    def _feature_frame(self, row: pd.Series, base_signal: dict, regime: RegimeState) -> pd.DataFrame:
        cols = {
            "return": row.get("return", 0),
            "range": row.get("range", 0),
            "body": row.get("body", 0),
            "atr": row.get("atr", 0),
            "atr_pct": row.get("atr_pct", 0),
            "ema_slope": row.get("ema_slope", 0),
            "sweep_count_5": row.get("sweep_count_5", 0),
            "side": 1 if base_signal.get("side") == "BUY" else -1,
            "regime_confidence": regime.confidence,
            "is_high_vol": int(regime.volatility_state == "HIGH_VOL"),
        }
        return pd.DataFrame([cols]).fillna(0)


class AnomalyDetector:
    """Detects abnormal bars/spreads so the engine can stand down before execution."""

    def __init__(self, config: dict):
        hybrid_cfg = config.get("hybrid_ai", {})
        self.z_threshold = float(hybrid_cfg.get("anomaly_z_threshold", 3.0))
        self.lookback = int(hybrid_cfg.get("anomaly_lookback", 100))

    def detect(self, df: pd.DataFrame) -> AnomalyState:
        if len(df) < max(30, self.lookback // 2):
            return AnomalyState(False, 0.0, ["not_enough_bars_for_anomaly"])

        recent = df.tail(self.lookback)
        row = recent.iloc[-1]
        checks = {
            "return_z": self._z_score(row.get("return", 0), recent["return"]),
            "range_z": self._z_score(row.get("range", 0), recent["range"]),
        }
        if "spread" in recent.columns:
            checks["spread_z"] = self._z_score(row.get("spread", 0), recent["spread"])

        score = max(abs(v) for v in checks.values()) if checks else 0.0
        reasons = [f"{name}={value:.2f}" for name, value in checks.items()]
        is_anomaly = score >= self.z_threshold
        if is_anomaly:
            reasons.append("anomaly_threshold_exceeded")
        return AnomalyState(is_anomaly, round(float(score), 4), reasons)

    @staticmethod
    def _z_score(value: float, series: pd.Series) -> float:
        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        std = float(clean.std() or 0)
        if std == 0:
            return 0.0
        return (float(value or 0) - float(clean.mean())) / std


class VolatilityModel:
    """EWMA volatility model that adapts stop, target, and sizing multipliers."""

    def __init__(self, config: dict):
        hybrid_cfg = config.get("hybrid_ai", {})
        self.ewm_span = int(hybrid_cfg.get("volatility_ewm_span", 20))
        self.base_sl_mult = float(hybrid_cfg.get("base_sl_atr_mult", 1.5))
        self.base_tp_mult = float(hybrid_cfg.get("base_tp_atr_mult", 3.0))

    def forecast(self, df: pd.DataFrame, regime: RegimeState) -> VolatilityForecast:
        row = df.iloc[-1]
        atr = float(row.get("atr", 0) or 0)
        close = max(float(row.get("close", 1) or 1), 1)
        atr_series = df["atr"].replace([np.inf, -np.inf], np.nan).dropna().tail(max(self.ewm_span, 5))
        forecast_atr = float(atr_series.ewm(span=self.ewm_span).mean().iloc[-1]) if not atr_series.empty else atr
        vol_ratio = forecast_atr / max(atr, 1e-9)
        atr_pct = atr / close

        sl_mult = self.base_sl_mult * min(max(vol_ratio, 0.85), 1.35)
        tp_mult = self.base_tp_mult
        size_mult = 1.0
        reasons = [f"forecast_atr={forecast_atr:.5f}", f"vol_ratio={vol_ratio:.3f}"]

        if regime.volatility_state == "HIGH_VOL":
            sl_mult *= 1.15
            tp_mult *= 0.9
            size_mult *= 0.65
            reasons.append("high_vol_defensive_sizing")
        elif regime.volatility_state == "LOW_VOL":
            sl_mult *= 0.9
            size_mult *= 0.85
            reasons.append("low_vol_smaller_breakout_size")

        return VolatilityForecast(
            current_atr=round(atr, 5),
            forecast_atr=round(forecast_atr, 5),
            atr_pct=round(atr_pct, 6),
            stop_loss_atr_mult=round(sl_mult, 4),
            take_profit_atr_mult=round(tp_mult, 4),
            position_size_multiplier=round(size_mult, 4),
            reasons=reasons,
        )


class RiskAI:
    """AI-style risk governor that can block or reduce trades before order creation."""

    def __init__(self, config: dict):
        self.config = config
        hybrid_cfg = config.get("hybrid_ai", {})
        risk_cfg = config.get("risk", {})
        self.max_anomaly_score = float(hybrid_cfg.get("max_anomaly_score", 3.0))
        self.max_lot = risk_cfg.get("max_lot")
        self.min_regime_confidence = float(hybrid_cfg.get("min_regime_confidence", 0.45))

    def decide(
        self,
        signal_score: float,
        regime: RegimeState,
        anomaly: AnomalyState,
        volatility: VolatilityForecast,
    ) -> RiskDecision:
        reasons: list[str] = []
        if anomaly.is_anomaly or anomaly.score > self.max_anomaly_score:
            return RiskDecision(False, 0.0, self._max_lot(), ["blocked_by_anomaly"] + anomaly.reasons)

        if regime.confidence < self.min_regime_confidence:
            return RiskDecision(False, 0.0, self._max_lot(), ["blocked_by_low_regime_confidence"])

        multiplier = volatility.position_size_multiplier
        if signal_score < 0.75:
            multiplier *= 0.75
            reasons.append("reduced_size_for_marginal_score")
        if regime.volatility_state == "HIGH_VOL":
            multiplier *= 0.8
            reasons.append("risk_ai_high_vol_reduction")

        multiplier = round(min(max(multiplier, 0.0), 1.25), 4)
        reasons.append(f"risk_multiplier={multiplier}")
        return RiskDecision(True, multiplier, self._max_lot(), reasons)

    def _max_lot(self) -> float | None:
        if self.max_lot is None:
            return None
        return float(self.max_lot)


class RLExecutionPolicy:
    """Execution policy layer with optional offline Q-table and safe rule fallback."""

    def __init__(self, config: dict):
        hybrid_cfg = config.get("hybrid_ai", {})
        self.q_table_path = hybrid_cfg.get("rl_q_table_path")
        self.q_table = self._load_q_table(self.q_table_path)
        self.default_slippage = float(config.get("execution", {}).get("deviation", 20))

    def plan(self, signal: dict, regime: RegimeState, anomaly: AnomalyState, volatility: VolatilityForecast) -> ExecutionPlan:
        if signal.get("side") == "NONE":
            return ExecutionPlan("NO_TRADE", 0.0, 0.0, "NONE", ["no_signal"])

        state_key = self._state_key(regime, volatility)
        if state_key in self.q_table:
            action = self.q_table[state_key]
            return ExecutionPlan(
                policy="Q_TABLE",
                aggressiveness=float(action.get("aggressiveness", 0.5)),
                max_slippage_points=float(action.get("max_slippage_points", self.default_slippage)),
                order_type=str(action.get("order_type", "MARKET")),
                reasons=[f"q_table_state={state_key}"],
            )

        aggressiveness = 0.55
        slippage = self.default_slippage
        order_type = "MARKET"
        reasons = [f"rule_state={state_key}"]

        if regime.volatility_state == "HIGH_VOL" or anomaly.score > 2.0:
            aggressiveness = 0.3
            slippage = min(slippage, 10.0)
            order_type = "LIMIT_OR_SKIP"
            reasons.append("defensive_execution_in_uncertain_market")
        elif regime.trend_state in {"TREND_UP", "TREND_DOWN"} and signal.get("confidence", 0) >= 0.8:
            aggressiveness = 0.75
            slippage = max(slippage, 20.0)
            reasons.append("aggressive_trend_follow_execution")

        return ExecutionPlan("RULE_FALLBACK", aggressiveness, slippage, order_type, reasons)

    def _state_key(self, regime: RegimeState, volatility: VolatilityForecast) -> str:
        vol_bucket = "atr_expanding" if volatility.forecast_atr > volatility.current_atr else "atr_contracting"
        return f"{regime.trend_state}|{regime.volatility_state}|{vol_bucket}"

    @staticmethod
    def _load_q_table(path_value: str | None) -> dict[str, dict]:
        if not path_value:
            return {}
        path = Path(path_value)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


class HybridAIEngine:
    """Coordinates regime, boosting, anomaly, volatility, risk, and execution layers."""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = bool(config.get("hybrid_ai", {}).get("enabled", False))
        self.regime_classifier = RegimeClassifier(config)
        self.signal_scorer = BoostingSignalScorer(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.volatility_model = VolatilityModel(config)
        self.risk_ai = RiskAI(config)
        self.execution_policy = RLExecutionPolicy(config)

    def evaluate(self, df: pd.DataFrame, base_signal: dict) -> dict:
        if not self.enabled:
            return base_signal

        signal = dict(base_signal)
        regime = self.regime_classifier.classify(df)
        anomaly = self.anomaly_detector.detect(df)
        volatility = self.volatility_model.forecast(df, regime)
        score, score_reasons = self.signal_scorer.score(df, signal, regime)
        risk = self.risk_ai.decide(score, regime, anomaly, volatility)

        signal["confidence"] = score
        signal.setdefault("reasons", [])
        signal["reasons"] = list(signal["reasons"]) + score_reasons
        signal["hybrid_ai"] = {
            "regime": asdict(regime),
            "anomaly": asdict(anomaly),
            "volatility": asdict(volatility),
            "risk": asdict(risk),
        }
        signal["risk_multiplier"] = risk.risk_multiplier
        signal["max_lot"] = risk.max_lot
        signal["stop_loss_atr_mult"] = volatility.stop_loss_atr_mult
        signal["take_profit_atr_mult"] = volatility.take_profit_atr_mult

        min_score = self.signal_scorer.threshold
        if signal.get("side") != "NONE" and score < min_score:
            signal["side"] = "NONE"
            signal["reasons"].append(f"blocked_by_hybrid_score<{min_score}")
        if signal.get("side") != "NONE" and not risk.approved:
            signal["side"] = "NONE"
            signal["reasons"].extend(risk.reasons)

        execution_plan = self.execution_policy.plan(signal, regime, anomaly, volatility)
        signal["execution_plan"] = asdict(execution_plan)
        return signal
