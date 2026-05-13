from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import pandas as pd
import joblib

from ml.meta_features import META_FEATURE_COLUMNS, normalize_meta_record


@dataclass
class TradeResult:
    side: str
    entry_time: object
    exit_time: object
    entry: float
    exit: float
    sl: float
    tp: float
    lot: float
    risk_amount: float
    pnl: float
    r_multiple: float
    result: str
    bars_held: int
    mae: float
    mfe: float
    spread: float
    slippage: float
    balance: float
    meta_probability: float


class Backtester:
    """
    Realistic-ish OHLC backtester for XAUUSD CFD-style symbols.

    Execution model:
    - BUY enters on Ask: ask = bid/open + spread + slippage
    - BUY exits on Bid
    - SELL enters on Bid with adverse slippage
    - SELL exits on Ask: ask = bid/high-low-close + spread
    - If SL and TP are both hit in the same candle, assume SL first.

    Optional ML meta-filter:
    - The base strategy generates a candidate setup.
    - The meta-model estimates whether that setup is worth taking.
    - Trades below meta_threshold are skipped.

    This is still not tick-perfect. It is intentionally conservative for OHLC data.
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_per_trade_pct: float = 1.0,
        rr: float = 2.0,
        sl_atr_mult: float = 1.5,
        max_holding_bars: int = 24,
        commission_per_lot: float = 7.0,
        slippage_points_min: float = 5,
        slippage_points_max: float = 30,
        point_value: float = 0.001,
        contract_size: float = 100.0,
        min_lot: float = 0.01,
        max_lot: float = 200.0,
        seed: int = 42,
        use_meta_filter: bool = False,
        meta_model_path: str = "models/meta_filter.pkl",
        meta_threshold: float = 0.55,
    ):
        self.initial_balance = float(initial_balance)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.rr = float(rr)
        self.sl_atr_mult = float(sl_atr_mult)
        self.max_holding_bars = int(max_holding_bars)
        self.commission_per_lot = float(commission_per_lot)
        self.slippage_points_min = float(slippage_points_min)
        self.slippage_points_max = float(slippage_points_max)
        self.point_value = float(point_value)
        self.contract_size = float(contract_size)
        self.min_lot = float(min_lot)
        self.max_lot = float(max_lot)
        self.use_meta_filter = bool(use_meta_filter)
        self.meta_threshold = float(meta_threshold)
        self.meta_model = None

        if self.use_meta_filter:
            model_path = Path(meta_model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Meta model not found: {meta_model_path}. "
                    "Run: python -m ml.train_meta_filter"
                )
            self.meta_model = joblib.load(model_path)

        random.seed(seed)

    def run(self, df: pd.DataFrame) -> dict:
        Path("logs").mkdir(parents=True, exist_ok=True)

        balance = self.initial_balance
        peak = balance
        max_drawdown = 0.0
        trades: list[TradeResult] = []
        equity_curve = []
        skipped_by_meta = 0

        if df.empty:
            return self._summary(pd.DataFrame(), balance, max_drawdown, skipped_by_meta)

        required_columns = {"open", "high", "low", "close"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        i = 0
        last_index = len(df) - self.max_holding_bars - 2

        while i < last_index:
            row = df.iloc[i]
            signal = self._generate_signal(row)

            if signal == "NONE":
                equity_curve.append({"time": row.get("time", i), "balance": round(balance, 2)})
                i += 1
                continue

            meta_probability = 1.0
            if self.use_meta_filter:
                meta_probability = self._meta_probability(row, signal)

                if meta_probability < self.meta_threshold:
                    skipped_by_meta += 1
                    equity_curve.append({"time": row.get("time", i), "balance": round(balance, 2)})
                    i += 1
                    continue

            next_row = df.iloc[i + 1]
            atr = float(row.get("atr", 0) or 0)

            if atr <= 0:
                i += 1
                continue

            spread = self._get_spread_price(next_row)
            slippage = self._random_slippage_price()
            raw_open_bid = float(next_row["open"])

            if signal == "BUY":
                # Buy opens at Ask, exits at Bid.
                entry = raw_open_bid + spread + slippage
                sl = raw_open_bid - (atr * self.sl_atr_mult)
                risk_distance = abs(entry - sl)
                tp = entry + (risk_distance * self.rr)
            else:
                # Sell opens at Bid, adverse slippage makes entry slightly worse/lower.
                entry = raw_open_bid - slippage
                # Sell stop is triggered on Ask, so include spread in SL placement.
                sl = raw_open_bid + spread + (atr * self.sl_atr_mult)
                risk_distance = abs(sl - entry)
                tp = entry - (risk_distance * self.rr)

            if risk_distance <= 0:
                i += 1
                continue

            risk_amount = balance * (self.risk_per_trade_pct / 100.0)
            lot = self._calculate_lot(risk_amount, risk_distance)

            if lot < self.min_lot:
                i += 1
                continue

            future = df.iloc[i + 1 : i + 1 + self.max_holding_bars]
            trade = self._simulate_trade(
                side=signal,
                future=future,
                entry=entry,
                sl=sl,
                tp=tp,
                lot=lot,
                risk_amount=risk_amount,
                entry_spread=spread,
                entry_slippage=slippage,
                balance_before=balance,
                meta_probability=meta_probability,
            )

            balance += trade.pnl
            peak = max(peak, balance)
            max_drawdown = max(max_drawdown, peak - balance)

            trade.balance = round(balance, 2)
            trades.append(trade)
            equity_curve.append({"time": trade.exit_time, "balance": round(balance, 2)})

            # Move forward to avoid overlapping trades.
            i += max(trade.bars_held, 1)

        trades_df = pd.DataFrame([t.__dict__ for t in trades])
        equity_df = pd.DataFrame(equity_curve)

        trades_df.to_csv("logs/backtest_trades.csv", index=False)
        equity_df.to_csv("logs/equity_curve.csv", index=False)

        return self._summary(trades_df, balance, max_drawdown, skipped_by_meta)

    def _generate_signal(self, row) -> str:
        if not bool(row.get("session_allowed", True)):
            return "NONE"

        if not bool(row.get("regime_allowed", True)):
            return "NONE"

        bullish = (
            bool(row.get("bullish_fvg_retest", False))
            and bool(row.get("bullish_bos", False))
            and bool(row.get("long_regime_ok", False))
            and bool(row.get("long_quality_ok", False))
        )

        bearish = (
            bool(row.get("bearish_fvg_retest", False))
            and bool(row.get("bearish_bos", False))
            and bool(row.get("short_regime_ok", False))
            and bool(row.get("short_quality_ok", False))
        )

        if bullish:
            return "BUY"

        if bearish:
            return "SELL"

        return "NONE"

    def _meta_probability(self, row, signal: str) -> float:
        if self.meta_model is None:
            return 1.0

        record = {col: row.get(col, 0) for col in META_FEATURE_COLUMNS}
        record["side"] = 1 if signal == "BUY" else -1
        record = normalize_meta_record(record)
        x = pd.DataFrame([record])[META_FEATURE_COLUMNS]
        return float(self.meta_model.predict_proba(x)[0][1])

    def _simulate_trade(
        self,
        side: str,
        future: pd.DataFrame,
        entry: float,
        sl: float,
        tp: float,
        lot: float,
        risk_amount: float,
        entry_spread: float,
        entry_slippage: float,
        balance_before: float,
        meta_probability: float = 1.0,
    ) -> TradeResult:
        mae = 0.0
        mfe = 0.0

        last_row = future.iloc[-1]
        exit_time = last_row.get("time", future.index[-1])
        bars_held = len(future)
        result = "TIMEOUT"

        # Conservative timeout exit.
        if side == "BUY":
            # Buy exits at Bid close.
            exit_price = float(last_row["close"])
        else:
            # Sell exits at Ask close.
            timeout_spread = self._get_spread_price(last_row)
            exit_price = float(last_row["close"]) + timeout_spread

        for n, (_, row) in enumerate(future.iterrows(), start=1):
            bid_high = float(row["high"])
            bid_low = float(row["low"])
            spread = self._get_spread_price(row)

            ask_high = bid_high + spread
            ask_low = bid_low + spread

            if side == "BUY":
                # BUY position closes on Bid.
                adverse = max(0.0, entry - bid_low)
                favorable = max(0.0, bid_high - entry)
                mae = max(mae, adverse)
                mfe = max(mfe, favorable)

                # Conservative OHLC assumption: SL first if both touched.
                if bid_low <= sl:
                    exit_price = sl
                    exit_time = row.get("time", n)
                    result = "LOSS"
                    bars_held = n
                    break

                if bid_high >= tp:
                    exit_price = tp
                    exit_time = row.get("time", n)
                    result = "WIN"
                    bars_held = n
                    break

            else:
                # SELL position closes on Ask.
                adverse = max(0.0, ask_high - entry)
                favorable = max(0.0, entry - ask_low)
                mae = max(mae, adverse)
                mfe = max(mfe, favorable)

                # Conservative OHLC assumption: SL first if both touched.
                if ask_high >= sl:
                    exit_price = sl
                    exit_time = row.get("time", n)
                    result = "LOSS"
                    bars_held = n
                    break

                if ask_low <= tp:
                    exit_price = tp
                    exit_time = row.get("time", n)
                    result = "WIN"
                    bars_held = n
                    break

        if side == "BUY":
            gross_pnl = (exit_price - entry) * self.contract_size * lot
        else:
            gross_pnl = (entry - exit_price) * self.contract_size * lot

        commission = self.commission_per_lot * lot
        pnl = gross_pnl - commission

        true_risk = abs(entry - sl) * self.contract_size * lot
        r_multiple = pnl / true_risk if true_risk > 0 else 0.0

        return TradeResult(
            side=side,
            entry_time=future.iloc[0].get("time", future.index[0]),
            exit_time=exit_time,
            entry=round(entry, 3),
            exit=round(exit_price, 3),
            sl=round(sl, 3),
            tp=round(tp, 3),
            lot=round(lot, 2),
            risk_amount=round(risk_amount, 2),
            pnl=round(pnl, 2),
            r_multiple=round(r_multiple, 3),
            result=result,
            bars_held=int(bars_held),
            mae=round(mae, 3),
            mfe=round(mfe, 3),
            spread=round(entry_spread, 3),
            slippage=round(entry_slippage, 3),
            balance=round(balance_before, 2),
            meta_probability=round(meta_probability, 4),
        )

    def _calculate_lot(self, risk_amount: float, risk_distance: float) -> float:
        raw_lot = risk_amount / (risk_distance * self.contract_size)
        lot = max(self.min_lot, min(raw_lot, self.max_lot))

        # Broker volume_step is 0.01 for your XAUUSDm symbol.
        lot = round(lot / 0.01) * 0.01
        return round(lot, 2)

    def _get_spread_price(self, row) -> float:
        spread_points = float(row.get("spread", 0) or 0)
        return spread_points * self.point_value

    def _random_slippage_price(self) -> float:
        points = random.uniform(self.slippage_points_min, self.slippage_points_max)
        return points * self.point_value

    def _summary(
        self,
        trades_df: pd.DataFrame,
        final_balance: float,
        max_drawdown: float,
        skipped_by_meta: int = 0,
    ) -> dict:
        if trades_df.empty:
            return {
                "trades": 0,
                "final_balance": round(final_balance, 2),
                "net_profit": round(final_balance - self.initial_balance, 2),
                "max_drawdown": round(max_drawdown, 2),
                "max_drawdown_pct": round(max_drawdown / max(self.initial_balance, 1) * 100, 2),
                "skipped_by_meta": int(skipped_by_meta),
            }

        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]

        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        expectancy = trades_df["pnl"].mean()
        avg_win = wins["pnl"].mean() if not wins.empty else 0.0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0.0

        max_consecutive_losses = 0
        current_losses = 0
        for pnl in trades_df["pnl"]:
            if pnl < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0

        return {
            "trades": int(len(trades_df)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "winrate": round(len(wins) / len(trades_df) * 100, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(final_balance - self.initial_balance, 2),
            "final_balance": round(final_balance, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown / max(self.initial_balance, 1) * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy_per_trade": round(expectancy, 2),
            "avg_r": round(trades_df["r_multiple"].mean(), 3),
            "median_r": round(trades_df["r_multiple"].median(), 3),
            "timeout_trades": int((trades_df["result"] == "TIMEOUT").sum()),
            "largest_win": round(trades_df["pnl"].max(), 2),
            "largest_loss": round(trades_df["pnl"].min(), 2),
            "max_consecutive_losses": int(max_consecutive_losses),
            "skipped_by_meta": int(skipped_by_meta),
            "avg_meta_probability": round(trades_df["meta_probability"].mean(), 4)
            if "meta_probability" in trades_df.columns
            else 1.0,
        }
