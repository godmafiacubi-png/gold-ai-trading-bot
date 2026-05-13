from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

from core.feature_engine import FeatureEngine
from core.htf_context import HTFContextEngine
from backtest.backtester import Backtester


THRESHOLD = 0.60
RR = 1.5
OUT_DIR = Path("logs/diagnostics")


def load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_features(config: dict) -> pd.DataFrame:
    raw = pd.read_csv("data/history.csv")
    features = FeatureEngine(config).build(raw)

    if config.get("htf", {}).get("enabled", False):
        htf_raw = pd.read_csv("data/history_h1.csv")
        htf_engine = HTFContextEngine(config)
        htf_context = htf_engine.build(htf_raw)
        features = htf_engine.merge_to_ltf(features, htf_context)

    return features


def run_backtest(config: dict, features: pd.DataFrame, use_ml: bool) -> tuple[dict, pd.DataFrame]:
    bt = Backtester(
        initial_balance=float(config["risk"].get("account_balance", 1000)),
        risk_per_trade_pct=float(config["risk"].get("risk_per_trade_pct", 1.0)),
        rr=RR,
        sl_atr_mult=1.5,
        max_holding_bars=24,
        commission_per_lot=7.0,
        slippage_points_min=5,
        slippage_points_max=30,
        point_value=0.001,
        contract_size=100.0,
        use_meta_filter=use_ml,
        meta_model_path="models/meta_filter.pkl",
        meta_threshold=THRESHOLD,
    )

    summary = bt.run(features)
    trades = pd.read_csv("logs/backtest_trades.csv")

    if trades.empty:
        return summary, trades

    trades["entry_time"] = trades["entry_time"].astype(str)
    trades["trade_key"] = trades["entry_time"] + "|" + trades["side"].astype(str)

    return summary, trades


def attach_context(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    out = trades.copy()
    features = features.copy()
    features["time"] = features["time"].astype(str)

    context_cols = [
        "time",
        "session",
        "htf_trend",
        "htf_premium",
        "htf_discount",
        "recent_htf_bullish_sweep",
        "recent_htf_bearish_sweep",
        "recent_htf_bullish_bos",
        "recent_htf_bearish_bos",
        "atr_pct",
        "ema_slope",
        "sweep_count_5",
        "long_regime_ok",
        "short_regime_ok",
        "long_quality_ok",
        "short_quality_ok",
    ]

    available_cols = [c for c in context_cols if c in features.columns]

    out = out.merge(
        features[available_cols],
        left_on="entry_time",
        right_on="time",
        how="left",
    )

    return out


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for value, group in df.groupby(group_col, dropna=False):
        wins = group[group["pnl"] > 0]
        losses = group[group["pnl"] < 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        rows.append({
            group_col: value,
            "trades": int(len(group)),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "winrate": round(len(wins) / len(group) * 100, 2) if len(group) else 0,
            "profit_factor": round(profit_factor, 2),
            "net_profit": round(group["pnl"].sum(), 2),
            "expectancy": round(group["pnl"].mean(), 2),
            "avg_r": round(group["r_multiple"].mean(), 3),
            "median_r": round(group["r_multiple"].median(), 3),
            "largest_win": round(group["pnl"].max(), 2),
            "largest_loss": round(group["pnl"].min(), 2),
        })

    return pd.DataFrame(rows).sort_values("net_profit", ascending=False)


def add_probability_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "meta_probability" not in out.columns:
        out["probability_bucket"] = "no_probability"
        return out

    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    labels = [
        "<0.50",
        "0.50-0.55",
        "0.55-0.60",
        "0.60-0.65",
        "0.65-0.70",
        "0.70-0.80",
        "0.80+",
    ]

    out["probability_bucket"] = pd.cut(
        out["meta_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return out


def build_skipped_report(base_trades: pd.DataFrame, ml_trades: pd.DataFrame) -> pd.DataFrame:
    if base_trades.empty:
        return pd.DataFrame()

    accepted_keys = set(ml_trades["trade_key"]) if not ml_trades.empty else set()

    skipped = base_trades[~base_trades["trade_key"].isin(accepted_keys)].copy()
    accepted = base_trades[base_trades["trade_key"].isin(accepted_keys)].copy()

    rows = []

    for label, group in [("accepted_by_ml", accepted), ("skipped_by_ml", skipped)]:
        if group.empty:
            rows.append({
                "bucket": label,
                "trades": 0,
                "winrate": 0,
                "profit_factor": 0,
                "net_profit": 0,
                "expectancy": 0,
                "avg_r": 0,
            })
            continue

        wins = group[group["pnl"] > 0]
        losses = group[group["pnl"] < 0]
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        rows.append({
            "bucket": label,
            "trades": int(len(group)),
            "winrate": round(len(wins) / len(group) * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "net_profit": round(group["pnl"].sum(), 2),
            "expectancy": round(group["pnl"].mean(), 2),
            "avg_r": round(group["r_multiple"].mean(), 3),
        })

    return pd.DataFrame(rows)


def print_and_save(title: str, df: pd.DataFrame, path: Path) -> None:
    print(f"\n===== {title} =====")

    if df.empty:
        print("No data")
        return

    print(df.to_string(index=False))
    df.to_csv(path, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    features = build_features(config)

    base_summary, base_trades = run_backtest(config, features, use_ml=False)
    base_trades = attach_context(base_trades, features)
    base_trades.to_csv(OUT_DIR / "base_trades_with_context.csv", index=False)

    ml_summary, ml_trades = run_backtest(config, features, use_ml=True)
    ml_trades = attach_context(ml_trades, features)
    ml_trades = add_probability_bucket(ml_trades)
    ml_trades.to_csv(OUT_DIR / "ml_trades_with_context.csv", index=False)

    summary_df = pd.DataFrame([
        {"mode": "base_no_ml", **base_summary},
        {"mode": "ml_threshold_0_60", **ml_summary},
    ])

    print_and_save(
        "SUMMARY",
        summary_df,
        OUT_DIR / "summary.csv",
    )

    for name, trades in [("BASE", base_trades), ("ML", ml_trades)]:
        print_and_save(
            f"{name} BY SIDE",
            summarize_group(trades, "side"),
            OUT_DIR / f"{name.lower()}_by_side.csv",
        )

        print_and_save(
            f"{name} BY SESSION",
            summarize_group(trades, "session"),
            OUT_DIR / f"{name.lower()}_by_session.csv",
        )

        print_and_save(
            f"{name} BY HTF TREND",
            summarize_group(trades, "htf_trend"),
            OUT_DIR / f"{name.lower()}_by_htf_trend.csv",
        )

    print_and_save(
        "ML BY PROBABILITY BUCKET",
        summarize_group(ml_trades, "probability_bucket"),
        OUT_DIR / "ml_by_probability_bucket.csv",
    )

    print_and_save(
        "ML ACCEPTED VS SKIPPED",
        build_skipped_report(base_trades, ml_trades),
        OUT_DIR / "ml_accepted_vs_skipped.csv",
    )

    print("\nExported diagnostics to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
