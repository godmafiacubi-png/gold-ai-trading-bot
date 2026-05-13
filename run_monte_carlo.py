import random
import numpy as np
import pandas as pd


START_BALANCE = 1000
SIMULATIONS = 1000


def simulate_equity(trades):
    balance = START_BALANCE
    peak = balance
    max_dd = 0

    equity = []

    risk_pct = 0.01

    for pnl_r in trades:
        risk_amount = balance * risk_pct

        # random execution degradation
        execution_noise = random.uniform(0.85, 1.05)

        pnl = pnl_r * risk_amount * execution_noise

        balance += pnl

        peak = max(peak, balance)

        dd = peak - balance
        max_dd = max(max_dd, dd)

        equity.append(balance)

    return {
        "final_balance": balance,
        "net_profit": balance - START_BALANCE,
        "max_drawdown_pct": (max_dd / START_BALANCE) * 100,
        "equity": equity,
    }


def main():
    trades = pd.read_csv("logs/backtest_trades.csv")

    if trades.empty:
        raise RuntimeError("No trades found")

    pnl_series = trades["r_multiple"].values

    results = []

    for _ in range(SIMULATIONS):
        shuffled = pnl_series.copy()

        np.random.shuffle(shuffled)

        result = simulate_equity(shuffled)

        results.append(result)

    df = pd.DataFrame(results)

    print("\n===== MONTE CARLO ROBUSTNESS =====")

    print("Simulations:", SIMULATIONS)

    print("\nNet Profit:")
    print("mean:", round(df["net_profit"].mean(), 2))
    print("median:", round(df["net_profit"].median(), 2))
    print("worst:", round(df["net_profit"].min(), 2))
    print("best:", round(df["net_profit"].max(), 2))

    print("\nMax Drawdown %:")
    print("mean:", round(df["max_drawdown_pct"].mean(), 2))
    print("median:", round(df["max_drawdown_pct"].median(), 2))
    print("worst:", round(df["max_drawdown_pct"].max(), 2))
    print("best:", round(df["max_drawdown_pct"].min(), 2))

    ruin = (df["final_balance"] < 700).sum()

    print("\nRisk of Ruin (<700 balance):")
    print(f"{ruin}/{SIMULATIONS}")
    print(
        "Probability:",
        round((ruin / SIMULATIONS) * 100, 2),
        "%"
    )

    profitable = (df["net_profit"] > 0).sum()

    print("\nProfitable Simulations:")
    print(f"{profitable}/{SIMULATIONS}")
    print(
        "Probability:",
        round((profitable / SIMULATIONS) * 100, 2),
        "%"
    )


if __name__ == "__main__":
    main()