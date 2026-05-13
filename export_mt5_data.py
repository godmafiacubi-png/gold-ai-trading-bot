import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5

# Try large history first, then fall back because some brokers/terminals
# refuse very large M5 requests even when H1 history is available.
REQUEST_SIZES = [200000, 150000, 100000, 75000, 50000, 30000, 10000]


def export_rates() -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize failed")

    info = mt5.symbol_info(SYMBOL)
    if info is None:
        mt5.shutdown()
        raise RuntimeError(f"Symbol not found in MT5: {SYMBOL}")

    if not info.visible:
        mt5.symbol_select(SYMBOL, True)

    last_error = None
    rates = None
    used_bars = None

    for bars in REQUEST_SIZES:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, bars)
        last_error = mt5.last_error()

        if rates is not None and len(rates) > 0:
            used_bars = bars
            break

        print(f"No M5 data for bars={bars}. MT5 last_error={last_error}")

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            "No M5 data returned from MT5 after all fallback attempts. "
            "Open XAUUSDm M5 chart in MT5, scroll/load history, then retry."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    print(f"Used request size: {used_bars}")
    return df


def main():
    df = export_rates()
    df.to_csv("data/history.csv", index=False)

    print("\n===== M5 EXPORT COMPLETE =====")
    print("Rows:", len(df))
    print("Start:", df["time"].min())
    print("End:", df["time"].max())
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
