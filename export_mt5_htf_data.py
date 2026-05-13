import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_H1

# Enough H1 context for 1-2 years plus HTF rolling windows.
BARS = 20000

if not mt5.initialize():
    print("MT5 initialize failed")
    quit()

rates = mt5.copy_rates_from_pos(
    SYMBOL,
    TIMEFRAME,
    0,
    BARS,
)

mt5.shutdown()

if rates is None or len(rates) == 0:
    print("No data returned from MT5")
    quit()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.sort_values("time").reset_index(drop=True)

df.to_csv("data/history_h1.csv", index=False)

print("\n===== H1 EXPORT COMPLETE =====")
print("Rows:", len(df))
print("Start:", df["time"].min())
print("End:", df["time"].max())
print(df.head())
print(df.tail())
