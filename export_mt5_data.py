import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5

# Roughly up to 1-2 years of M5 bars, depending on broker history availability.
BARS = 200000

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

df.to_csv("data/history.csv", index=False)

print("\n===== M5 EXPORT COMPLETE =====")
print("Rows:", len(df))
print("Start:", df["time"].min())
print("End:", df["time"].max())
print(df.head())
print(df.tail())
