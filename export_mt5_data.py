import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 50000

if not mt5.initialize():
    print("MT5 initialize failed")
    quit()

rates = mt5.copy_rates_from_pos(
    SYMBOL,
    TIMEFRAME,
    0,
    BARS
)

mt5.shutdown()

df = pd.DataFrame(rates)

df["time"] = pd.to_datetime(
    df["time"],
    unit="s"
)

df.to_csv(
    "data/history.csv",
    index=False
)

print("Export completed")
print(df.head())