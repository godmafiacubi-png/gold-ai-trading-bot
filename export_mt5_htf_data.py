import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 3000

if not mt5.initialize():
    print("MT5 initialize failed")
    quit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

mt5.shutdown()

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

df.to_csv("data/history_h1.csv", index=False)

print("Exported data/history_h1.csv")
print(df.head())
print(df.tail())