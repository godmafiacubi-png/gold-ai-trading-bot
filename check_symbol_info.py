import MetaTrader5 as mt5

mt5.initialize()

symbol = "XAUUSDm"

info = mt5.symbol_info(symbol)

print(info)

mt5.shutdown()