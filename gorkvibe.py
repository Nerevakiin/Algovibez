# Install once: pip install yfinance vectorbt pandas matplotlib

import yfinance as yf
import vectorbt as vbt
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ Parameters ------------------
symbol = "SPY"          # Change to any ticker: AAPL, BTC-USD, etc.
start = "2015-01-01"
end = "2025-11-01"

fast_ema = 12
slow_ema = 50

initial_capital = 100_000
qty = 1  # or use portfolio sizing later

# ------------------ Download data ------------------
price = yf.download(symbol, start=start, end=end, progress=False)['Adj Close']

# ------------------ Calculate EMAs ------------------
fast_ema_series = price.ewm(span=fast_ema, adjust=False).mean()
slow_ema_series = price.ewm(span=slow_ema, adjust=False).mean()

# ------------------ Generate signals with vectorbt ------------------
entries = fast_ema_series > slow_ema_series  # Golden cross → buy
exits  = fast_ema_series < slow_ema_series  # Death cross → sell

# Create portfolio simulation
pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=initial_capital,
    fees=0.001,        # 0.1% trading fee (optional)
    freq='1d'
)

# ------------------ Results ------------------
print(pf.stats())
print(f"Total Return: {pf.total_return():.2%}")
print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
print(f"Max Drawdown: {pf.max_drawdown():.2%}")
print(f"Win Rate: {pf.win_rate():.2%}")

# ------------------ Plot ------------------
fig = pf.plot()
fast_ema_vbt = fast_ema_series.vbt.plot(trace_kwargs=dict(name=f'EMA {fast_ema}'))
slow_ema_vbt = slow_ema_series.vbt.plot(trace_kwargs=dict(name=f'EMA {slow_ema}'))

fig.add_trace(fast_ema_vbt.data[0])
fig.add_trace(slow_ema_vbt.data[0])
fig.update_layout(title=f"{symbol} - EMA {fast_ema}/{slow_ema} Crossover Strategy")
plt.show()