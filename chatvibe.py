import pandas as pd
import numpy as np

class EMACrossStrategy:
    def __init__(self, df, fast_period=12, slow_period=26):
        self.df = df.copy()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def prepare(self):
        self.df['EMA_fast'] = self.df['Close'].ewm(span=self.fast_period, adjust=False).mean()
        self.df['EMA_slow'] = self.df['Close'].ewm(span=self.slow_period, adjust=False).mean()
        self.df['Signal'] = np.where(self.df['EMA_fast'] > self.df['EMA_slow'], 1, 0)
        self.df['Position'] = self.df['Signal'].diff()
        return self.df

    def backtest(self, initial_balance=1000):
        balance = initial_balance
        position = 0
        entry_price = 0
        trades = []

        for i, row in self.df.iterrows():
            # Buy signal
            if row['Position'] == 1 and position == 0:
                position = balance / row['Close']
                entry_price = row['Close']
                balance = 0
                trades.append((i, 'BUY', entry_price))

            # Sell signal
            elif row['Position'] == -1 and position > 0:
                exit_price = row['Close']
                balance = position * exit_price
                trades.append((i, 'SELL', exit_price))
                position = 0

        # Close open position at the end
        if position > 0:
            balance = position * self.df.iloc[-1]['Close']

        return balance, trades

# Example usage:
# df = pd.read_csv('data.csv')  # Must contain a 'Close' column
# strategy = EMACrossStrategy(df)
# df_with_signals = strategy.prepare()
# final_balance, trades = strategy.backtest()
# print("Final Balance:", final_balance)
# print(trades)
