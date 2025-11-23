import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class EMACrossoverBot:
    def __init__(self, symbol, short_ema=9, long_ema=21, initial_capital=10000):
        """
        Initialize the EMA Crossover Trading Bot
        
        Parameters:
        - symbol: Stock/Crypto ticker symbol
        - short_ema: Period for short EMA (default: 9)
        - long_ema: Period for long EMA (default: 21)
        - initial_capital: Starting capital for backtesting (default: 10000)
        """
        self.symbol = symbol
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.initial_capital = initial_capital
        self.data = None
        self.position = 0  # 0 = no position, 1 = long position
        
    def fetch_data(self, start_date, end_date=None):
        """Fetch historical price data"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching data for {self.symbol} from {start_date} to {end_date}...")
        self.data = yf.download(self.symbol, start=start_date, end=end_date)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        print(f"Data fetched: {len(self.data)} rows")
        return self.data
    
    def calculate_ema(self):
        """Calculate short and long EMAs"""
        self.data['EMA_short'] = self.data['Close'].ewm(span=self.short_ema, adjust=False).mean()
        self.data['EMA_long'] = self.data['Close'].ewm(span=self.long_ema, adjust=False).mean()
        
    def generate_signals(self):
        """Generate buy/sell signals based on EMA crossover"""
        self.data['Signal'] = 0
        
        # Buy signal: short EMA crosses above long EMA
        self.data.loc[self.data['EMA_short'] > self.data['EMA_long'], 'Signal'] = 1
        
        # Sell signal: short EMA crosses below long EMA
        self.data.loc[self.data['EMA_short'] < self.data['EMA_long'], 'Signal'] = -1
        
        # Detect actual crossover points (change in signal)
        self.data['Position'] = self.data['Signal'].diff()
        
    def backtest(self):
        """Backtest the strategy"""
        capital = self.initial_capital
        shares = 0
        trades = []
        
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            
            # Buy signal (crossover up)
            if row['Position'] == 2:  # Signal changed from -1 to 1
                if capital > 0:
                    shares = capital / row['Close']
                    capital = 0
                    trades.append({
                        'Date': self.data.index[i],
                        'Type': 'BUY',
                        'Price': row['Close'],
                        'Shares': shares
                    })
                    
            # Sell signal (crossover down)
            elif row['Position'] == -2:  # Signal changed from 1 to -1
                if shares > 0:
                    capital = shares * row['Close']
                    trades.append({
                        'Date': self.data.index[i],
                        'Type': 'SELL',
                        'Price': row['Close'],
                        'Shares': shares,
                        'Capital': capital
                    })
                    shares = 0
        
        # Close any open position at the end
        if shares > 0:
            capital = shares * self.data.iloc[-1]['Close']
            trades.append({
                'Date': self.data.index[-1],
                'Type': 'SELL (Close)',
                'Price': self.data.iloc[-1]['Close'],
                'Shares': shares,
                'Capital': capital
            })
        
        final_capital = capital if capital > 0 else shares * self.data.iloc[-1]['Close']
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate buy and hold return
        buy_hold_return = ((self.data.iloc[-1]['Close'] - self.data.iloc[0]['Close']) / 
                          self.data.iloc[0]['Close']) * 100
        
        return {
            'trades': trades,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': len(trades)
        }
    
    def plot_strategy(self):
        """Plot the price, EMAs, and buy/sell signals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot price and EMAs
        ax1.plot(self.data.index, self.data['Close'], label='Price', alpha=0.7)
        ax1.plot(self.data.index, self.data['EMA_short'], 
                label=f'EMA {self.short_ema}', alpha=0.8)
        ax1.plot(self.data.index, self.data['EMA_long'], 
                label=f'EMA {self.long_ema}', alpha=0.8)
        
        # Mark buy signals
        buy_signals = self.data[self.data['Position'] == 2]
        ax1.scatter(buy_signals.index, buy_signals['Close'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        # Mark sell signals
        sell_signals = self.data[self.data['Position'] == -2]
        ax1.scatter(sell_signals.index, sell_signals['Close'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{self.symbol} - EMA Crossover Strategy')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot signal line
        ax2.plot(self.data.index, self.data['Signal'], label='Signal', color='blue')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.fill_between(self.data.index, self.data['Signal'], 0, 
                         where=(self.data['Signal'] > 0), alpha=0.3, color='green', label='Long')
        ax2.fill_between(self.data.index, self.data['Signal'], 0, 
                         where=(self.data['Signal'] < 0), alpha=0.3, color='red', label='Short/Cash')
        ax2.set_ylabel('Position')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run(self, start_date, end_date=None):
        """Run the complete strategy"""
        print(f"\n{'='*60}")
        print(f"EMA CROSSOVER STRATEGY BACKTEST")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Short EMA: {self.short_ema}, Long EMA: {self.long_ema}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*60}\n")
        
        # Fetch data
        self.fetch_data(start_date, end_date)
        
        # Calculate indicators
        self.calculate_ema()
        
        # Generate signals
        self.generate_signals()
        
        # Backtest
        results = self.backtest()
        
        # Print results
        print("\nBACKTEST RESULTS:")
        print(f"{'-'*60}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"{'-'*60}\n")
        
        print("TRADE HISTORY:")
        print(f"{'-'*60}")
        for trade in results['trades']:
            print(f"{trade['Date'].strftime('%Y-%m-%d')} | {trade['Type']:12s} | "
                  f"Price: ${trade['Price']:8.2f} | Shares: {trade['Shares']:8.2f}")
        print(f"{'-'*60}\n")
        
        # Plot
        self.plot_strategy()
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the bot
    bot = EMACrossoverBot(
        symbol='AAPL',      # Change to any stock/crypto symbol
        short_ema=9,        # Fast EMA period
        long_ema=21,        # Slow EMA period
        initial_capital=10000
    )
    
    # Run backtest from 1 year ago to today
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    results = bot.run(start_date)
    
    # You can also test different symbols and parameters:
    # bot_btc = EMACrossoverBot('BTC-USD', short_ema=12, long_ema=26, initial_capital=10000)
    # results_btc = bot_btc.run('2023-01-01', '2024-01-01')