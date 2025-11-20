import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

class EMACrossoverStrategy:
    def __init__(self, fast_ema=12, slow_ema=26, initial_capital=10000):
        """
        Initialize the EMA Crossover Strategy
        
        Args:
            fast_ema (int): Period for fast EMA
            slow_ema (int): Period for slow EMA
            initial_capital (float): Starting capital
        """
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.initial_capital = initial_capital
        self.data = None
        
    def calculate_emas(self, data):
        """Calculate EMA indicators"""
        data['fast_ema'] = data['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        data['slow_ema'] = data['Close'].ewm(span=self.slow_ema, adjust=False).mean()
        return data
    
    def generate_signals(self, data):
        """Generate buy/sell signals based on EMA crossover"""
        data['signal'] = 0
        
        # Golden Cross: Fast EMA crosses above Slow EMA (Buy Signal)
        data.loc[data['fast_ema'] > data['slow_ema'], 'signal'] = 1
        
        # Death Cross: Fast EMA crosses below Slow EMA (Sell Signal)
        data.loc[data['fast_ema'] < data['slow_ema'], 'signal'] = -1
        
        # Calculate position (1 for long, 0 for cash)
        data['position'] = data['signal'].replace(to_replace=0, method='ffill')
        data['position'] = data['position'].shift(1)  # Avoid look-ahead bias
        
        return data
    
    def backtest(self, data):
        """Run backtest on the provided data"""
        # Calculate EMAs and signals
        data = self.calculate_emas(data)
        data = self.generate_signals(data)
        
        # Initialize portfolio columns
        data['returns'] = data['Close'].pct_change()
        data['strategy_returns'] = data['returns'] * data['position']
        data['cumulative_returns'] = (1 + data['returns']).cumprod()
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()
        
        # Calculate portfolio value
        data['portfolio_value'] = self.initial_capital * data['cumulative_strategy_returns']
        
        # Calculate benchmark (buy and hold)
        data['benchmark_value'] = self.initial_capital * data['cumulative_returns']
        
        self.data = data
        return data
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if self.data is None:
            raise ValueError("Run backtest first")
            
        strategy_returns = self.data['strategy_returns'].dropna()
        benchmark_returns = self.data['returns'].dropna()
        
        # Total Returns
        total_return = (self.data['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        benchmark_return = (self.data['benchmark_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized Returns
        days = len(self.data)
        annualized_return = (1 + total_return/100) ** (365/days) - 1
        annualized_benchmark = (1 + benchmark_return/100) ** (365/days) - 1
        
        # Volatility
        volatility = strategy_returns.std() * np.sqrt(252) * 100
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe = annualized_return / (volatility/100) if volatility != 0 else 0
        benchmark_sharpe = annualized_benchmark / (benchmark_volatility/100) if benchmark_volatility != 0 else 0
        
        # Maximum Drawdown
        rolling_max = self.data['portfolio_value'].expanding().max()
        drawdown = (self.data['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Win Rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        metrics = {
            'Total Return (%)': total_return,
            'Benchmark Return (%)': benchmark_return,
            'Annualized Return (%)': annualized_return * 100,
            'Annualized Volatility (%)': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades
        }
        
        return metrics
    
    def plot_results(self):
        """Plot backtest results"""
        if self.data is None:
            raise ValueError("Run backtest first")
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot price and EMAs
        ax1.plot(self.data.index, self.data['Close'], label='Close Price', linewidth=1)
        ax1.plot(self.data.index, self.data['fast_ema'], label=f'Fast EMA ({self.fast_ema})', alpha=0.7)
        ax1.plot(self.data.index, self.data['slow_ema'], label=f'Slow EMA ({self.slow_ema})', alpha=0.7)
        
        # Plot buy/sell signals
        buy_signals = self.data[self.data['signal'] == 1]
        sell_signals = self.data[self.data['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', 
                   label='Buy Signal', alpha=1, s=50)
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', 
                   label='Sell Signal', alpha=1, s=50)
        
        ax1.set_title('EMA Crossover Strategy - Price and Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot portfolio value vs benchmark
        ax2.plot(self.data.index, self.data['portfolio_value'], label='Strategy', linewidth=2)
        ax2.plot(self.data.index, self.data['benchmark_value'], label='Buy & Hold', linewidth=2, alpha=0.7)
        ax2.set_title('Portfolio Value Comparison')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot drawdown
        rolling_max = self.data['portfolio_value'].expanding().max()
        drawdown = (self.data['portfolio_value'] - rolling_max) / rolling_max * 100
        ax3.fill_between(self.data.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(self.data.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def download_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Example usage and backtest
if __name__ == "__main__":
    # Download sample data
    print("Downloading data...")
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    data = download_data(symbol, start_date, end_date)
    
    if data is not None and not data.empty:
        # Initialize and run strategy
        strategy = EMACrossoverStrategy(fast_ema=12, slow_ema=26, initial_capital=10000)
        
        print("Running backtest...")
        results = strategy.backtest(data)
        
        # Calculate and display metrics
        metrics = strategy.calculate_metrics()
        
        print("\n" + "="*50)
        print("EMA CROSSOVER STRATEGY BACKTEST RESULTS")
        print("="*50)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Plot results
        print("\nGenerating plots...")
        strategy.plot_results()
        
        # Show recent trades
        recent_signals = results[results['signal'] != 0].tail(10)
        print(f"\nRecent trading signals:")
        for date, row in recent_signals.iterrows():
            signal = "BUY" if row['signal'] == 1 else "SELL"
            print(f"{date.strftime('%Y-%m-%d')}: {signal} at ${row['Close']:.2f}")
    
    else:
        print("Failed to download data. Using sample data...")
        
        # Create sample data if download fails
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * (1 + returns).cumprod()
        
        sample_data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Run strategy on sample data
        strategy = EMACrossoverStrategy(fast_ema=12, slow_ema=26, initial_capital=10000)
        results = strategy.backtest(sample_data)
        metrics = strategy.calculate_metrics()
        
        print("\nBACKTEST RESULTS (Sample Data):")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        strategy.plot_results()