import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EMACrossoverStrategy:
    def __init__(self, short_window=12, long_window=26):
        self.short_window = short_window
        self.long_window = long_window
        self.data = None
        self.signals = None

    def prepare_data(self, df):
        """
        Prepare data with EMA indicators
        Expected columns: 'Close' price
        """
        self.data = df.copy()
        self.data['ShortEMA'] = self.data['Close'].ewm(span=self.short_window).mean()
        self.data['LongEMA'] = self.data['Close'].ewm(span=self.long_window).mean()
        
        # Generate signals
        self.data['Signal'] = 0
        self.data['Signal'][self.short_window:] = np.where(
            self.data['ShortEMA'][self.short_window:] > self.data['LongEMA'][self.short_window:], 1, 0
        )
        
        # Generate trading positions
        self.data['Position'] = self.data['Signal'].diff()
        return self.data

    def backtest(self, initial_capital=10000.0):
        """
        Backtest the strategy
        Returns performance metrics and trade log
        """
        if self.data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio['Close'] = self.data['Close']
        portfolio['Position'] = self.data['Position']
        
        # Calculate holdings and cash
        portfolio['Holdings'] = 0.0
        portfolio['Cash'] = initial_capital
        portfolio['Total'] = initial_capital
        portfolio['Returns'] = 0.0
        
        position = 0
        cash = initial_capital
        
        for i, (index, row) in enumerate(portfolio.iterrows()):
            if row['Position'] == 1:  # Buy signal
                shares_bought = cash // row['Close']
                position += shares_bought
                cash -= shares_bought * row['Close']
            elif row['Position'] == -1:  # Sell signal
                cash += position * row['Close']
                position = 0
            
            portfolio.loc[index, 'Holdings'] = position * row['Close']
            portfolio.loc[index, 'Cash'] = cash
            portfolio.loc[index, 'Total'] = portfolio.loc[index, 'Cash'] + portfolio.loc[index, 'Holdings']
            
            if i > 0:
                portfolio.loc[index, 'Returns'] = (portfolio.loc[index, 'Total'] / portfolio.iloc[i-1]['Total']) - 1
        
        # Calculate performance metrics
        total_return = (portfolio['Total'][-1] / initial_capital - 1) * 100
        total_trades = (self.data['Position'] != 0).sum()
        win_rate = self._calculate_win_rate(portfolio)
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'portfolio': portfolio
        }

    def _calculate_win_rate(self, portfolio):
        """
        Calculate win rate based on trade profitability
        """
        trades = []
        position = 0
        entry_price = 0
        
        for i, (index, row) in enumerate(portfolio.iterrows()):
            if row['Position'] == 1 and position == 0:  # Enter long
                position = 1
                entry_price = row['Close']
            elif row['Position'] == -1 and position == 1:  # Exit long
                exit_price = row['Close']
                trades.append((exit_price - entry_price) / entry_price)
                position = 0
        
        if len(trades) == 0:
            return 0.0
        
        profitable_trades = [t for t in trades if t > 0]
        return len(profitable_trades) / len(trades) * 100

    def plot_results(self, portfolio):
        """
        Plot portfolio performance
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Price and EMAs
        ax[0].plot(self.data['Close'], label='Close Price', alpha=0.7)
        ax[0].plot(self.data['ShortEMA'], label=f'Short EMA ({self.short_window})', alpha=0.7)
        ax[0].plot(self.data['LongEMA'], label=f'Long EMA ({self.long_window})', alpha=0.7)
        
        # Plot buy/sell signals
        buy_signals = self.data[self.data['Position'] == 1]
        sell_signals = self.data[self.data['Position'] == -1]
        ax[0].scatter(buy_signals.index, buy_signals['ShortEMA'], marker='^', color='g', s=100, label='Buy Signal')
        ax[0].scatter(sell_signals.index, sell_signals['ShortEMA'], marker='v', color='r', s=100, label='Sell Signal')
        
        ax[0].set_title('EMA Crossover Strategy')
        ax[0].legend()
        ax[0].grid(True)
        
        # Plot 2: Portfolio Value
        ax[1].plot(portfolio['Total'], label='Portfolio Value', color='purple')
        ax[1].set_title('Portfolio Value Over Time')
        ax[1].legend()
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Sample data (replace with real data)
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    sample_data = pd.DataFrame({
        'Close': prices
    }, index=dates)
    
    # Initialize strategy
    strategy = EMACrossoverStrategy(short_window=12, long_window=26)
    
    # Prepare data
    strategy.prepare_data(sample_data)
    
    # Run backtest
    results = strategy.backtest(initial_capital=10000.0)
    
    # Print results
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    
    # Plot results
    strategy.plot_results(results['portfolio'])