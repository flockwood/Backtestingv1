from alpaca_data_loader import AlpacaDataLoader
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_sma_crossover_signals(data, short_window=10, long_window=20):
    """Generate SMA crossover trading signals"""
    data = data.copy()
    
    # Calculate moving averages
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    data['Signal'] = 0
    data.loc[data.index[long_window:], 'Signal'] = np.where(
        data['SMA_short'][long_window:] > data['SMA_long'][long_window:], 1, 0
    )
    data['Position'] = data['Signal'].diff()
    
    return data

def run_backtest(data, initial_capital=10000):
    """Run a simple backtest on the data with signals"""
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(data)):
        if data['Position'].iloc[i] == 1:  # Buy signal
            if capital > 0:
                position = capital / data['Close'].iloc[i]
                trades.append(('BUY', data.index[i], data['Close'].iloc[i], position))
                capital = 0
        elif data['Position'].iloc[i] == -1:  # Sell signal
            if position > 0:
                capital = position * data['Close'].iloc[i]
                trades.append(('SELL', data.index[i], data['Close'].iloc[i], position))
                position = 0
    
    # Calculate final value
    if position > 0:
        final_value = position * data['Close'].iloc[-1]
    else:
        final_value = capital
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': ((final_value - initial_capital) / initial_capital) * 100,
        'num_trades': len(trades),
        'trades': trades
    }

def main():
    # Configuration
    symbol = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    initial_capital = 10000
    
    print("=== Alpaca-Powered Backtesting System ===\n")
    
    # Initialize Alpaca data loader
    loader = AlpacaDataLoader()
    
    # Download data from Alpaca Markets
    print(f"Downloading {symbol} data from Alpaca Markets...")
    data = loader.download_stock_data(symbol, start_date, end_date)
    
    if data.empty:
        print("Failed to download data. Please check your API credentials.")
        return
    
    # Generate trading signals
    print(f"\nGenerating SMA crossover signals...")
    data_with_signals = generate_sma_crossover_signals(data, 
                                                       short_window=10, 
                                                       long_window=20)
    
    # Run backtest
    print(f"\nRunning backtest...")
    results = run_backtest(data_with_signals, initial_capital)
    
    # Print results
    print("\n=== Backtest Results ===")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Data points: {len(data)}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Show recent trades
    if results['trades']:
        print("\nRecent Trades (last 5):")
        for trade in results['trades'][-5:]:
            action, date, price, shares = trade
            print(f"  {action}: {date.date()} @ ${price:.2f} ({shares:.2f} shares)")

if __name__ == "__main__":
    main()