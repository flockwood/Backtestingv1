from src.data_loader import download_stock_data
from src.strategy import generate_sma_crossover_signals
from src.backtester import run_backtest

def main():
    # Configuration
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    initial_capital = 10000
    
    print("=== Simple Backtesting System ===\n")
    
    # Try Alpaca first, fall back to CSV if needed
    try:
        # Download fresh data from Alpaca
        data = download_stock_data(symbol, start_date, end_date, source='alpaca')
    except Exception as e:
        print(f"Alpaca download failed: {e}")
        print("Falling back to CSV data...")
        data = download_stock_data(symbol, start_date, end_date, source='csv')
    
    # Rest of the backtesting logic remains the same
    print(f"\nGenerating SMA crossover signals...")
    data_with_signals = generate_sma_crossover_signals(data, 
                                                       short_window=10, 
                                                       long_window=20)
    
    print(f"\nRunning backtest...")
    results = run_backtest(data_with_signals, initial_capital)
    
    # Print results
    print("\n=== Backtest Results ===")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")

if __name__ == "__main__":
    main()