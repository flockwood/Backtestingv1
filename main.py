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
    
    # Step 1: Download data
    data = download_stock_data(symbol, start_date, end_date)
    
    # Debug: Check data structure after download
    print("\nData shape:", data.shape)
    print("Data columns:", data.columns.tolist())
    print("\nFirst 3 rows:")
    print(data.head(3))
    
    # Step 2: Generate signals
    print(f"\nGenerating SMA crossover signals (10-day vs 20-day)...")
    data_with_signals = generate_sma_crossover_signals(data, 
                                                       short_window=10, 
                                                       long_window=20)
    
    # Debug: Check signals
    print("\nSignal distribution:")
    print(data_with_signals['signal'].value_counts().sort_index())
    
    # Show when signals occur (excluding NaN)
    signal_dates = data_with_signals[data_with_signals['signal'].notna() & (data_with_signals['signal'] != 0)]
    if len(signal_dates) > 0:
        print(f"\nTrading signals:")
        print(signal_dates[['Close', 'SMA_short', 'SMA_long', 'signal']].head(10))
    
    # Step 3: Run backtest
    print(f"\nRunning backtest...")
    results = run_backtest(data_with_signals, initial_capital)
    
    # Step 4: Print results
    print("\n=== Backtest Results ===")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    
    # Show all trades
    if results['trades']:
        print(f"\nAll {len(results['trades'])} trades:")
        for i, trade in enumerate(results['trades']):
            print(f"  {i+1}. {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} "
                  f"{trade['shares']:.2f} shares at ${trade['price']:.2f}")

if __name__ == "__main__":
    main()