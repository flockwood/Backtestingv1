"""
Modern Backtesting System with Clean Architecture

This is the main entry point for running backtests using the refactored
architecture with proper separation of concerns.
"""

from datetime import datetime, timedelta

# Import components from our new architecture
from data import AlpacaDataLoader, CSVDataLoader
from strategies import SMAStrategy
from core import BacktestEngine
from analysis import PerformanceAnalyzer


def main():
    """Main function to run a backtest with the new architecture."""
    
    print("="*60)
    print("MODERN BACKTESTING SYSTEM")
    print("="*60)
    
    # Configuration
    symbol = 'AAPL'
    initial_capital = 10000
    commission = 0.001  # 0.1% commission
    position_size = 0.95  # Use 95% of available cash per trade
    
    # Date range (last 90 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Commission: {commission:.3%}")
    print("-" * 60)
    
    # Initialize components
    try:
        # 1. Initialize Data Loader
        print("Initializing data loader...")
        data_loader = AlpacaDataLoader()
        print(f"✓ {data_loader}")
        
        # 2. Initialize Strategy
        print("Initializing strategy...")
        strategy = SMAStrategy(short_window=10, long_window=20)
        print(f"✓ {strategy}")
        print(f"  {strategy.get_description()}")
        
        # 3. Initialize Backtesting Engine
        print("Initializing backtest engine...")
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            position_size=position_size
        )
        print(f"✓ Backtest Engine ready")
        
        print("-" * 60)
        
        # 4. Run Backtest
        print("Starting backtest...")
        result = engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # 5. Display Results
        PerformanceAnalyzer.print_results(result)
        PerformanceAnalyzer.print_trade_summary(result)
        PerformanceAnalyzer.print_detailed_analysis(result)
        
    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        print("You can try using CSV data instead by modifying the data_loader initialization:")
        print("  data_loader = CSVDataLoader()")
        return False
    
    return True


def run_multiple_strategies():
    """Example of running multiple strategies for comparison."""
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    # Configuration
    symbol = 'AAPL'
    initial_capital = 10000
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Different strategy configurations
    strategies = [
        SMAStrategy(short_window=5, long_window=15),
        SMAStrategy(short_window=10, long_window=20),
        SMAStrategy(short_window=20, long_window=50)
    ]
    
    data_loader = AlpacaDataLoader()
    results = []
    
    for strategy in strategies:
        print(f"\nTesting {strategy}...")
        
        engine = BacktestEngine(initial_capital=initial_capital)
        
        try:
            result = engine.run_backtest(
                data_loader=data_loader,
                strategy=strategy,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            results.append((strategy, result))
            
        except Exception as e:
            print(f"❌ Error with {strategy}: {e}")
            continue
    
    # Compare results
    if results:
        print("\n" + "="*60)
        print("STRATEGY COMPARISON RESULTS")
        print("="*60)
        
        print(f"{'Strategy':<20} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Sharpe':<8}")
        print("-" * 60)
        
        for strategy, result in results:
            strategy_name = f"{strategy.parameters['short_window']}/{strategy.parameters['long_window']} SMA"
            print(f"{strategy_name:<20} {result.total_return:>8.2%} {result.num_trades:>6} "
                  f"{result.win_rate:>8.2%} {result.sharpe_ratio:>6.2f}")


if __name__ == "__main__":
    # Run single backtest
    main()
    
    print("\n" + "="*60)
    print("Backtest completed! ✨")
    print("="*60)