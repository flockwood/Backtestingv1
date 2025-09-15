"""
Main script with database integration for storing backtest results.
"""

from datetime import datetime, timedelta

# Import components
from data import AlpacaDataLoader
from strategies import SMAStrategy
from core import BacktestEngine
from core.costs import SimpleCostModel
from analysis import PerformanceAnalyzer
from database import DatabaseConfig, DatabaseManager, BacktestRepository


def main():
    """Run backtest with database storage."""

    print("="*60)
    print("BACKTESTING WITH DATABASE STORAGE")
    print("="*60)

    # Configuration
    symbol = 'AAPL'
    initial_capital = 10000
    commission = 0.001
    position_size = 0.95

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("-" * 60)

    # Initialize database
    print("Initializing database connection...")
    try:
        config = DatabaseConfig.from_env()
        db_manager = DatabaseManager(config)

        if not db_manager.test_connection():
            print("✗ Failed to connect to database")
            print("Please ensure PostgreSQL is running and database is initialized")
            print("Run: python database/init_db.py")
            return False

        repository = BacktestRepository(db_manager)
        print("✓ Database connected")

    except Exception as e:
        print(f"✗ Database error: {e}")
        print("Continuing without database storage...")
        db_manager = None
        repository = None

    try:
        # Initialize components
        print("Initializing components...")
        data_loader = AlpacaDataLoader()
        strategy = SMAStrategy(short_window=10, long_window=20)

        # Initialize cost model
        cost_model = SimpleCostModel(
            fixed_fee=0,
            percentage_fee=0.001,
            base_slippage_bps=5,
            market_impact_coefficient=10,
            bid_ask_spread_bps=10
        )

        # Initialize engine with database support
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            position_size=position_size,
            cost_model=cost_model,
            save_to_db=True if repository else False,
            db_manager=db_manager,
            repository=repository
        )

        print("✓ All components ready")
        print("-" * 60)

        # Run backtest
        print("Starting backtest...")
        result = engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        # Display results
        PerformanceAnalyzer.print_results(result)
        PerformanceAnalyzer.print_trade_summary(result)

        # Transaction cost analysis
        print("\n" + "="*60)
        print("TRANSACTION COST ANALYSIS")
        print("="*60)
        print(f"Total Commission:    ${result.total_commission:>12,.2f}")
        print(f"Total Slippage:      ${result.total_slippage:>12,.2f}")
        print(f"Total Spread Cost:   ${result.total_spread_cost:>12,.2f}")
        print("-" * 60)
        print(f"Total Trading Costs: ${result.total_transaction_costs:>12,.2f}")

        if repository:
            print("\n" + "="*60)
            print("DATABASE STORAGE")
            print("="*60)
            print("✓ Results saved to database")
            print("Run 'python database/query_results.py' to view stored results")

    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        return False

    finally:
        if db_manager:
            db_manager.close()

    return True


def run_parameter_sweep():
    """Run multiple strategy configurations and save to database."""

    print("\n" + "="*60)
    print("PARAMETER SWEEP WITH DATABASE STORAGE")
    print("="*60)

    # Configuration
    symbol = 'AAPL'
    initial_capital = 10000
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    # Initialize database
    config = DatabaseConfig.from_env()
    db_manager = DatabaseManager(config)
    repository = BacktestRepository(db_manager)

    # Different strategy configurations
    parameter_sets = [
        (5, 15),
        (10, 20),
        (15, 30),
        (20, 50),
        (10, 30),
        (5, 20)
    ]

    data_loader = AlpacaDataLoader()
    results = []

    for short_window, long_window in parameter_sets:
        print(f"\nTesting SMA({short_window}, {long_window})...")

        strategy = SMAStrategy(short_window=short_window, long_window=long_window)

        engine = BacktestEngine(
            initial_capital=initial_capital,
            save_to_db=True,
            db_manager=db_manager,
            repository=repository
        )

        try:
            result = engine.run_backtest(
                data_loader=data_loader,
                strategy=strategy,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            results.append((strategy, result))
            print(f"  Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Query and display best results from database
    if results:
        print("\n" + "="*60)
        print("PARAMETER SWEEP RESULTS FROM DATABASE")
        print("="*60)

        best_strategies = repository.get_best_strategies(symbol=symbol, limit=10)

        print(f"{'Parameters':<15} {'Avg Return':<12} {'Avg Sharpe':<12} {'Runs':<8}")
        print("-" * 60)

        for strategy in best_strategies[:5]:
            params = strategy['parameters']
            param_str = f"SMA({params.get('short_window')}/{params.get('long_window')})"
            print(f"{param_str:<15} {strategy['avg_return']:>10.2%}  "
                  f"{strategy['avg_sharpe']:>10.2f}  {strategy['num_runs']:>6}")

    db_manager.close()


if __name__ == "__main__":
    # Run single backtest with database storage
    success = main()

    if success:
        # Optionally run parameter sweep
        response = input("\nRun parameter sweep? (y/n): ")
        if response.lower() == 'y':
            run_parameter_sweep()

    print("\n" + "="*60)
    print("Complete! Results stored in database.")
    print("="*60)