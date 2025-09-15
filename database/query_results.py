#!/usr/bin/env python3
"""Query and analyze stored backtest results."""

import sys
from pathlib import Path
# tabulate is optional, use simple formatting if not available
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database import DatabaseConfig, DatabaseManager, BacktestRepository


def display_best_strategies(repository: BacktestRepository, symbol: str = None):
    """Display best performing strategies."""
    print("\n" + "="*80)
    print("BEST PERFORMING STRATEGIES")
    print("="*80)

    results = repository.get_best_strategies(symbol=symbol, min_trades=1, limit=10)

    if not results:
        print("No results found")
        return

    # Format for display
    headers = ["Strategy", "Type", "Symbol", "Avg Return", "Avg Sharpe", "Runs", "Total Trades"]
    rows = []

    for r in results:
        rows.append([
            r['name'][:30],
            r['strategy_type'],
            r['symbol'],
            f"{r['avg_return']:.2%}" if r['avg_return'] else "N/A",
            f"{r['avg_sharpe']:.2f}" if r['avg_sharpe'] else "N/A",
            r['num_runs'],
            r['total_trades']
        ])

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Simple formatting without tabulate
        print(" | ".join(headers))
        print("-" * 80)
        for row in rows:
            print(" | ".join(str(x) for x in row))


def display_recent_runs(repository: BacktestRepository, limit: int = 10):
    """Display recent backtest runs."""
    print("\n" + "="*80)
    print(f"RECENT BACKTEST RUNS (Last {limit})")
    print("="*80)

    results = repository.get_backtest_runs(limit=limit)

    if not results:
        print("No results found")
        return

    headers = ["ID", "Strategy", "Symbol", "Period", "Return", "Sharpe", "Trades", "Created"]
    rows = []

    for r in results:
        period = f"{r['start_date'].strftime('%Y-%m-%d')} to {r['end_date'].strftime('%Y-%m-%d')}"
        rows.append([
            r['id'],
            r['strategy_name'][:20],
            r['symbol'],
            period[:20],
            f"{r['total_return']:.2%}" if r['total_return'] else "N/A",
            f"{r['sharpe_ratio']:.2f}" if r['sharpe_ratio'] else "N/A",
            r['num_trades'] or 0,
            r['created_at'].strftime('%Y-%m-%d %H:%M')
        ])

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        # Simple formatting without tabulate
        print(" | ".join(headers))
        print("-" * 80)
        for row in rows:
            print(" | ".join(str(x) for x in row))


def display_strategy_stats(repository: BacktestRepository, strategy_id: int):
    """Display statistics for a specific strategy."""
    print(f"\n" + "="*80)
    print(f"STRATEGY STATISTICS (ID: {strategy_id})")
    print("="*80)

    strategy = repository.get_strategy(strategy_id)
    if not strategy:
        print("Strategy not found")
        return

    print(f"Name: {strategy.name}")
    print(f"Type: {strategy.strategy_type}")
    print(f"Parameters: {strategy.parameters}")

    stats = repository.get_strategy_statistics(strategy_id)
    if stats:
        print(f"\nPerformance Statistics:")
        print(f"  Total Runs:     {stats['total_runs']}")
        print(f"  Avg Return:     {stats['avg_return']:.2%}" if stats['avg_return'] else "  Avg Return:     N/A")
        print(f"  Avg Sharpe:     {stats['avg_sharpe']:.2f}" if stats['avg_sharpe'] else "  Avg Sharpe:     N/A")
        print(f"  Best Return:    {stats['best_return']:.2%}" if stats['best_return'] else "  Best Return:    N/A")
        print(f"  Worst Return:   {stats['worst_return']:.2%}" if stats['worst_return'] else "  Worst Return:   N/A")
        print(f"  Avg Trades:     {stats['avg_trades']}")


def display_run_details(repository: BacktestRepository, run_id: int):
    """Display detailed information about a specific run."""
    print(f"\n" + "="*80)
    print(f"BACKTEST RUN DETAILS (ID: {run_id})")
    print("="*80)

    # Get trades
    trades = repository.get_trades_for_run(run_id)
    if trades:
        print(f"\nTrades ({len(trades)} total):")
        headers = ["Date", "Action", "Symbol", "Quantity", "Price", "Value", "Commission"]
        rows = []

        for t in trades[:10]:  # Show first 10 trades
            rows.append([
                t['trade_date'].strftime('%Y-%m-%d %H:%M'),
                t['action'],
                t['symbol'],
                f"{t['quantity']:.2f}",
                f"${t['price']:.2f}",
                f"${t['trade_value']:.2f}" if t['trade_value'] else "N/A",
                f"${t['commission']:.2f}" if t['commission'] else "$0.00"
            ])

        if HAS_TABULATE:
            print(tabulate(rows, headers=headers, tablefmt="simple"))
        else:
            print(" | ".join(headers))
            print("-" * 80)
            for row in rows:
                print(" | ".join(str(x) for x in row))

        if len(trades) > 10:
            print(f"... and {len(trades) - 10} more trades")

    # Get daily performance summary
    daily_perf = repository.get_daily_performance(run_id)
    if daily_perf:
        df = pd.DataFrame(daily_perf)
        print(f"\nPerformance Summary:")
        print(f"  Days:           {len(daily_perf)}")
        print(f"  Start Value:    ${daily_perf[0]['portfolio_value']:.2f}")
        print(f"  End Value:      ${daily_perf[-1]['portfolio_value']:.2f}")
        print(f"  Max Value:      ${df['portfolio_value'].max():.2f}")
        print(f"  Min Value:      ${df['portfolio_value'].min():.2f}")


def main():
    """Main function to demonstrate database queries."""

    print("="*80)
    print("BACKTESTING DATABASE QUERY TOOL")
    print("="*80)

    # Initialize database connection
    config = DatabaseConfig.from_env()
    db_manager = DatabaseManager(config)

    # Test connection
    if not db_manager.test_connection():
        print("✗ Failed to connect to database")
        print("Please run 'python database/init_db.py' first")
        return

    repository = BacktestRepository(db_manager)

    try:
        # Display various queries
        display_best_strategies(repository)
        display_recent_runs(repository, limit=5)

        # Get some strategies to show details
        strategies = repository.find_strategies()
        if strategies:
            display_strategy_stats(repository, strategies[0].id)

        # Get recent runs to show details
        runs = repository.get_backtest_runs(limit=1)
        if runs:
            display_run_details(repository, runs[0]['id'])

        print("\n" + "="*80)
        print("Query examples completed!")
        print("="*80)

    except Exception as e:
        print(f"Error querying database: {e}")

    finally:
        db_manager.close()


if __name__ == "__main__":
    main()