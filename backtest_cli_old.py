#!/usr/bin/env python3
"""
Command-line interface for the backtesting system.
Makes it easy to run backtests with different configurations.
"""

import click
import json
from datetime import datetime, timedelta
from data import AlpacaDataLoader, CSVDataLoader
from strategies import SMAStrategy
from core import BacktestEngine
from core.costs import SimpleCostModel, ZeroCostModel
from analysis import PerformanceAnalyzer, PerformanceReport


@click.command()
@click.option('--symbol', '-s', default='AAPL', help='Stock symbol to backtest')
@click.option('--strategy', '-st', default='sma', type=click.Choice(['sma']), help='Strategy type')
@click.option('--short', default=10, help='Short window for SMA')
@click.option('--long', default=20, help='Long window for SMA')
@click.option('--start-date', '-sd', help='Start date (YYYY-MM-DD), default: 90 days ago')
@click.option('--end-date', '-ed', help='End date (YYYY-MM-DD), default: today')
@click.option('--capital', '-c', default=10000, type=float, help='Initial capital')
@click.option('--data-source', '-ds', default='alpaca', type=click.Choice(['alpaca', 'csv']), help='Data source')
@click.option('--costs/--no-costs', default=True, help='Include transaction costs')
@click.option('--report', '-r', is_flag=True, help='Generate HTML report')
@click.option('--output', '-o', help='Output file for report')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--compare', is_flag=True, help='Compare multiple strategies')
@click.option('--optimize', is_flag=True, help='Optimize parameters')
def backtest(symbol, strategy, short, long, start_date, end_date, capital, 
            data_source, costs, report, output, verbose, compare, optimize):
    """
    Professional backtesting system for quantitative trading strategies.
    
    Examples:
    
        # Simple backtest
        python backtest_cli.py -s AAPL
        
        # Custom parameters
        python backtest_cli.py -s MSFT --short 5 --long 15 --capital 25000
        
        # Generate report
        python backtest_cli.py -s GOOGL --report -o googl_report.html
        
        # Compare strategies
        python backtest_cli.py -s AAPL --compare
        
        # Optimize parameters
        python backtest_cli.py -s TSLA --optimize
    """
    
    # Header
    click.echo("="*60)
    click.echo(click.style("BACKTESTING SYSTEM", bold=True, fg='cyan'))
    click.echo("="*60)
    
    # Set dates if not provided
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Display configuration
    click.echo(f"\n📊 Configuration:")
    click.echo(f"  Symbol: {click.style(symbol, fg='yellow')}")
    click.echo(f"  Strategy: {strategy.upper()} ({short}/{long})")
    click.echo(f"  Period: {start_date} to {end_date}")
    click.echo(f"  Capital: ${capital:,.2f}")
    click.echo(f"  Costs: {'Enabled' if costs else 'Disabled'}")
    click.echo(f"  Data: {data_source}")
    
    if compare:
        run_comparison(symbol, capital, start_date, end_date, data_source, costs)
    elif optimize:
        run_optimization(symbol, capital, start_date, end_date, data_source)
    else:
        run_single_backtest(symbol, strategy, short, long, start_date, end_date, 
                           capital, data_source, costs, report, output, verbose)


def run_single_backtest(symbol, strategy, short, long, start_date, end_date,
                        capital, data_source, costs, report, output, verbose):
    """Run a single backtest."""
    
    click.echo(f"\n⚙️  Running backtest...")
    
    # Initialize data loader
    if data_source == 'alpaca':
        data_loader = AlpacaDataLoader()
    else:
        data_loader = CSVDataLoader()
    
    # Initialize strategy
    if strategy == 'sma':
        strat = SMAStrategy(short_window=short, long_window=long)
    
    # Initialize cost model
    if costs:
        cost_model = SimpleCostModel(
            percentage_fee=0.001,
            base_slippage_bps=5,
            market_impact_coefficient=10,
            bid_ask_spread_bps=10
        )
    else:
        cost_model = ZeroCostModel()
    
    # Initialize engine
    engine = BacktestEngine(
        initial_capital=capital,
        cost_model=cost_model,
        position_size=0.95
    )
    
    try:
        # Run backtest
        result = engine.run_backtest(
            data_loader=data_loader,
            strategy=strat,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Display results
        click.echo(f"\n" + "="*60)
        click.echo(click.style("RESULTS", bold=True, fg='green'))
        click.echo("="*60)
        
        if verbose:
            PerformanceAnalyzer.print_comprehensive_analysis(result)
        else:
            PerformanceAnalyzer.print_results(result)
            
            # Quick summary
            click.echo(f"\n💰 Performance Summary:")
            click.echo(f"  Return: {click.style(f'{result.total_return:.2%}', fg='green' if result.total_return > 0 else 'red')}")
            click.echo(f"  Sharpe: {result.sharpe_ratio:.2f}")
            click.echo(f"  Max DD: {result.max_drawdown:.2%}")
            click.echo(f"  Trades: {result.num_trades}")
            
            if costs:
                click.echo(f"  Costs: ${result.total_transaction_costs:.2f}")
        
        # Generate report if requested
        if report:
            output_file = output or f"{symbol}_backtest_report.html"
            click.echo(f"\n📄 Generating report: {output_file}")
            
            report_gen = PerformanceReport(result)
            report_gen.create_tearsheet(output_file, show_plots=False)
            click.echo(f"✓ Report saved to {output_file}")
        
        return result
        
    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg='red'))
        return None


def run_comparison(symbol, capital, start_date, end_date, data_source, costs):
    """Compare multiple strategies."""
    
    click.echo(f"\n🔍 Comparing strategies for {symbol}...")
    
    strategies = [
        {'name': 'Conservative', 'short': 20, 'long': 50},
        {'name': 'Moderate', 'short': 10, 'long': 30},
        {'name': 'Aggressive', 'short': 5, 'long': 15},
    ]
    
    results = []
    
    for config in strategies:
        click.echo(f"\n  Testing {config['name']} ({config['short']}/{config['long']})...")
        
        result = run_single_backtest(
            symbol=symbol,
            strategy='sma',
            short=config['short'],
            long=config['long'],
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            data_source=data_source,
            costs=costs,
            report=False,
            output=None,
            verbose=False
        )
        
        if result:
            results.append({
                'name': config['name'],
                'params': f"{config['short']}/{config['long']}",
                'return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown,
                'trades': result.num_trades
            })
    
    # Display comparison table
    if results:
        click.echo(f"\n" + "="*60)
        click.echo(click.style("STRATEGY COMPARISON", bold=True, fg='cyan'))
        click.echo("="*60)
        click.echo(f"\n{'Strategy':<15} {'Params':<10} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Trades':<10}")
        click.echo("-"*70)
        
        for r in results:
            return_color = 'green' if r['return'] > 0 else 'red'
            click.echo(f"{r['name']:<15} {r['params']:<10} "
                      f"{click.style(f'{r['return']:.2%}', fg=return_color):<10} "
                      f"{r['sharpe']:.2f}      {r['max_dd']:.2%}     {r['trades']}")
        
        # Find best strategy
        best = max(results, key=lambda x: x['sharpe'])
        click.echo(f"\n🏆 Best Strategy: {best['name']} (Sharpe: {best['sharpe']:.2f})")


def run_optimization(symbol, capital, start_date, end_date, data_source):
    """Optimize strategy parameters."""
    
    click.echo(f"\n🔧 Optimizing parameters for {symbol}...")
    
    # Parameter ranges
    short_windows = [5, 10, 15, 20]
    long_windows = [20, 30, 40, 50]
    
    best_sharpe = -float('inf')
    best_params = None
    results = []
    
    total = len(short_windows) * len(long_windows)
    current = 0
    
    with click.progressbar(length=total, label='Testing combinations') as bar:
        for short in short_windows:
            for long in long_windows:
                if short >= long:
                    bar.update(1)
                    continue
                
                # Run backtest
                result = run_single_backtest(
                    symbol=symbol,
                    strategy='sma',
                    short=short,
                    long=long,
                    start_date=start_date,
                    end_date=end_date,
                    capital=capital,
                    data_source=data_source,
                    costs=True,
                    report=False,
                    output=None,
                    verbose=False
                )
                
                if result and result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_params = (short, long)
                    best_result = result
                
                if result:
                    results.append({
                        'short': short,
                        'long': long,
                        'sharpe': result.sharpe_ratio,
                        'return': result.total_return
                    })
                
                bar.update(1)
    
    # Display results
    if best_params:
        click.echo(f"\n" + "="*60)
        click.echo(click.style("OPTIMIZATION RESULTS", bold=True, fg='green'))
        click.echo("="*60)
        click.echo(f"\n🏆 Optimal Parameters:")
        click.echo(f"  Short Window: {best_params[0]}")
        click.echo(f"  Long Window: {best_params[1]}")
        click.echo(f"  Sharpe Ratio: {best_sharpe:.2f}")
        click.echo(f"  Return: {best_result.total_return:.2%}")
        
        # Show top 5
        click.echo(f"\n📊 Top 5 Combinations:")
        sorted_results = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:5]
        for i, r in enumerate(sorted_results, 1):
            click.echo(f"  {i}. SMA({r['short']}/{r['long']}): Sharpe={r['sharpe']:.2f}, Return={r['return']:.2%}")


if __name__ == '__main__':
    backtest()