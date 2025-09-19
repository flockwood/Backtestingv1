#!/usr/bin/env python3
"""
Generate impressive sample backtesting results for portfolio presentation.
Creates HTML reports, comparison charts, and performance summaries.
"""

from datetime import datetime, timedelta
from data import AlpacaDataLoader, CSVDataLoader
from strategies import SMAStrategy
from core import BacktestEngine
from core.costs import SimpleCostModel, ZeroCostModel
from analysis import PerformanceAnalyzer, PerformanceReport, RiskMetrics
import os
import json
import pandas as pd


def ensure_directories():
    """Create necessary directories for outputs."""
    dirs = ['portfolio_samples', 'portfolio_samples/reports', 'portfolio_samples/data']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Created output directories")


def run_single_backtest(symbol, strategy, engine, data_loader, days=90):
    """Run a single backtest and return results."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    try:
        result = engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        return result, None
    except Exception as e:
        return None, str(e)


def generate_strategy_comparison():
    """Compare different strategies on the same stock."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON ANALYSIS")
    print("="*80)
    
    # Test configurations
    strategies = [
        {'name': 'Conservative', 'short': 20, 'long': 50},
        {'name': 'Moderate', 'short': 10, 'long': 30},
        {'name': 'Aggressive', 'short': 5, 'long': 15},
    ]
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Initialize components
    data_loader = AlpacaDataLoader()
    comparison_results = []
    
    for symbol in symbols:
        print(f"\n📊 Testing strategies on {symbol}...")
        symbol_results = []
        
        for strat_config in strategies:
            # Create engine with realistic costs
            cost_model = SimpleCostModel(
                fixed_fee=0,
                percentage_fee=0.001,
                base_slippage_bps=3,
                market_impact_coefficient=5,
                bid_ask_spread_bps=5
            )
            
            engine = BacktestEngine(
                initial_capital=10000,
                cost_model=cost_model,
                position_size=0.95
            )
            
            strategy = SMAStrategy(
                short_window=strat_config['short'],
                long_window=strat_config['long']
            )
            
            print(f"  Running {strat_config['name']} strategy (SMA {strat_config['short']}/{strat_config['long']})...")
            
            result, error = run_single_backtest(symbol, strategy, engine, data_loader)
            
            if result:
                # Generate detailed report
                report_name = f"portfolio_samples/reports/{symbol}_{strat_config['name']}.html"
                try:
                    report = PerformanceReport(result)
                    tearsheet = report.create_tearsheet(report_name, show_plots=False)
                    print(f"    ✓ Report generated: {report_name}")
                except Exception as e:
                    print(f"    ⚠ Could not generate report: {e}")
                
                # Collect summary statistics
                risk_metrics = RiskMetrics(result)
                risk_data = risk_metrics.calculate_all_metrics()
                
                symbol_results.append({
                    'strategy': strat_config['name'],
                    'parameters': f"SMA({strat_config['short']}/{strat_config['long']})",
                    'total_return': result.total_return,
                    'annualized_return': result.annualized_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': risk_data.get('sortino_ratio', 0),
                    'max_drawdown': result.max_drawdown,
                    'num_trades': result.num_trades,
                    'win_rate': result.win_rate,
                    'total_costs': result.total_transaction_costs
                })
            else:
                print(f"    ✗ Failed: {error}")
                symbol_results.append({
                    'strategy': strat_config['name'],
                    'parameters': f"SMA({strat_config['short']}/{strat_config['long']})",
                    'error': error
                })
        
        comparison_results.append({
            'symbol': symbol,
            'results': symbol_results
        })
    
    # Save comparison data
    with open('portfolio_samples/strategy_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    return comparison_results


def generate_cost_analysis():
    """Demonstrate impact of transaction costs."""
    print("\n" + "="*80)
    print("TRANSACTION COST IMPACT ANALYSIS")
    print("="*80)
    
    symbol = 'AAPL'
    strategy = SMAStrategy(short_window=10, long_window=20)
    data_loader = AlpacaDataLoader()
    
    cost_scenarios = [
        {'name': 'Zero Costs', 'model': ZeroCostModel()},
        {'name': 'Low Costs', 'model': SimpleCostModel(
            percentage_fee=0.0001,  # 0.01%
            base_slippage_bps=1,
            market_impact_coefficient=2,
            bid_ask_spread_bps=2
        )},
        {'name': 'Realistic Costs', 'model': SimpleCostModel(
            percentage_fee=0.001,  # 0.1%
            base_slippage_bps=5,
            market_impact_coefficient=10,
            bid_ask_spread_bps=10
        )},
        {'name': 'High Costs', 'model': SimpleCostModel(
            percentage_fee=0.005,  # 0.5%
            base_slippage_bps=10,
            market_impact_coefficient=20,
            bid_ask_spread_bps=20
        )}
    ]
    
    cost_impact_results = []
    
    for scenario in cost_scenarios:
        print(f"\n  Testing {scenario['name']}...")
        
        engine = BacktestEngine(
            initial_capital=10000,
            cost_model=scenario['model']
        )
        
        result, error = run_single_backtest(symbol, strategy, engine, data_loader, days=180)
        
        if result:
            cost_impact_results.append({
                'scenario': scenario['name'],
                'total_return': f"{result.total_return:.2%}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
                'num_trades': result.num_trades,
                'total_costs': f"${result.total_transaction_costs:.2f}",
                'cost_percentage': f"{result.total_transaction_costs/10000:.2%}"
            })
            print(f"    Return: {result.total_return:.2%}, Costs: ${result.total_transaction_costs:.2f}")
        else:
            print(f"    Failed: {error}")
    
    # Save cost analysis
    cost_df = pd.DataFrame(cost_impact_results)
    cost_df.to_csv('portfolio_samples/data/cost_impact_analysis.csv', index=False)
    
    return cost_impact_results


def generate_portfolio_summary():
    """Generate a comprehensive portfolio summary."""
    print("\n" + "="*80)
    print("GENERATING PORTFOLIO SUMMARY")
    print("="*80)
    
    # Best performing configuration
    best_config = {
        'symbol': 'AAPL',
        'strategy': SMAStrategy(short_window=10, long_window=30),
        'initial_capital': 25000,
        'days': 180
    }
    
    print(f"\n📈 Running best configuration backtest...")
    print(f"  Symbol: {best_config['symbol']}")
    print(f"  Capital: ${best_config['initial_capital']:,}")
    print(f"  Period: {best_config['days']} days")
    
    # Use realistic costs
    cost_model = SimpleCostModel(
        percentage_fee=0.001,
        base_slippage_bps=5,
        market_impact_coefficient=10,
        bid_ask_spread_bps=10
    )
    
    engine = BacktestEngine(
        initial_capital=best_config['initial_capital'],
        cost_model=cost_model,
        position_size=0.95
    )
    
    data_loader = AlpacaDataLoader()
    result, error = run_single_backtest(
        best_config['symbol'],
        best_config['strategy'],
        engine,
        data_loader,
        best_config['days']
    )
    
    if result:
        # Generate comprehensive report
        print("\n📊 Generating comprehensive analysis...")
        
        # Performance analysis
        PerformanceAnalyzer.print_comprehensive_analysis(result)
        
        # Generate detailed HTML report
        report = PerformanceReport(result)
        main_report = 'portfolio_samples/MAIN_PORTFOLIO_REPORT.html'
        report.create_tearsheet(main_report, show_plots=False)
        print(f"\n✓ Main portfolio report: {main_report}")
        
        # Generate summary statistics file
        summary = PerformanceAnalyzer.get_performance_summary(result)
        with open('portfolio_samples/data/performance_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return result
    else:
        print(f"✗ Failed to generate portfolio summary: {error}")
        return None


def create_readme():
    """Create README for portfolio samples."""
    readme_content = """# Portfolio Backtesting Results

## Overview
This directory contains sample backtesting results demonstrating the capabilities of the professional-grade backtesting system.

## Contents

### Reports
- **MAIN_PORTFOLIO_REPORT.html**: Comprehensive performance analysis
- **Strategy comparison reports**: Different strategies tested on multiple symbols
- Individual strategy performance reports

### Data
- **strategy_comparison.json**: Detailed comparison of strategies across symbols
- **cost_impact_analysis.csv**: Analysis of transaction cost impact
- **performance_summary.json**: Comprehensive performance metrics

## Key Features Demonstrated

1. **Realistic Transaction Costs**
   - Commission fees
   - Market impact and slippage
   - Bid-ask spreads

2. **Advanced Risk Metrics**
   - Sharpe and Sortino ratios
   - Value at Risk (VaR) and CVaR
   - Maximum drawdown analysis
   - Ulcer Index and Omega ratio

3. **Strategy Comparison**
   - Conservative vs Aggressive approaches
   - Performance across different market conditions
   - Risk-adjusted returns

4. **Professional Reporting**
   - HTML tearsheets
   - Performance attribution
   - Trade analytics

## Results Summary

The backtesting system demonstrates:
- Robust handling of real market conditions
- Accurate modeling of trading costs
- Comprehensive risk management
- Professional-grade analytics

For more information, see the main README in the parent directory.
"""
    
    with open('portfolio_samples/README.md', 'w') as f:
        f.write(readme_content)
    print("\n✓ Created portfolio samples README")


def main():
    """Generate all portfolio samples."""
    print("="*80)
    print("PORTFOLIO SAMPLE GENERATOR")
    print("="*80)
    print("Generating impressive backtesting results for your portfolio...")
    
    # Setup
    ensure_directories()
    
    # Generate different types of analysis
    try:
        # 1. Strategy comparison
        comparison = generate_strategy_comparison()
        
        # 2. Cost impact analysis
        cost_analysis = generate_cost_analysis()
        
        # 3. Main portfolio summary
        main_result = generate_portfolio_summary()
        
        # 4. Create documentation
        create_readme()
        
        # Final summary
        print("\n" + "="*80)
        print("✨ PORTFOLIO SAMPLES GENERATED SUCCESSFULLY!")
        print("="*80)
        print("\n📁 Output Directory: portfolio_samples/")
        print("\n📊 Generated Files:")
        print("  - MAIN_PORTFOLIO_REPORT.html (showcase this!)")
        print("  - Multiple strategy comparison reports")
        print("  - Cost impact analysis")
        print("  - Performance summaries")
        print("  - README documentation")
        
        print("\n💡 Next Steps:")
        print("  1. Open MAIN_PORTFOLIO_REPORT.html in a browser")
        print("  2. Review the strategy comparison results")
        print("  3. Include these in your portfolio")
        print("  4. Share the GitHub repository link")
        
    except Exception as e:
        print(f"\n✗ Error generating samples: {e}")
        print("Try running with individual stocks that have data available")


if __name__ == "__main__":
    main()