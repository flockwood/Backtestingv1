#!/usr/bin/env python3
"""
Test all trading strategies and compare their performance.
"""

from datetime import datetime, timedelta
from data import AlpacaDataLoader
from strategies import (
    SMAStrategy, 
    RSIStrategy, 
    BollingerBandsStrategy, 
    MACDStrategy
)
from core import BacktestEngine
from core.costs import SimpleCostModel
from analysis import PerformanceAnalyzer
import pandas as pd


def test_strategy(strategy, symbol='AAPL', days=180, verbose=False):
    """Test a single strategy and return results."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {strategy.get_description() if hasattr(strategy, 'get_description') else str(strategy)}")
    print('='*60)
    
    # Initialize components
    cost_model = SimpleCostModel(
        percentage_fee=0.001,
        base_slippage_bps=5,
        market_impact_coefficient=10,
        bid_ask_spread_bps=10
    )
    
    engine = BacktestEngine(
        initial_capital=10000,
        cost_model=cost_model,
        position_size=0.95
    )
    
    data_loader = AlpacaDataLoader()
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    try:
        # Run backtest
        result = engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if verbose:
            PerformanceAnalyzer.print_results(result)
        else:
            # Quick summary
            print(f"✓ Return: {result.total_return:.2%}")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"  Max DD: {result.max_drawdown:.2%}")
            print(f"  Trades: {result.num_trades}")
            print(f"  Win Rate: {result.win_rate:.2%}")
        
        return {
            'strategy': str(strategy),
            'return': result.total_return,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown,
            'trades': result.num_trades,
            'win_rate': result.win_rate,
            'costs': result.total_transaction_costs
        }
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def compare_all_strategies(symbol='AAPL', days=180):
    """Compare all available strategies."""
    
    print("="*80)
    print(f"STRATEGY COMPARISON FOR {symbol}")
    print(f"Period: Last {days} days")
    print("="*80)
    
    # Define strategies to test
    strategies = [
        # SMA Strategies
        SMAStrategy(short_window=10, long_window=20),
        SMAStrategy(short_window=20, long_window=50),
        
        # RSI Strategies
        RSIStrategy(period=14, oversold=30, overbought=70),
        RSIStrategy(period=14, oversold=20, overbought=80),  # More extreme levels
        
        # Bollinger Bands Strategies
        BollingerBandsStrategy(period=20, num_std=2, strategy_type='mean_reversion'),
        BollingerBandsStrategy(period=20, num_std=2, strategy_type='breakout'),
        
        # MACD Strategies
        MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
        MACDStrategy(fast_period=12, slow_period=26, signal_period=9, use_histogram=True),
    ]
    
    results = []
    
    for strategy in strategies:
        result = test_strategy(strategy, symbol, days)
        if result:
            results.append(result)
    
    # Display comparison table
    if results:
        print("\n" + "="*80)
        print("STRATEGY COMPARISON RESULTS")
        print("="*80)
        
        # Convert to DataFrame for nice display
        df = pd.DataFrame(results)
        df = df.sort_values('sharpe', ascending=False)
        
        print("\nTop Performers by Sharpe Ratio:")
        print("-"*80)
        print(f"{'Strategy':<40} {'Return':>8} {'Sharpe':>8} {'Max DD':>8} {'Trades':>8} {'Win%':>8}")
        print("-"*80)
        
        for _, row in df.head(10).iterrows():
            strategy_name = row['strategy'][:40]
            return_color = '🟢' if row['return'] > 0 else '🔴'
            print(f"{strategy_name:<40} {return_color}{row['return']:>7.1%} {row['sharpe']:>8.2f} "
                  f"{row['max_dd']:>8.1%} {row['trades']:>8.0f} {row['win_rate']:>7.1%}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        best_return = df.loc[df['return'].idxmax()]
        best_sharpe = df.loc[df['sharpe'].idxmax()]
        most_trades = df.loc[df['trades'].idxmax()]
        
        print(f"\n📈 Best Return: {best_return['strategy'][:40]}")
        print(f"   Return: {best_return['return']:.2%}, Sharpe: {best_sharpe['sharpe']:.2f}")
        
        print(f"\n🎯 Best Risk-Adjusted (Sharpe): {best_sharpe['strategy'][:40]}")
        print(f"   Sharpe: {best_sharpe['sharpe']:.2f}, Return: {best_sharpe['return']:.2%}")
        
        print(f"\n📊 Most Active: {most_trades['strategy'][:40]}")
        print(f"   Trades: {most_trades['trades']:.0f}, Return: {most_trades['return']:.2%}")
        
        # Strategy type performance
        print("\n" + "="*80)
        print("PERFORMANCE BY STRATEGY TYPE")
        print("="*80)
        
        for strategy_type in ['SMA', 'RSI', 'Bollinger', 'MACD']:
            type_results = [r for r in results if strategy_type in r['strategy']]
            if type_results:
                avg_return = sum(r['return'] for r in type_results) / len(type_results)
                avg_sharpe = sum(r['sharpe'] for r in type_results) / len(type_results)
                print(f"{strategy_type:15} Avg Return: {avg_return:>7.2%}  Avg Sharpe: {avg_sharpe:>7.2f}")
        
        return df
    
    return None


def test_single_strategy_detail():
    """Test a single strategy with detailed output."""
    
    print("\n" + "="*80)
    print("DETAILED STRATEGY TEST - RSI Mean Reversion")
    print("="*80)
    
    strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    test_strategy(strategy, symbol='AAPL', days=90, verbose=True)


def main():
    """Main function to run all tests."""
    
    print("="*80)
    print("TRADING STRATEGY TEST SUITE")
    print("="*80)
    
    # Test individual strategies
    print("\n1️⃣ Testing RSI Strategy...")
    rsi_strategy = RSIStrategy(period=14, oversold=30, overbought=70)
    test_strategy(rsi_strategy, 'AAPL', 90)
    
    print("\n2️⃣ Testing Bollinger Bands Strategy...")
    bb_strategy = BollingerBandsStrategy(period=20, num_std=2, strategy_type='mean_reversion')
    test_strategy(bb_strategy, 'AAPL', 90)
    
    print("\n3️⃣ Testing MACD Strategy...")
    macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    test_strategy(macd_strategy, 'AAPL', 90)
    
    # Compare all strategies
    print("\n" + "="*80)
    print("4️⃣ COMPARING ALL STRATEGIES")
    print("="*80)
    
    # Test on different stocks
    for symbol in ['AAPL', 'MSFT', 'NVDA']:
        print(f"\n\n🔍 Testing on {symbol}...")
        df = compare_all_strategies(symbol, days=180)
        
        if df is not None:
            # Save results
            filename = f"strategy_comparison_{symbol}.csv"
            df.to_csv(filename, index=False)
            print(f"\n✓ Results saved to {filename}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    # You can run specific tests or all tests
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--compare':
            compare_all_strategies(sys.argv[2] if len(sys.argv) > 2 else 'AAPL')
        elif sys.argv[1] == '--detail':
            test_single_strategy_detail()
    else:
        main()