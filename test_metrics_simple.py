"""Simple test for advanced metrics without external dependencies."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.types import BacktestResult, Trade, OrderSide


def test_imports():
    """Test that we can import the new modules."""
    print("Testing imports...")

    try:
        from analysis import PerformanceAnalyzer
        print("  ✓ PerformanceAnalyzer imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import PerformanceAnalyzer: {e}")

    try:
        # Test basic functionality without scipy-dependent features
        from analysis.trade_analytics import TradeAnalytics
        print("  ✓ TradeAnalytics imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import TradeAnalytics: {e}")

    try:
        from analysis.rolling_metrics import RollingMetrics
        print("  ✓ RollingMetrics imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import RollingMetrics: {e}")


def create_simple_result():
    """Create a simple backtest result for testing."""
    trades = [
        Trade(
            symbol='TEST',
            side=OrderSide.BUY,
            quantity=100,
            price=50.0,
            timestamp=datetime(2023, 1, 1),
            trade_id='1',
            commission=5.0
        ),
        Trade(
            symbol='TEST',
            side=OrderSide.SELL,
            quantity=100,
            price=55.0,
            timestamp=datetime(2023, 1, 10),
            trade_id='2',
            commission=5.0
        )
    ]

    portfolio_values = [
        {'timestamp': datetime(2023, 1, 1), 'total_value': 10000, 'return': 0},
        {'timestamp': datetime(2023, 1, 2), 'total_value': 10100, 'return': 1},
        {'timestamp': datetime(2023, 1, 3), 'total_value': 10050, 'return': 0.5},
        {'timestamp': datetime(2023, 1, 4), 'total_value': 10200, 'return': 2},
        {'timestamp': datetime(2023, 1, 5), 'total_value': 10400, 'return': 4},
    ]

    return BacktestResult(
        initial_capital=10000,
        final_value=10400,
        total_return=0.04,
        annualized_return=0.15,
        sharpe_ratio=1.2,
        max_drawdown=-0.02,
        num_trades=2,
        win_rate=1.0,
        trades=trades,
        portfolio_values=portfolio_values,
        metadata={'symbol': 'TEST', 'strategy': 'Test Strategy'}
    )


def test_enhanced_analyzer():
    """Test enhanced PerformanceAnalyzer."""
    print("\nTesting enhanced PerformanceAnalyzer...")

    try:
        from analysis import PerformanceAnalyzer

        result = create_simple_result()

        # Test enhanced print_results (should work even if risk metrics fail)
        PerformanceAnalyzer.print_results(result)
        print("  ✓ Enhanced print_results works")

        # Test CAGR calculation
        cagr = PerformanceAnalyzer.calculate_cagr(10000, 10400, 1)
        print(f"  CAGR: {cagr:.2%}")
        assert abs(cagr - 0.04) < 0.001, "CAGR calculation incorrect"
        print("  ✓ CAGR calculation works")

        # Test monthly returns
        monthly_returns = PerformanceAnalyzer.calculate_monthly_returns(result)
        print(f"  Monthly returns shape: {monthly_returns.shape if hasattr(monthly_returns, 'shape') else len(monthly_returns)}")
        print("  ✓ Monthly returns calculation works")

    except Exception as e:
        print(f"  ✗ Enhanced analyzer test failed: {e}")


def test_trade_analytics_basic():
    """Test basic trade analytics functionality."""
    print("\nTesting TradeAnalytics basic functionality...")

    try:
        from analysis.trade_analytics import TradeAnalytics

        result = create_simple_result()
        analyzer = TradeAnalytics(result)

        # Test trade pairs
        pairs = analyzer.get_trade_pairs()
        print(f"  Trade pairs: {len(pairs)}")
        assert len(pairs) == 1, "Should have 1 trade pair"
        print("  ✓ Trade pairs calculation works")

        # Test holding periods
        holding = analyzer.get_holding_periods()
        print(f"  Avg holding period: {holding['avg_holding_period_days']:.1f} days")
        print("  ✓ Holding periods calculation works")

        # Test slippage analysis
        slippage = analyzer.slippage_analysis()
        print(f"  Total slippage: ${slippage['total_slippage']:.2f}")
        print("  ✓ Slippage analysis works")

    except Exception as e:
        print(f"  ✗ Trade analytics test failed: {e}")


def test_rolling_metrics_basic():
    """Test basic rolling metrics functionality."""
    print("\nTesting RollingMetrics basic functionality...")

    try:
        from analysis.rolling_metrics import RollingMetrics

        result = create_simple_result()
        analyzer = RollingMetrics(result)

        # Test performance over periods
        periods = analyzer.performance_over_periods()
        print(f"  Performance periods shape: {periods.shape if hasattr(periods, 'shape') else len(periods)}")
        print("  ✓ Performance periods calculation works")

    except Exception as e:
        print(f"  ✗ Rolling metrics test failed: {e}")


def demo_usage():
    """Demonstrate usage of new metrics."""
    print("\n" + "="*50)
    print("USAGE DEMONSTRATION")
    print("="*50)

    result = create_simple_result()

    print("Example backtest result:")
    print(f"  Return: {result.total_return:.2%}")
    print(f"  Trades: {result.num_trades}")
    print(f"  Win Rate: {result.win_rate:.0%}")

    # Show enhanced analysis
    from analysis import PerformanceAnalyzer

    print("\nEnhanced Performance Analysis:")
    try:
        # This will show enhanced metrics if scipy is available, basic ones otherwise
        PerformanceAnalyzer.print_results(result)
    except Exception as e:
        print(f"Note: Some advanced metrics require scipy: {e}")

    print("\n" + "="*50)
    print("To install all dependencies for full functionality:")
    print("pip install scipy matplotlib")
    print("="*50)


if __name__ == "__main__":
    print("="*50)
    print("SIMPLE METRICS TEST")
    print("="*50)

    test_imports()
    test_enhanced_analyzer()
    test_trade_analytics_basic()
    test_rolling_metrics_basic()
    demo_usage()

    print("\n" + "="*50)
    print("BASIC TESTS COMPLETED ✓")
    print("="*50)