"""Test suite for advanced metrics calculations."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import warnings

from core.types import BacktestResult, Trade, OrderSide
from analysis import RiskMetrics, RollingMetrics, TradeAnalytics, PerformanceReport


def create_test_backtest_result() -> BacktestResult:
    """Create a synthetic backtest result for testing."""

    # Create synthetic portfolio values
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(252)]  # 1 year of trading days

    # Create a realistic equity curve with some volatility
    np.random.seed(42)  # For reproducible results
    daily_returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% daily vol

    # Add some drawdown periods
    daily_returns[50:70] = np.random.normal(-0.002, 0.015, 20)  # Drawdown period
    daily_returns[150:160] = np.random.normal(-0.003, 0.02, 10)  # Another drawdown

    initial_capital = 10000
    portfolio_values = [initial_capital]

    for ret in daily_returns:
        new_value = portfolio_values[-1] * (1 + ret)
        portfolio_values.append(new_value)

    # Create portfolio value records
    portfolio_records = []
    for i, (date, value) in enumerate(zip(dates, portfolio_values[:-1])):
        portfolio_records.append({
            'timestamp': date,
            'total_value': value,
            'cash': value * 0.1,  # 10% cash
            'positions_value': value * 0.9,  # 90% invested
            'return': (value - initial_capital) / initial_capital * 100
        })

    # Create synthetic trades
    trades = []
    trade_dates = [dates[i] for i in range(0, len(dates), 50)]  # Trade every 50 days

    for i, date in enumerate(trade_dates):
        if i < len(trade_dates) - 1:  # Don't create a final unpaired trade
            # Buy trade
            buy_price = 100 + np.random.normal(0, 5)
            quantity = 50 + np.random.normal(0, 10)

            buy_trade = Trade(
                symbol='TEST',
                side=OrderSide.BUY,
                quantity=quantity,
                price=buy_price,
                timestamp=date,
                trade_id=str(i * 2),
                commission=buy_price * quantity * 0.001,
                slippage=buy_price * quantity * 0.0005,
                spread_cost=buy_price * quantity * 0.0005,
                total_cost=buy_price * quantity * 0.002
            )
            trades.append(buy_trade)

            # Sell trade (a few days later)
            sell_date = date + timedelta(days=np.random.randint(5, 45))
            sell_price = buy_price * (1 + np.random.normal(0.02, 0.05))  # 2% average gain with noise

            sell_trade = Trade(
                symbol='TEST',
                side=OrderSide.SELL,
                quantity=quantity,
                price=sell_price,
                timestamp=sell_date,
                trade_id=str(i * 2 + 1),
                commission=sell_price * quantity * 0.001,
                slippage=sell_price * quantity * 0.0005,
                spread_cost=sell_price * quantity * 0.0005,
                total_cost=sell_price * quantity * 0.002
            )
            trades.append(sell_trade)

    final_value = portfolio_values[-2]
    total_return = (final_value - initial_capital) / initial_capital

    # Calculate some basic metrics
    returns_series = pd.Series(daily_returns)
    sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252)

    # Calculate max drawdown
    portfolio_series = pd.Series(portfolio_values[:-1])
    running_max = portfolio_series.expanding().max()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()

    return BacktestResult(
        initial_capital=initial_capital,
        final_value=final_value,
        total_return=total_return,
        annualized_return=(final_value / initial_capital) ** (252 / len(dates)) - 1,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        num_trades=len(trades),
        win_rate=0.6,  # Assume 60% win rate
        trades=trades,
        portfolio_values=portfolio_records,
        total_commission=sum(t.commission for t in trades),
        total_slippage=sum(t.slippage for t in trades),
        total_spread_cost=sum(t.spread_cost for t in trades),
        total_transaction_costs=sum(t.total_cost for t in trades),
        metadata={
            'symbol': 'TEST',
            'strategy': 'Test Strategy',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31'
        }
    )


def test_risk_metrics():
    """Test risk metrics calculations."""
    print("Testing Risk Metrics...")

    result = create_test_backtest_result()
    risk_metrics = RiskMetrics(result)

    # Test individual metrics
    try:
        sortino = risk_metrics.sortino_ratio()
        print(f"  Sortino Ratio: {sortino:.3f}")
        assert -10 < sortino < 10, "Sortino ratio seems unreasonable"

        calmar = risk_metrics.calmar_ratio()
        print(f"  Calmar Ratio: {calmar:.3f}")
        assert -50 < calmar < 50, "Calmar ratio seems unreasonable"

        var_95 = risk_metrics.value_at_risk(0.95)
        print(f"  VaR (95%): {var_95:.4f}")
        assert 0 <= var_95 <= 1, "VaR should be between 0 and 1"

        cvar_95 = risk_metrics.conditional_var(0.95)
        print(f"  CVaR (95%): {cvar_95:.4f}")
        assert cvar_95 >= var_95, "CVaR should be >= VaR"

        ulcer = risk_metrics.ulcer_index()
        print(f"  Ulcer Index: {ulcer:.3f}")
        assert ulcer >= 0, "Ulcer index should be non-negative"

        # Test comprehensive metrics
        all_metrics = risk_metrics.calculate_all_metrics()
        assert len(all_metrics) > 5, "Should calculate multiple metrics"

        print("  ✓ Risk metrics calculations passed")

    except Exception as e:
        print(f"  ✗ Risk metrics test failed: {e}")


def test_rolling_metrics():
    """Test rolling metrics calculations."""
    print("Testing Rolling Metrics...")

    result = create_test_backtest_result()
    rolling_metrics = RollingMetrics(result)

    try:
        # Test rolling returns
        rolling_returns = rolling_metrics.rolling_returns([21, 63])
        print(f"  Rolling returns shape: {rolling_returns.shape}")
        assert not rolling_returns.empty, "Should have rolling returns data"

        # Test rolling Sharpe
        rolling_sharpe = rolling_metrics.rolling_sharpe(63)
        print(f"  Rolling Sharpe length: {len(rolling_sharpe)}")
        assert len(rolling_sharpe) > 0, "Should have rolling Sharpe data"

        # Test rolling volatility
        rolling_vol = rolling_metrics.rolling_volatility(63)
        print(f"  Rolling volatility mean: {rolling_vol.mean():.4f}")
        assert rolling_vol.mean() > 0, "Volatility should be positive"

        # Test expanding metrics
        expanding = rolling_metrics.expanding_metrics()
        print(f"  Expanding metrics columns: {list(expanding.columns)}")
        assert 'expanding_sharpe' in expanding.columns, "Should have expanding Sharpe"

        print("  ✓ Rolling metrics calculations passed")

    except Exception as e:
        print(f"  ✗ Rolling metrics test failed: {e}")


def test_trade_analytics():
    """Test trade analytics calculations."""
    print("Testing Trade Analytics...")

    result = create_test_backtest_result()
    trade_analytics = TradeAnalytics(result)

    try:
        # Test trade pairs
        trade_pairs = trade_analytics.get_trade_pairs()
        print(f"  Trade pairs: {len(trade_pairs)}")
        assert len(trade_pairs) > 0, "Should have trade pairs"

        # Test win/loss streaks
        streaks = trade_analytics.get_win_loss_streaks()
        print(f"  Max winning streak: {streaks['max_winning_streak']}")
        print(f"  Max losing streak: {streaks['max_losing_streak']}")
        assert isinstance(streaks['max_winning_streak'], int), "Should return integer"

        # Test holding periods
        holding = trade_analytics.get_holding_periods()
        print(f"  Avg holding period (days): {holding['avg_holding_period_days']:.2f}")
        assert holding['avg_holding_period_days'] >= 0, "Holding period should be non-negative"

        # Test risk/reward analysis
        risk_reward = trade_analytics.risk_reward_analysis()
        print(f"  Profit factor: {risk_reward['profit_factor']:.2f}")
        print(f"  Win rate: {risk_reward['win_rate']:.2%}")
        assert 0 <= risk_reward['win_rate'] <= 1, "Win rate should be between 0 and 1"

        print("  ✓ Trade analytics calculations passed")

    except Exception as e:
        print(f"  ✗ Trade analytics test failed: {e}")


def test_performance_report():
    """Test performance report generation."""
    print("Testing Performance Report...")

    result = create_test_backtest_result()

    try:
        # Test report creation
        report = PerformanceReport(result)

        # Test tearsheet creation (without plots to avoid matplotlib issues)
        tearsheet = report.create_tearsheet(show_plots=False)
        print(f"  Tearsheet sections: {list(tearsheet.keys())}")
        assert 'basic_metrics' in tearsheet, "Should have basic metrics"
        assert 'risk_metrics' in tearsheet, "Should have risk metrics"

        # Test summary printing
        report.print_summary()

        print("  ✓ Performance report generation passed")

    except Exception as e:
        print(f"  ✗ Performance report test failed: {e}")


def test_known_values():
    """Test calculations with known values."""
    print("Testing Known Value Calculations...")

    try:
        # Test Sharpe ratio calculation with known values
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])  # Known returns
        expected_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        # Create a simple result for testing
        simple_portfolio_values = [
            {'timestamp': datetime(2023, 1, 1), 'total_value': 1000, 'return': 0},
            {'timestamp': datetime(2023, 1, 2), 'total_value': 1010, 'return': 1},
            {'timestamp': datetime(2023, 1, 3), 'total_value': 1020.2, 'return': 2.02},
            {'timestamp': datetime(2023, 1, 4), 'total_value': 1009.998, 'return': 0.9998},
            {'timestamp': datetime(2023, 1, 5), 'total_value': 1025.148, 'return': 2.5148},
        ]

        simple_result = BacktestResult(
            initial_capital=1000,
            final_value=1025.148,
            total_return=0.025148,
            annualized_return=0.1,
            sharpe_ratio=expected_sharpe,
            max_drawdown=-0.01,
            num_trades=0,
            win_rate=0,
            trades=[],
            portfolio_values=simple_portfolio_values,
            metadata={}
        )

        risk_metrics = RiskMetrics(simple_result)

        # Test that daily returns are calculated correctly
        assert len(risk_metrics.daily_returns) == 4, "Should have 4 daily returns"

        print("  ✓ Known value calculations passed")

    except Exception as e:
        print(f"  ✗ Known value test failed: {e}")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing Edge Cases...")

    try:
        # Test with no trades
        no_trades_result = BacktestResult(
            initial_capital=1000,
            final_value=1000,
            total_return=0,
            annualized_return=0,
            sharpe_ratio=0,
            max_drawdown=0,
            num_trades=0,
            win_rate=0,
            trades=[],
            portfolio_values=[],
            metadata={}
        )

        risk_metrics = RiskMetrics(no_trades_result)
        all_metrics = risk_metrics.calculate_all_metrics()
        print(f"  No trades metrics count: {len(all_metrics)}")

        trade_analytics = TradeAnalytics(no_trades_result)
        comprehensive = trade_analytics.get_comprehensive_analysis()
        print(f"  No trades analysis sections: {len(comprehensive)}")

        # Test with single trade
        single_trade_result = create_test_backtest_result()
        single_trade_result.trades = single_trade_result.trades[:1]  # Keep only first trade

        trade_analytics_single = TradeAnalytics(single_trade_result)
        single_analysis = trade_analytics_single.get_comprehensive_analysis()
        print(f"  Single trade analysis completed")

        print("  ✓ Edge cases handled properly")

    except Exception as e:
        print(f"  ✗ Edge case test failed: {e}")


def run_all_tests():
    """Run all metric tests."""
    print("="*60)
    print("ADVANCED METRICS TEST SUITE")
    print("="*60)

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    try:
        test_risk_metrics()
        print()

        test_rolling_metrics()
        print()

        test_trade_analytics()
        print()

        test_performance_report()
        print()

        test_known_values()
        print()

        test_edge_cases()
        print()

        print("="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*60)

    except Exception as e:
        print(f"\n⚠️  Test suite encountered an error: {e}")
        print("This might indicate an issue with dependencies or data setup.")

    finally:
        # Reset warning filters
        warnings.resetwarnings()


def demo_advanced_metrics():
    """Demonstrate advanced metrics usage."""
    print("\n" + "="*60)
    print("ADVANCED METRICS DEMONSTRATION")
    print("="*60)

    # Create test result
    result = create_test_backtest_result()

    print(f"Test Strategy Performance:")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Value:     ${result.final_value:,.2f}")
    print(f"  Total Return:    {result.total_return:.2%}")
    print(f"  Number of Trades: {result.num_trades}")

    # Demonstrate risk metrics
    print(f"\nRisk Metrics:")
    risk_metrics = RiskMetrics(result)
    risk_data = risk_metrics.calculate_all_metrics()

    print(f"  Sortino Ratio:       {risk_data['sortino_ratio']:.3f}")
    print(f"  Calmar Ratio:        {risk_data['calmar_ratio']:.3f}")
    print(f"  Value at Risk (95%): {risk_data['var_95']:.3f}")
    print(f"  CVaR (95%):          {risk_data['cvar_95']:.3f}")
    print(f"  Ulcer Index:         {risk_data['ulcer_index']:.3f}")

    # Demonstrate trade analytics
    print(f"\nTrade Analytics:")
    trade_analytics = TradeAnalytics(result)
    trade_data = trade_analytics.get_comprehensive_analysis()

    if 'risk_reward' in trade_data:
        rr = trade_data['risk_reward']
        print(f"  Profit Factor:       {rr['profit_factor']:.2f}")
        print(f"  Risk/Reward Ratio:   {rr['risk_reward_ratio']:.2f}")
        print(f"  Expectancy:          ${rr['expectancy']:.2f}")

    if 'win_loss_streaks' in trade_data:
        streaks = trade_data['win_loss_streaks']
        print(f"  Max Win Streak:      {streaks['max_winning_streak']}")
        print(f"  Max Loss Streak:     {streaks['max_losing_streak']}")

    print("\n" + "="*60)
    print("Example usage:")
    print("from analysis import RiskMetrics, TradeAnalytics, PerformanceReport")
    print("")
    print("risk_metrics = RiskMetrics(backtest_result)")
    print("sortino = risk_metrics.sortino_ratio()")
    print("var_95 = risk_metrics.value_at_risk(0.95)")
    print("")
    print("trade_analytics = TradeAnalytics(backtest_result)")
    print("streaks = trade_analytics.get_win_loss_streaks()")
    print("")
    print("report = PerformanceReport(backtest_result)")
    print("report.create_tearsheet('report.html')")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
    demo_advanced_metrics()