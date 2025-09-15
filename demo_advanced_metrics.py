"""
Demonstration of advanced metrics capabilities.
Run after a backtest to see comprehensive analysis.
"""

from datetime import datetime, timedelta

# Import components
from data import AlpacaDataLoader
from strategies import SMAStrategy
from core import BacktestEngine
from core.costs import SimpleCostModel
from analysis import (
    PerformanceAnalyzer,
    RiskMetrics,
    RollingMetrics,
    TradeAnalytics,
    PerformanceReport
)


def run_enhanced_backtest():
    """Run a backtest and demonstrate advanced analytics."""

    print("="*80)
    print("ADVANCED METRICS DEMONSTRATION")
    print("="*80)

    # Configuration
    symbol = 'AAPL'
    initial_capital = 10000

    # Date range (last 60 days for quick demo)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    print(f"Running backtest: {symbol} from {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("-" * 80)

    try:
        # Initialize components
        data_loader = AlpacaDataLoader()
        strategy = SMAStrategy(short_window=5, long_window=15)

        cost_model = SimpleCostModel(
            fixed_fee=0,
            percentage_fee=0.001,  # 0.1%
            base_slippage_bps=3,
            market_impact_coefficient=5,
            bid_ask_spread_bps=5
        )

        engine = BacktestEngine(
            initial_capital=initial_capital,
            cost_model=cost_model,
            position_size=0.9
        )

        # Run backtest
        result = engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        # Demonstrate advanced analytics
        demonstrate_advanced_analysis(result)

    except Exception as e:
        print(f"Error running backtest: {e}")
        print("You can still see the demo with synthetic data...")
        demonstrate_with_synthetic_data()


def demonstrate_advanced_analysis(result):
    """Demonstrate advanced analysis capabilities."""

    print("\n" + "="*80)
    print("1. ENHANCED PERFORMANCE ANALYSIS")
    print("="*80)

    # Standard analysis with enhancements
    PerformanceAnalyzer.print_results(result)

    print("\n" + "="*80)
    print("2. ADVANCED RISK METRICS")
    print("="*80)

    risk_metrics = RiskMetrics(result)
    risk_data = risk_metrics.calculate_all_metrics()

    print(f"Advanced Risk Metrics:")
    print(f"  Sortino Ratio:        {risk_data.get('sortino_ratio', 'N/A'):.3f}")
    print(f"  Calmar Ratio:         {risk_data.get('calmar_ratio', 'N/A'):.3f}")
    print(f"  Value at Risk (95%):  {risk_data.get('var_95', 'N/A'):.4f}")
    print(f"  CVaR (95%):           {risk_data.get('cvar_95', 'N/A'):.4f}")
    print(f"  Downside Deviation:   {risk_data.get('downside_deviation', 'N/A'):.4f}")
    print(f"  Omega Ratio:          {risk_data.get('omega_ratio', 'N/A'):.2f}")
    print(f"  Ulcer Index:          {risk_data.get('ulcer_index', 'N/A'):.3f}")
    print(f"  Tail Ratio:           {risk_data.get('tail_ratio', 'N/A'):.2f}")

    # Drawdown analysis
    dd_details = risk_metrics.max_drawdown_details()
    if dd_details.get('max_drawdown_duration'):
        print(f"\nDrawdown Analysis:")
        print(f"  Max DD Duration:      {dd_details.get('max_drawdown_duration', 'N/A')} periods")
        print(f"  Recovery Time:        {dd_details.get('recovery_time', 'N/A')}")
        print(f"  DD Start Date:        {dd_details.get('start_date', 'N/A')}")
        print(f"  DD End Date:          {dd_details.get('end_date', 'N/A')}")

    print("\n" + "="*80)
    print("3. ROLLING PERFORMANCE METRICS")
    print("="*80)

    rolling_metrics = RollingMetrics(result)

    # Rolling Sharpe ratios
    for window in [21, 63]:
        rolling_sharpe = rolling_metrics.rolling_sharpe(window)
        if len(rolling_sharpe.dropna()) > 0:
            current_sharpe = rolling_sharpe.dropna().iloc[-1]
            mean_sharpe = rolling_sharpe.mean()
            print(f"  {window}-Day Rolling Sharpe:")
            print(f"    Current:            {current_sharpe:.3f}")
            print(f"    Average:            {mean_sharpe:.3f}")

    # Performance over periods
    periods = rolling_metrics.performance_over_periods()
    if not periods.empty:
        print(f"\nPerformance Over Different Periods:")
        for _, row in periods.iterrows():
            print(f"  {row['period']:>3}: {row['total_return']:>7.2%} "
                  f"(Annualized: {row['annualized_return']:>7.2%})")

    print("\n" + "="*80)
    print("4. COMPREHENSIVE TRADE ANALYSIS")
    print("="*80)

    trade_analytics = TradeAnalytics(result)
    trade_data = trade_analytics.get_comprehensive_analysis()

    # Risk/Reward Analysis
    if 'risk_reward' in trade_data and trade_data['risk_reward']:
        rr = trade_data['risk_reward']
        print(f"Risk/Reward Analysis:")
        print(f"  Profit Factor:        {rr.get('profit_factor', 'N/A'):.2f}")
        print(f"  Risk/Reward Ratio:    {rr.get('risk_reward_ratio', 'N/A'):.2f}")
        print(f"  Expectancy:           ${rr.get('expectancy', 'N/A'):.2f}")
        print(f"  Kelly Percentage:     {rr.get('kelly_percentage', 'N/A')*100:.1f}%")
        print(f"  Average Win:          ${rr.get('avg_win', 'N/A'):.2f}")
        print(f"  Average Loss:         ${abs(rr.get('avg_loss', 0)):.2f}")

    # Win/Loss Streaks
    if 'win_loss_streaks' in trade_data:
        streaks = trade_data['win_loss_streaks']
        print(f"\nStreak Analysis:")
        print(f"  Max Winning Streak:   {streaks.get('max_winning_streak', 'N/A')}")
        print(f"  Max Losing Streak:    {streaks.get('max_losing_streak', 'N/A')}")
        print(f"  Current Streak:       {streaks.get('current_streak', 'N/A')} ({streaks.get('current_streak_type', 'N/A')})")

    # Holding Periods
    if 'holding_periods' in trade_data:
        holding = trade_data['holding_periods']
        print(f"\nHolding Period Analysis:")
        print(f"  Avg Holding Period:   {holding.get('avg_holding_period_days', 'N/A'):.1f} days")
        print(f"  Winning Trades Avg:   {holding.get('avg_winning_holding_period', 'N/A')/24:.1f} days")
        print(f"  Losing Trades Avg:    {holding.get('avg_losing_holding_period', 'N/A')/24:.1f} days")

    # Transaction Cost Analysis
    if 'slippage_analysis' in trade_data:
        slippage = trade_data['slippage_analysis']
        print(f"\nTransaction Cost Analysis:")
        print(f"  Total Slippage:       ${slippage.get('total_slippage', 'N/A'):.2f}")
        print(f"  Avg Slippage/Trade:   ${slippage.get('avg_slippage_per_trade', 'N/A'):.2f}")
        print(f"  Avg Slippage %:       {slippage.get('avg_slippage_pct', 'N/A'):.3f}%")

    print("\n" + "="*80)
    print("5. COMPREHENSIVE PERFORMANCE REPORT")
    print("="*80)

    # Generate comprehensive report
    report = PerformanceReport(result)
    report.print_summary()

    # Optional: Create detailed reports
    try:
        # Create tearsheet (data only, no plots without matplotlib)
        tearsheet = report.create_tearsheet(show_plots=False)
        print(f"\nTearsheet generated with {len(tearsheet)} sections")

        # Could save to Excel if openpyxl is available
        # report.export_to_excel('performance_report.xlsx')

        # Could save to HTML if matplotlib is available
        # report.create_tearsheet('performance_report.html', show_plots=True)

    except Exception as e:
        print(f"Note: Full reporting requires additional packages: {e}")


def demonstrate_with_synthetic_data():
    """Demonstrate with synthetic data if real backtest fails."""
    print("\n" + "="*60)
    print("SYNTHETIC DATA DEMONSTRATION")
    print("="*60)

    # This would use the create_test_backtest_result from test file
    print("For a full demonstration with synthetic data, run:")
    print("python3 test_advanced_metrics.py")


def show_usage_examples():
    """Show code examples for using advanced metrics."""

    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)

    usage_examples = """
# After running a backtest and getting a BacktestResult:

from analysis import RiskMetrics, RollingMetrics, TradeAnalytics, PerformanceReport

# 1. Advanced Risk Analysis
risk_metrics = RiskMetrics(backtest_result)
sortino = risk_metrics.sortino_ratio(target_return=0.02)
var_95 = risk_metrics.value_at_risk(confidence=0.95)
all_risk_metrics = risk_metrics.calculate_all_metrics()

# 2. Rolling Performance Analysis
rolling = RollingMetrics(backtest_result)
rolling_sharpe = rolling.rolling_sharpe(window=63)  # 3-month
rolling_vol = rolling.rolling_volatility(window=21)  # 1-month
performance_periods = rolling.performance_over_periods()

# 3. Detailed Trade Analysis
trade_analyzer = TradeAnalytics(backtest_result)
trade_pairs = trade_analyzer.get_trade_pairs()
win_streaks = trade_analyzer.get_win_loss_streaks()
risk_reward = trade_analyzer.risk_reward_analysis()
mae_mfe = trade_analyzer.calculate_mae_mfe()

# 4. Comprehensive Reporting
report = PerformanceReport(backtest_result)
report.print_summary()

# Generate tearsheet with charts (requires matplotlib)
tearsheet = report.create_tearsheet('performance_report.html')

# Export to Excel (requires openpyxl)
report.export_to_excel('performance_data.xlsx')

# 5. Enhanced Standard Analysis
from analysis import PerformanceAnalyzer

# Now includes advanced metrics automatically
PerformanceAnalyzer.print_results(backtest_result)
PerformanceAnalyzer.print_comprehensive_analysis(backtest_result)

# Get all metrics as dictionary
summary = PerformanceAnalyzer.get_performance_summary(backtest_result)
    """

    print(usage_examples)

    print("="*80)
    print("INSTALLATION NOTES")
    print("="*80)
    print("For full functionality, install optional dependencies:")
    print("  pip install scipy matplotlib openpyxl")
    print("")
    print("Core metrics work without these dependencies, but you'll get:")
    print("  - Basic risk metrics only (without scipy)")
    print("  - No charts or visualizations (without matplotlib)")
    print("  - No Excel export (without openpyxl)")
    print("="*80)


if __name__ == "__main__":
    run_enhanced_backtest()
    show_usage_examples()