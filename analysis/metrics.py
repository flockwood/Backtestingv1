import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.types import BacktestResult, Trade
from .risk_metrics import RiskMetrics
from .rolling_metrics import RollingMetrics
from .trade_analytics import TradeAnalytics


class PerformanceAnalyzer:
    """
    Analyze backtest performance and generate metrics.
    """
    
    @staticmethod
    def print_results(result: BacktestResult) -> None:
        """Print formatted backtest results."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        
        # Basic Performance
        print(f"Initial Capital:      ${result.initial_capital:,.2f}")
        print(f"Final Value:          ${result.final_value:,.2f}")
        print(f"Total Return:         {result.total_return:.2%}")
        print(f"Annualized Return:    {result.annualized_return:.2%}")
        
        # Risk Metrics
        print(f"Sharpe Ratio:         {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:         {result.max_drawdown:.2%}")

        # Enhanced Risk Metrics
        risk_metrics = RiskMetrics(result)
        try:
            sortino = risk_metrics.sortino_ratio()
            calmar = risk_metrics.calmar_ratio()
            print(f"Sortino Ratio:        {sortino:.2f}")
            print(f"Calmar Ratio:         {calmar:.2f}")
        except:
            pass  # Skip if calculation fails
        
        # Trade Statistics
        print(f"Number of Trades:     {result.num_trades}")
        print(f"Win Rate:             {result.win_rate:.2%}")

        # Transaction Costs (if available)
        if hasattr(result, 'total_transaction_costs') and result.total_transaction_costs:
            print(f"Total Trading Costs:  ${result.total_transaction_costs:.2f} ({result.total_transaction_costs/result.initial_capital:.2%})")
        
        # Strategy Info
        print(f"Symbol:               {result.metadata.get('symbol', 'N/A')}")
        print(f"Strategy:             {result.metadata.get('strategy', 'N/A')}")
        print(f"Period:               {result.metadata.get('start_date')} to {result.metadata.get('end_date')}")
        print(f"Commission:           {result.metadata.get('commission', 0):.3%}")
        
        print("="*50)
    
    @staticmethod
    def print_trade_summary(result: BacktestResult, show_all: bool = False) -> None:
        """Print trade summary."""
        if not result.trades:
            print("No trades executed.")
            return
        
        print(f"\nTRADE SUMMARY ({len(result.trades)} trades)")
        print("-" * 60)
        
        trades_to_show = result.trades if show_all else result.trades[-10:]
        
        for i, trade in enumerate(trades_to_show, 1):
            print(f"{i:2d}. {trade.timestamp.date()} | "
                  f"{trade.side.value:4s} | "
                  f"{trade.quantity:8.2f} @ ${trade.price:7.2f} | "
                  f"${trade.value:9.2f}")
        
        if not show_all and len(result.trades) > 10:
            print(f"    ... and {len(result.trades) - 10} more trades")
    
    @staticmethod
    def calculate_monthly_returns(result: BacktestResult) -> pd.DataFrame:
        """Calculate monthly returns from portfolio values."""
        if not result.portfolio_values:
            return pd.DataFrame()
        
        portfolio_df = pd.DataFrame(result.portfolio_values)
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Resample to monthly and calculate returns
        monthly = portfolio_df['total_value'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        return monthly_returns
    
    @staticmethod
    def calculate_trade_analysis(result: BacktestResult) -> Dict[str, Any]:
        """Analyze individual trades for detailed statistics."""
        if not result.trades:
            return {}
        
        # Convert trades to DataFrame for analysis
        trades_data = []
        buy_price = None
        
        for trade in result.trades:
            if trade.side.value == "BUY":
                buy_price = trade.price
            elif trade.side.value == "SELL" and buy_price is not None:
                pnl = (trade.price - buy_price) * trade.quantity - trade.commission
                pnl_pct = (trade.price - buy_price) / buy_price
                
                trades_data.append({
                    'buy_price': buy_price,
                    'sell_price': trade.price,
                    'quantity': trade.quantity,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_days': (trade.timestamp - trades_data[-1]['buy_timestamp'] 
                                    if trades_data else pd.Timedelta(days=0)).days,
                    'buy_timestamp': trade.timestamp
                })
                buy_price = None
        
        if not trades_data:
            return {}
        
        trades_df = pd.DataFrame(trades_data)
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        analysis = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'avg_win_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_trade_duration': trades_df['duration_days'].mean(),
            'profit_factor': (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) 
                            if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf)
        }
        
        return analysis

    @staticmethod
    def print_comprehensive_analysis(result: BacktestResult,
                                   benchmark_returns: Optional[pd.Series] = None) -> None:
        """
        Print comprehensive performance analysis using all advanced metrics.

        Args:
            result: BacktestResult object
            benchmark_returns: Optional benchmark returns for comparison
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*80)

        # Basic results
        PerformanceAnalyzer.print_results(result)

        # Advanced risk metrics
        print("\n" + "="*50)
        print("ADVANCED RISK METRICS")
        print("="*50)

        risk_metrics = RiskMetrics(result)
        risk_data = risk_metrics.calculate_all_metrics()

        print(f"Value at Risk (95%):  {risk_data.get('var_95', 0):.4f}")
        print(f"CVaR (95%):           {risk_data.get('cvar_95', 0):.4f}")
        print(f"Downside Deviation:   {risk_data.get('downside_deviation', 0):.4f}")
        print(f"Omega Ratio:          {risk_data.get('omega_ratio', 0):.2f}")
        print(f"Ulcer Index:          {risk_data.get('ulcer_index', 0):.2f}")
        print(f"Tail Ratio:           {risk_data.get('tail_ratio', 0):.2f}")

        # Drawdown details
        dd_details = risk_data
        if dd_details.get('max_drawdown_duration'):
            print(f"\nDrawdown Analysis:")
            print(f"Max DD Duration:      {dd_details.get('max_drawdown_duration', 0)} days")
            print(f"Recovery Time:        {dd_details.get('recovery_time', 'N/A')} days")

        # Trade analytics
        print("\n" + "="*50)
        print("TRADE ANALYTICS")
        print("="*50)

        trade_analytics = TradeAnalytics(result)
        trade_data = trade_analytics.get_comprehensive_analysis()

        if 'risk_reward' in trade_data:
            rr = trade_data['risk_reward']
            print(f"Profit Factor:        {rr.get('profit_factor', 0):.2f}")
            print(f"Risk/Reward Ratio:    {rr.get('risk_reward_ratio', 0):.2f}")
            print(f"Expectancy:           ${rr.get('expectancy', 0):.2f}")
            print(f"Kelly %:              {rr.get('kelly_percentage', 0)*100:.1f}%")

        if 'win_loss_streaks' in trade_data:
            streaks = trade_data['win_loss_streaks']
            print(f"\nStreak Analysis:")
            print(f"Max Win Streak:       {streaks.get('max_winning_streak', 0)}")
            print(f"Max Loss Streak:      {streaks.get('max_losing_streak', 0)}")

        # Monthly returns table
        monthly_table = PerformanceAnalyzer.create_monthly_returns_table(result)
        if not monthly_table.empty:
            print("\n" + "="*50)
            print("MONTHLY RETURNS (%)")
            print("="*50)
            print(monthly_table.round(2))

        print("\n" + "="*80)

    @staticmethod
    def get_performance_summary(result: BacktestResult) -> Dict[str, Any]:
        """
        Get comprehensive performance summary as dictionary.

        Args:
            result: BacktestResult object

        Returns:
            Dictionary with all performance metrics
        """
        summary = {
            'basic_metrics': {
                'initial_capital': result.initial_capital,
                'final_value': result.final_value,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'num_trades': result.num_trades,
                'win_rate': result.win_rate
            }
        }

        # Add transaction costs if available
        if hasattr(result, 'total_transaction_costs'):
            summary['transaction_costs'] = {
                'total_commission': getattr(result, 'total_commission', 0),
                'total_slippage': getattr(result, 'total_slippage', 0),
                'total_spread_cost': getattr(result, 'total_spread_cost', 0),
                'total_transaction_costs': getattr(result, 'total_transaction_costs', 0)
            }

        # Advanced risk metrics
        try:
            risk_metrics = RiskMetrics(result)
            summary['risk_metrics'] = risk_metrics.calculate_all_metrics()
        except Exception:
            summary['risk_metrics'] = {}

        # Trade analytics
        try:
            trade_analytics = TradeAnalytics(result)
            trade_data = trade_analytics.get_comprehensive_analysis()
            summary['trade_analytics'] = {
                'risk_reward': trade_data.get('risk_reward', {}),
                'win_loss_streaks': trade_data.get('win_loss_streaks', {}),
                'holding_periods': trade_data.get('holding_periods', {}),
                'slippage_analysis': trade_data.get('slippage_analysis', {})
            }
        except Exception:
            summary['trade_analytics'] = {}

        return summary
    
    @staticmethod
    def print_detailed_analysis(result: BacktestResult) -> None:
        """Print detailed trade analysis."""
        analysis = PerformanceAnalyzer.calculate_trade_analysis(result)
        
        if not analysis:
            print("No completed trade pairs for detailed analysis.")
            return
        
        print(f"\nDETAILED TRADE ANALYSIS")
        print("-" * 40)
        print(f"Total Trade Pairs:    {analysis['total_trades']}")
        print(f"Winning Trades:       {analysis['winning_trades']} ({analysis['win_rate']:.1%})")
        print(f"Losing Trades:        {analysis['losing_trades']}")
        print(f"Average Win:          ${analysis['avg_win']:.2f} ({analysis['avg_win_pct']:.2%})")
        print(f"Average Loss:         ${analysis['avg_loss']:.2f} ({analysis['avg_loss_pct']:.2%})")
        print(f"Largest Win:          ${analysis['largest_win']:.2f}")
        print(f"Largest Loss:         ${analysis['largest_loss']:.2f}")
        print(f"Profit Factor:        {analysis['profit_factor']:.2f}")
        print(f"Avg Trade Duration:   {analysis['avg_trade_duration']:.1f} days")

    @staticmethod
    def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Args:
            initial_value: Starting portfolio value
            final_value: Ending portfolio value
            years: Number of years

        Returns:
            CAGR as decimal
        """
        if initial_value <= 0 or years <= 0:
            return 0.0
        return (final_value / initial_value) ** (1 / years) - 1

    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio vs benchmark.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series

        Returns:
            Information ratio
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Align series
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
        if len(aligned) < 2:
            return 0.0

        excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        tracking_error = excess_returns.std()

        if tracking_error == 0:
            return 0.0

        return excess_returns.mean() / tracking_error

    @staticmethod
    def create_monthly_returns_table(result: BacktestResult) -> pd.DataFrame:
        """
        Create monthly returns table.

        Args:
            result: BacktestResult object

        Returns:
            DataFrame with monthly returns by year
        """
        monthly_returns = PerformanceAnalyzer.calculate_monthly_returns(result)

        if len(monthly_returns) == 0:
            return pd.DataFrame()

        # Create year-month pivot table
        monthly_df = monthly_returns.to_frame('returns')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month_name()

        pivot_table = monthly_df.pivot_table(
            values='returns',
            index='year',
            columns='month',
            aggfunc='first'
        )

        # Reorder months
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']

        # Only include months that exist in the data
        existing_months = [m for m in month_order if m in pivot_table.columns]
        pivot_table = pivot_table[existing_months]

        # Add yearly totals
        pivot_table['Year'] = (pivot_table + 1).prod(axis=1) - 1

        return pivot_table * 100  # Convert to percentages