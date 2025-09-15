"""Comprehensive performance reporting for backtesting results."""

import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mdates = None
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import io
import base64
from pathlib import Path

from core.types import BacktestResult
from .risk_metrics import RiskMetrics
from .rolling_metrics import RollingMetrics
from .trade_analytics import TradeAnalytics
from .metrics import PerformanceAnalyzer


class PerformanceReport:
    """Generate comprehensive performance reports for backtesting results."""

    def __init__(self, result: BacktestResult, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize performance report generator.

        Args:
            result: BacktestResult object
            benchmark_returns: Optional benchmark returns for comparison
        """
        self.result = result
        self.benchmark_returns = benchmark_returns

        # Initialize analysis modules
        self.risk_metrics = RiskMetrics(result)
        self.rolling_metrics = RollingMetrics(result)
        self.trade_analytics = TradeAnalytics(result)

        # Prepare portfolio data
        if result.portfolio_values:
            self.portfolio_df = pd.DataFrame(result.portfolio_values)
            self.portfolio_df['timestamp'] = pd.to_datetime(self.portfolio_df['timestamp'])
            self.portfolio_df.set_index('timestamp', inplace=True)
            self.portfolio_df['daily_returns'] = self.portfolio_df['total_value'].pct_change()
        else:
            self.portfolio_df = pd.DataFrame()

    def create_tearsheet(self, output_path: Optional[str] = None,
                        show_plots: bool = True) -> Dict[str, Any]:
        """
        Create a comprehensive performance tearsheet.

        Args:
            output_path: Path to save HTML report (optional)
            show_plots: Whether to display plots

        Returns:
            Dictionary with all analysis data
        """
        # Collect all metrics
        tearsheet_data = {
            'basic_metrics': self._get_basic_metrics(),
            'risk_metrics': self._get_risk_metrics(),
            'trade_analytics': self._get_trade_analytics(),
            'rolling_metrics': self._get_rolling_metrics(),
            'drawdown_analysis': self._get_drawdown_analysis(),
            'monthly_returns': self._get_monthly_returns(),
            'charts': {}
        }

        # Generate charts (only if matplotlib is available)
        if (show_plots or output_path) and HAS_MATPLOTLIB:
            tearsheet_data['charts'] = self._create_charts()
        elif not HAS_MATPLOTLIB and (show_plots or output_path):
            print("Note: Charts require matplotlib. Install with: pip install matplotlib")

        # Generate HTML report if path provided
        if output_path:
            self._generate_html_report(tearsheet_data, output_path)

        return tearsheet_data

    def _get_basic_metrics(self) -> Dict[str, Any]:
        """Get basic performance metrics."""
        return {
            'initial_capital': self.result.initial_capital,
            'final_value': self.result.final_value,
            'total_return': self.result.total_return,
            'annualized_return': self.result.annualized_return,
            'sharpe_ratio': self.result.sharpe_ratio,
            'max_drawdown': self.result.max_drawdown,
            'num_trades': self.result.num_trades,
            'win_rate': self.result.win_rate,
            'volatility': self.portfolio_df['daily_returns'].std() * np.sqrt(252) if len(self.portfolio_df) > 0 else 0,
            'total_trading_costs': getattr(self.result, 'total_transaction_costs', 0),
            'cost_ratio': getattr(self.result, 'total_transaction_costs', 0) / self.result.initial_capital if self.result.initial_capital > 0 else 0
        }

    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get advanced risk metrics."""
        return self.risk_metrics.calculate_all_metrics()

    def _get_trade_analytics(self) -> Dict[str, Any]:
        """Get comprehensive trade analytics."""
        return self.trade_analytics.get_comprehensive_analysis()

    def _get_rolling_metrics(self) -> Dict[str, Any]:
        """Get rolling performance metrics."""
        rolling_data = {}

        # Rolling Sharpe ratios
        for window in [21, 63, 126]:
            rolling_sharpe = self.rolling_metrics.rolling_sharpe(window)
            if len(rolling_sharpe) > 0:
                rolling_data[f'rolling_sharpe_{window}d'] = {
                    'current': rolling_sharpe.iloc[-1] if not pd.isna(rolling_sharpe.iloc[-1]) else 0,
                    'mean': rolling_sharpe.mean(),
                    'std': rolling_sharpe.std(),
                    'min': rolling_sharpe.min(),
                    'max': rolling_sharpe.max()
                }

        # Rolling volatility
        rolling_vol = self.rolling_metrics.rolling_volatility(63)
        if len(rolling_vol) > 0:
            rolling_data['rolling_volatility'] = {
                'current': rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else 0,
                'mean': rolling_vol.mean(),
                'std': rolling_vol.std()
            }

        # Performance over different periods
        period_performance = self.rolling_metrics.performance_over_periods()
        if not period_performance.empty:
            rolling_data['period_returns'] = period_performance.to_dict('records')

        return rolling_data

    def _get_drawdown_analysis(self) -> Dict[str, Any]:
        """Get detailed drawdown analysis."""
        dd_details = self.risk_metrics.max_drawdown_details()
        all_drawdowns = self.risk_metrics.get_all_drawdowns(min_duration=1)

        drawdown_data = dd_details.copy()

        if not all_drawdowns.empty:
            # Top 5 drawdowns
            top_drawdowns = all_drawdowns.head(5)
            drawdown_data['top_drawdowns'] = top_drawdowns.to_dict('records')

            # Drawdown statistics
            drawdown_data['drawdown_stats'] = {
                'total_drawdowns': len(all_drawdowns),
                'avg_duration': all_drawdowns['duration'].mean(),
                'avg_magnitude': all_drawdowns['max_drawdown'].mean(),
                'recovery_factor': abs(self.result.total_return / self.result.max_drawdown) if self.result.max_drawdown != 0 else float('inf')
            }

        return drawdown_data

    def _get_monthly_returns(self) -> Dict[str, Any]:
        """Get monthly returns analysis."""
        monthly_table = PerformanceAnalyzer.create_monthly_returns_table(self.result)
        monthly_returns = PerformanceAnalyzer.calculate_monthly_returns(self.result)

        data = {}

        if not monthly_table.empty:
            data['monthly_table'] = monthly_table.to_dict()

        if len(monthly_returns) > 0:
            data['monthly_stats'] = {
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'positive_months': (monthly_returns > 0).sum(),
                'negative_months': (monthly_returns < 0).sum(),
                'monthly_win_rate': (monthly_returns > 0).mean(),
                'avg_monthly_return': monthly_returns.mean(),
                'monthly_volatility': monthly_returns.std()
            }

        return data

    def _create_charts(self) -> Dict[str, str]:
        """Create performance charts."""
        charts = {}

        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 10
        })

        try:
            # 1. Equity Curve
            charts['equity_curve'] = self._create_equity_curve()

            # 2. Drawdown Chart
            charts['drawdown'] = self._create_drawdown_chart()

            # 3. Rolling Sharpe
            charts['rolling_sharpe'] = self._create_rolling_sharpe_chart()

            # 4. Monthly Returns Heatmap
            charts['monthly_heatmap'] = self._create_monthly_heatmap()

            # 5. Return Distribution
            charts['return_distribution'] = self._create_return_distribution()

        except Exception as e:
            print(f"Warning: Could not create some charts: {e}")

        return charts

    def _create_equity_curve(self) -> str:
        """Create equity curve chart."""
        if len(self.portfolio_df) == 0:
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot portfolio value
        ax.plot(self.portfolio_df.index, self.portfolio_df['total_value'],
               label='Portfolio', linewidth=2, color='blue')

        # Add benchmark if available
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 0:
            # Align benchmark with portfolio
            aligned_benchmark = self.benchmark_returns.reindex(self.portfolio_df.index, method='nearest')
            benchmark_value = self.result.initial_capital * (1 + aligned_benchmark).cumprod()
            ax.plot(benchmark_value.index, benchmark_value,
                   label='Benchmark', linewidth=2, color='red', alpha=0.7)

        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_drawdown_chart(self) -> str:
        """Create drawdown chart."""
        if len(self.portfolio_df) == 0:
            return ""

        fig, ax = plt.subplots(figsize=(12, 4))

        portfolio_values = self.portfolio_df['total_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max

        ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown, color='red', linewidth=1)

        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_rolling_sharpe_chart(self) -> str:
        """Create rolling Sharpe ratio chart."""
        if len(self.portfolio_df) == 0:
            return ""

        fig, ax = plt.subplots(figsize=(12, 4))

        rolling_sharpe = self.rolling_metrics.rolling_sharpe(63)  # 3-month window
        if len(rolling_sharpe) > 0:
            ax.plot(rolling_sharpe.index, rolling_sharpe,
                   linewidth=2, color='green', label='3-Month Rolling Sharpe')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_title('Rolling Sharpe Ratio (3-Month)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_monthly_heatmap(self) -> str:
        """Create monthly returns heatmap."""
        monthly_table = PerformanceAnalyzer.create_monthly_returns_table(self.result)

        if monthly_table.empty:
            return ""

        fig, ax = plt.subplots(figsize=(12, max(6, len(monthly_table) * 0.5)))

        # Remove the 'Year' column for heatmap
        heatmap_data = monthly_table.drop('Year', axis=1, errors='ignore')

        im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)

        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels([col[:3] for col in heatmap_data.columns])
        ax.set_yticklabels(heatmap_data.index)

        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if not pd.isna(value):
                    text_color = 'white' if abs(value) > 5 else 'black'
                    ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color=text_color, fontsize=8)

        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Monthly Return (%)')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _create_return_distribution(self) -> str:
        """Create return distribution histogram."""
        if len(self.portfolio_df) == 0:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        daily_returns = self.portfolio_df['daily_returns'].dropna()

        # Histogram
        ax1.hist(daily_returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(daily_returns.mean() * 100, color='red', linestyle='--', label='Mean')
        ax1.legend()

        # Q-Q plot to check normality (requires scipy)
        try:
            from scipy import stats
            stats.probplot(daily_returns, dist="norm", plot=ax2)
        except ImportError:
            # Simple scatter plot as fallback
            sorted_returns = np.sort(daily_returns)
            theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_returns))
            ax2.scatter(theoretical_quantiles, sorted_returns, alpha=0.6)
            ax2.set_xlabel('Theoretical Quantiles')
            ax2.set_ylabel('Sample Quantiles')
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    def _generate_html_report(self, tearsheet_data: Dict[str, Any], output_path: str):
        """Generate HTML report."""
        html_template = self._get_html_template()

        # Replace placeholders with actual data
        html_content = html_template.format(
            strategy_name=self.result.metadata.get('strategy', 'Unknown Strategy'),
            symbol=self.result.metadata.get('symbol', 'Unknown Symbol'),
            start_date=self.result.metadata.get('start_date', 'Unknown'),
            end_date=self.result.metadata.get('end_date', 'Unknown'),
            report_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

            # Basic metrics
            initial_capital=tearsheet_data['basic_metrics']['initial_capital'],
            final_value=tearsheet_data['basic_metrics']['final_value'],
            total_return=tearsheet_data['basic_metrics']['total_return'] * 100,
            annualized_return=tearsheet_data['basic_metrics']['annualized_return'] * 100,
            sharpe_ratio=tearsheet_data['basic_metrics']['sharpe_ratio'],
            max_drawdown=tearsheet_data['basic_metrics']['max_drawdown'] * 100,
            volatility=tearsheet_data['basic_metrics']['volatility'] * 100,
            num_trades=tearsheet_data['basic_metrics']['num_trades'],
            win_rate=tearsheet_data['basic_metrics']['win_rate'] * 100,

            # Risk metrics
            sortino_ratio=tearsheet_data['risk_metrics'].get('sortino_ratio', 0),
            calmar_ratio=tearsheet_data['risk_metrics'].get('calmar_ratio', 0),
            var_95=tearsheet_data['risk_metrics'].get('var_95', 0) * 100,
            cvar_95=tearsheet_data['risk_metrics'].get('cvar_95', 0) * 100,
            ulcer_index=tearsheet_data['risk_metrics'].get('ulcer_index', 0),

            # Charts
            equity_curve=tearsheet_data['charts'].get('equity_curve', ''),
            drawdown_chart=tearsheet_data['charts'].get('drawdown', ''),
            rolling_sharpe=tearsheet_data['charts'].get('rolling_sharpe', ''),
            monthly_heatmap=tearsheet_data['charts'].get('monthly_heatmap', ''),
            return_distribution=tearsheet_data['charts'].get('return_distribution', '')
        )

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML report saved to: {output_path}")

    def _get_html_template(self) -> str:
        """Get HTML template for the report."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Backtesting Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; background-color: #f0f0f0; padding: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
        .metric-title {{ font-weight: bold; margin-bottom: 10px; }}
        .chart-section {{ margin: 30px 0; }}
        .chart-title {{ font-size: 18px; font-weight: bold; margin: 20px 0 10px 0; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .metric-value {{ font-size: 20px; color: #333; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Backtesting Performance Report</h1>
        <h2>{strategy_name} - {symbol}</h2>
        <p>Period: {start_date} to {end_date}</p>
        <p>Generated: {report_date}</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-title">Basic Performance</div>
            <p>Initial Capital: <span class="metric-value">${initial_capital:,.2f}</span></p>
            <p>Final Value: <span class="metric-value">${final_value:,.2f}</span></p>
            <p>Total Return: <span class="metric-value">{total_return:.2f}%</span></p>
            <p>Annualized Return: <span class="metric-value">{annualized_return:.2f}%</span></p>
        </div>

        <div class="metric-box">
            <div class="metric-title">Risk Metrics</div>
            <p>Sharpe Ratio: <span class="metric-value">{sharpe_ratio:.2f}</span></p>
            <p>Sortino Ratio: <span class="metric-value">{sortino_ratio:.2f}</span></p>
            <p>Max Drawdown: <span class="metric-value">{max_drawdown:.2f}%</span></p>
            <p>Volatility: <span class="metric-value">{volatility:.2f}%</span></p>
        </div>

        <div class="metric-box">
            <div class="metric-title">Trading Statistics</div>
            <p>Number of Trades: <span class="metric-value">{num_trades}</span></p>
            <p>Win Rate: <span class="metric-value">{win_rate:.2f}%</span></p>
            <p>Value at Risk (95%): <span class="metric-value">{var_95:.2f}%</span></p>
            <p>CVaR (95%): <span class="metric-value">{cvar_95:.2f}%</span></p>
        </div>
    </div>

    <div class="chart-section">
        <div class="chart-title">Equity Curve</div>
        <div class="chart"><img src="{equity_curve}" style="max-width: 100%; height: auto;"></div>

        <div class="chart-title">Drawdown Analysis</div>
        <div class="chart"><img src="{drawdown_chart}" style="max-width: 100%; height: auto;"></div>

        <div class="chart-title">Rolling Sharpe Ratio</div>
        <div class="chart"><img src="{rolling_sharpe}" style="max-width: 100%; height: auto;"></div>

        <div class="chart-title">Monthly Returns Heatmap</div>
        <div class="chart"><img src="{monthly_heatmap}" style="max-width: 100%; height: auto;"></div>

        <div class="chart-title">Return Distribution</div>
        <div class="chart"><img src="{return_distribution}" style="max-width: 100%; height: auto;"></div>
    </div>
</body>
</html>
        """

    def export_to_excel(self, output_path: str):
        """
        Export comprehensive analysis to Excel file.

        Args:
            output_path: Path for Excel file
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Basic metrics
                basic_df = pd.DataFrame([self._get_basic_metrics()]).T
                basic_df.columns = ['Value']
                basic_df.to_excel(writer, sheet_name='Basic Metrics')

                # Risk metrics
                risk_df = pd.DataFrame([self._get_risk_metrics()]).T
                risk_df.columns = ['Value']
                risk_df.to_excel(writer, sheet_name='Risk Metrics')

                # Trades
                if self.result.trades:
                    trades_data = []
                    for trade in self.result.trades:
                        trades_data.append({
                            'Date': trade.timestamp,
                            'Symbol': trade.symbol,
                            'Side': trade.side.value,
                            'Quantity': trade.quantity,
                            'Price': trade.price,
                            'Value': trade.value,
                            'Commission': trade.commission
                        })
                    trades_df = pd.DataFrame(trades_data)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)

                # Portfolio values
                if len(self.portfolio_df) > 0:
                    portfolio_export = self.portfolio_df[['total_value', 'daily_returns']].copy()
                    portfolio_export.to_excel(writer, sheet_name='Portfolio Values')

                # Monthly returns
                monthly_table = PerformanceAnalyzer.create_monthly_returns_table(self.result)
                if not monthly_table.empty:
                    monthly_table.to_excel(writer, sheet_name='Monthly Returns')

            print(f"Excel report saved to: {output_path}")

        except Exception as e:
            print(f"Failed to create Excel report: {e}")

    def print_summary(self):
        """Print a concise summary of the performance report."""
        basic = self._get_basic_metrics()
        risk = self._get_risk_metrics()

        print("\n" + "="*60)
        print("PERFORMANCE REPORT SUMMARY")
        print("="*60)

        print(f"Strategy: {self.result.metadata.get('strategy', 'Unknown')}")
        print(f"Symbol: {self.result.metadata.get('symbol', 'Unknown')}")
        print(f"Period: {self.result.metadata.get('start_date')} to {self.result.metadata.get('end_date')}")

        print(f"\nReturns:")
        print(f"  Total Return:        {basic['total_return']:.2%}")
        print(f"  Annualized Return:   {basic['annualized_return']:.2%}")
        print(f"  Volatility:          {basic['volatility']:.2%}")

        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:        {basic['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:       {risk.get('sortino_ratio', 0):.2f}")
        print(f"  Maximum Drawdown:    {basic['max_drawdown']:.2%}")
        print(f"  VaR (95%):           {risk.get('var_95', 0):.2%}")

        print(f"\nTrading:")
        print(f"  Number of Trades:    {basic['num_trades']}")
        print(f"  Win Rate:            {basic['win_rate']:.2%}")
        print(f"  Transaction Costs:   ${basic['total_trading_costs']:.2f} ({basic['cost_ratio']:.2%})")

        print("="*60)