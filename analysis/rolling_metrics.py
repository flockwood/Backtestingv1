"""Rolling window performance metrics for time-series analysis."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
import warnings
from core.types import BacktestResult


class RollingMetrics:
    """Rolling window calculations for portfolio performance analysis."""

    def __init__(self, result: BacktestResult, trading_days_per_year: int = 252):
        """
        Initialize rolling metrics calculator.

        Args:
            result: BacktestResult object containing backtest data
            trading_days_per_year: Number of trading days per year for annualization
        """
        self.result = result
        self.trading_days = trading_days_per_year

        # Convert portfolio values to DataFrame
        if result.portfolio_values:
            self.portfolio_df = pd.DataFrame(result.portfolio_values)
            self.portfolio_df['timestamp'] = pd.to_datetime(self.portfolio_df['timestamp'])
            self.portfolio_df.set_index('timestamp', inplace=True)

            # Calculate daily returns
            self.portfolio_df['daily_returns'] = self.portfolio_df['total_value'].pct_change()
            self.portfolio_df['cumulative_returns'] = (1 + self.portfolio_df['daily_returns']).cumprod() - 1

            # Clean data
            self.portfolio_df = self.portfolio_df.dropna()
        else:
            self.portfolio_df = pd.DataFrame()

    def rolling_returns(self,
                       windows: Union[int, List[int]] = [21, 63, 126, 252],
                       annualized: bool = True) -> pd.DataFrame:
        """
        Calculate rolling returns for specified windows.

        Args:
            windows: Rolling window sizes in days (or list of windows)
            annualized: Whether to annualize the returns

        Returns:
            DataFrame with rolling returns for each window
        """
        if len(self.portfolio_df) == 0:
            return pd.DataFrame()

        if isinstance(windows, int):
            windows = [windows]

        rolling_returns_df = pd.DataFrame(index=self.portfolio_df.index)

        for window in windows:
            if len(self.portfolio_df) < window:
                warnings.warn(f"Insufficient data for {window}-day rolling window")
                continue

            # Calculate rolling compound returns
            rolling_return = self.portfolio_df['total_value'].pct_change(periods=window)

            if annualized:
                # Annualize the rolling returns
                annualization_factor = self.trading_days / window
                rolling_return = (1 + rolling_return) ** annualization_factor - 1

            rolling_returns_df[f'{window}D_return'] = rolling_return

        return rolling_returns_df

    def rolling_sharpe(self,
                      window: int = 63,
                      risk_free_rate: float = 0.02) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            window: Rolling window size in days
            risk_free_rate: Annual risk-free rate

        Returns:
            Series with rolling Sharpe ratios
        """
        if len(self.portfolio_df) == 0 or len(self.portfolio_df) < window:
            return pd.Series(dtype=float, index=self.portfolio_df.index if len(self.portfolio_df) > 0 else [])

        daily_returns = self.portfolio_df['daily_returns']
        daily_rf_rate = risk_free_rate / self.trading_days

        # Calculate rolling excess returns
        excess_returns = daily_returns - daily_rf_rate

        # Rolling mean and std of excess returns
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = excess_returns.rolling(window=window).std()

        # Annualized rolling Sharpe ratio
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(self.trading_days)

        # Replace infinite values
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan)

        return rolling_sharpe

    def rolling_volatility(self,
                          window: int = 63,
                          annualized: bool = True) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            window: Rolling window size in days
            annualized: Whether to annualize the volatility

        Returns:
            Series with rolling volatility
        """
        if len(self.portfolio_df) == 0 or len(self.portfolio_df) < window:
            return pd.Series(dtype=float, index=self.portfolio_df.index if len(self.portfolio_df) > 0 else [])

        rolling_vol = self.portfolio_df['daily_returns'].rolling(window=window).std()

        if annualized:
            rolling_vol = rolling_vol * np.sqrt(self.trading_days)

        return rolling_vol

    def rolling_max_drawdown(self, window: int = 252) -> pd.Series:
        """
        Calculate rolling maximum drawdown.

        Args:
            window: Rolling window size in days

        Returns:
            Series with rolling maximum drawdowns
        """
        if len(self.portfolio_df) == 0 or len(self.portfolio_df) < window:
            return pd.Series(dtype=float, index=self.portfolio_df.index if len(self.portfolio_df) > 0 else [])

        portfolio_values = self.portfolio_df['total_value']

        def calc_max_drawdown(window_values):
            """Calculate max drawdown for a window of values."""
            peak = window_values.expanding().max()
            drawdown = (window_values - peak) / peak
            return drawdown.min()

        rolling_dd = portfolio_values.rolling(window=window).apply(calc_max_drawdown, raw=False)

        return rolling_dd

    def rolling_beta(self,
                    benchmark_returns: pd.Series,
                    window: int = 63) -> pd.Series:
        """
        Calculate rolling beta vs benchmark.

        Args:
            benchmark_returns: Benchmark return series (same frequency as portfolio)
            window: Rolling window size

        Returns:
            Series with rolling beta values
        """
        if len(self.portfolio_df) == 0 or len(self.portfolio_df) < window:
            return pd.Series(dtype=float, index=self.portfolio_df.index if len(self.portfolio_df) > 0 else [])

        portfolio_returns = self.portfolio_df['daily_returns']

        # Align the series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
        aligned_data.columns = ['portfolio', 'benchmark']

        if len(aligned_data) < window:
            warnings.warn("Insufficient aligned data for beta calculation")
            return pd.Series(dtype=float, index=portfolio_returns.index)

        def calc_beta(data):
            """Calculate beta for a window of data."""
            if len(data) < 2:
                return np.nan
            covariance = data.cov().iloc[0, 1]
            benchmark_variance = data['benchmark'].var()
            if benchmark_variance == 0:
                return np.nan
            return covariance / benchmark_variance

        rolling_beta = aligned_data.rolling(window=window).apply(calc_beta, raw=False)

        # Extract just the beta values (first column of result)
        if isinstance(rolling_beta, pd.DataFrame):
            rolling_beta = rolling_beta.iloc[:, 0]

        return rolling_beta.reindex(portfolio_returns.index)

    def stability_of_returns(self, window: int = 252) -> pd.Series:
        """
        Calculate stability of returns (R-squared of equity curve regression).

        Args:
            window: Rolling window size

        Returns:
            Series with R-squared values
        """
        if len(self.portfolio_df) == 0 or len(self.portfolio_df) < window:
            return pd.Series(dtype=float, index=self.portfolio_df.index if len(self.portfolio_df) > 0 else [])

        log_equity = np.log(self.portfolio_df['total_value'])

        def calc_r_squared(log_values):
            """Calculate R-squared of log equity curve."""
            if len(log_values) < 3:
                return np.nan

            x = np.arange(len(log_values))
            try:
                # Linear regression of log equity vs time
                correlation = np.corrcoef(x, log_values)[0, 1]
                r_squared = correlation ** 2
                return r_squared
            except:
                return np.nan

        rolling_r2 = log_equity.rolling(window=window).apply(calc_r_squared, raw=False)

        return rolling_r2

    def rolling_sortino(self,
                       window: int = 63,
                       target_return: float = 0.0,
                       risk_free_rate: float = 0.02) -> pd.Series:
        """
        Calculate rolling Sortino ratio.

        Args:
            window: Rolling window size
            target_return: Target return for downside calculation
            risk_free_rate: Risk-free rate

        Returns:
            Series with rolling Sortino ratios
        """
        if len(self.portfolio_df) == 0 or len(self.portfolio_df) < window:
            return pd.Series(dtype=float, index=self.portfolio_df.index if len(self.portfolio_df) > 0 else [])

        daily_returns = self.portfolio_df['daily_returns']
        daily_rf_rate = risk_free_rate / self.trading_days
        excess_returns = daily_returns - daily_rf_rate

        def calc_sortino(returns_window):
            """Calculate Sortino ratio for a window."""
            if len(returns_window) < 2:
                return np.nan

            mean_return = returns_window.mean()
            downside_returns = returns_window[returns_window < target_return]

            if len(downside_returns) == 0:
                return np.inf if mean_return > 0 else np.nan

            downside_std = np.sqrt(np.mean(downside_returns**2))
            if downside_std == 0:
                return np.inf if mean_return > 0 else np.nan

            sortino = (mean_return * self.trading_days) / (downside_std * np.sqrt(self.trading_days))
            return sortino

        rolling_sortino = excess_returns.rolling(window=window).apply(calc_sortino, raw=False)

        return rolling_sortino

    def expanding_metrics(self,
                         risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        Calculate expanding window metrics (cumulative from start).

        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation

        Returns:
            DataFrame with expanding metrics
        """
        if len(self.portfolio_df) == 0:
            return pd.DataFrame()

        daily_returns = self.portfolio_df['daily_returns']
        daily_rf_rate = risk_free_rate / self.trading_days
        excess_returns = daily_returns - daily_rf_rate

        metrics_df = pd.DataFrame(index=self.portfolio_df.index)

        # Expanding returns (cumulative)
        metrics_df['expanding_return'] = self.portfolio_df['cumulative_returns']

        # Expanding Sharpe ratio
        expanding_mean = excess_returns.expanding().mean()
        expanding_std = excess_returns.expanding().std()
        metrics_df['expanding_sharpe'] = (expanding_mean / expanding_std) * np.sqrt(self.trading_days)

        # Expanding volatility
        metrics_df['expanding_volatility'] = daily_returns.expanding().std() * np.sqrt(self.trading_days)

        # Expanding max drawdown
        portfolio_values = self.portfolio_df['total_value']
        expanding_max = portfolio_values.expanding().max()
        expanding_dd = (portfolio_values - expanding_max) / expanding_max
        metrics_df['expanding_max_dd'] = expanding_dd.expanding().min()

        # Clean infinite values
        metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)

        return metrics_df

    def get_rolling_summary(self,
                           windows: List[int] = [21, 63, 126, 252],
                           risk_free_rate: float = 0.02) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive rolling metrics summary.

        Args:
            windows: List of rolling windows to calculate
            risk_free_rate: Risk-free rate for calculations

        Returns:
            Dictionary of DataFrames with rolling metrics
        """
        summary = {}

        try:
            # Rolling returns
            summary['returns'] = self.rolling_returns(windows)

            # For each window, calculate Sharpe and volatility
            for window in windows:
                if len(self.portfolio_df) >= window:
                    key = f'{window}D'
                    summary[f'sharpe_{key}'] = self.rolling_sharpe(window, risk_free_rate)
                    summary[f'volatility_{key}'] = self.rolling_volatility(window)
                    summary[f'max_dd_{key}'] = self.rolling_max_drawdown(window)

            # Expanding metrics
            summary['expanding'] = self.expanding_metrics(risk_free_rate)

        except Exception as e:
            warnings.warn(f"Error calculating rolling metrics: {e}")

        return summary

    def performance_over_periods(self) -> pd.DataFrame:
        """
        Calculate performance over different time periods.

        Returns:
            DataFrame with period performance
        """
        if len(self.portfolio_df) == 0:
            return pd.DataFrame()

        portfolio_values = self.portfolio_df['total_value']
        end_value = portfolio_values.iloc[-1]
        start_value = portfolio_values.iloc[0]

        periods_data = []

        # Different lookback periods
        lookbacks = [
            ('1W', 7), ('1M', 21), ('3M', 63), ('6M', 126),
            ('1Y', 252), ('2Y', 504), ('3Y', 756)
        ]

        for period_name, days in lookbacks:
            if len(portfolio_values) > days:
                period_start_value = portfolio_values.iloc[-days-1]
                period_return = (end_value / period_start_value) - 1

                # Annualize if less than a year
                if days < self.trading_days:
                    period_return_annualized = (1 + period_return) ** (self.trading_days / days) - 1
                else:
                    period_return_annualized = period_return

                periods_data.append({
                    'period': period_name,
                    'days': days,
                    'total_return': period_return,
                    'annualized_return': period_return_annualized,
                    'start_value': period_start_value,
                    'end_value': end_value
                })

        return pd.DataFrame(periods_data)