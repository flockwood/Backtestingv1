"""Advanced risk metrics for portfolio performance analysis."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None
import warnings
from core.types import BacktestResult


class RiskMetrics:
    """Advanced risk metrics calculations for backtesting results."""

    def __init__(self, result: BacktestResult, trading_days_per_year: int = 252):
        """
        Initialize risk metrics calculator.

        Args:
            result: BacktestResult object containing backtest data
            trading_days_per_year: Number of trading days per year for annualization
        """
        self.result = result
        self.trading_days = trading_days_per_year

        # Convert portfolio values to DataFrame for easier manipulation
        if result.portfolio_values:
            self.portfolio_df = pd.DataFrame(result.portfolio_values)
            self.portfolio_df['timestamp'] = pd.to_datetime(self.portfolio_df['timestamp'])
            self.portfolio_df.set_index('timestamp', inplace=True)

            # Calculate daily returns
            self.portfolio_df['daily_returns'] = self.portfolio_df['total_value'].pct_change()
            self.daily_returns = self.portfolio_df['daily_returns'].dropna()
        else:
            self.portfolio_df = pd.DataFrame()
            self.daily_returns = pd.Series(dtype=float)

    def sortino_ratio(self,
                     target_return: float = 0.0,
                     risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio focusing on downside deviation.

        Args:
            target_return: Target return (default: 0%)
            risk_free_rate: Risk-free rate for annualization

        Returns:
            Sortino ratio
        """
        if len(self.daily_returns) == 0:
            return 0.0

        # Calculate excess returns
        daily_risk_free = risk_free_rate / self.trading_days
        excess_returns = self.daily_returns - daily_risk_free

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < target_return]

        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        downside_deviation = np.sqrt(np.mean(downside_returns**2))

        if downside_deviation == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0

        # Annualized Sortino ratio
        annualized_excess_return = excess_returns.mean() * self.trading_days
        annualized_downside_dev = downside_deviation * np.sqrt(self.trading_days)

        return annualized_excess_return / annualized_downside_dev

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annualized return / maximum drawdown).

        Returns:
            Calmar ratio
        """
        max_dd = abs(self.result.max_drawdown)
        if max_dd == 0:
            return float('inf') if self.result.annualized_return > 0 else 0.0

        return self.result.annualized_return / max_dd

    def max_drawdown_details(self) -> dict:
        """
        Calculate detailed maximum drawdown information.

        Returns:
            Dictionary with drawdown details
        """
        if len(self.portfolio_df) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'recovery_time': 0,
                'start_date': None,
                'end_date': None,
                'recovery_date': None
            }

        portfolio_values = self.portfolio_df['total_value']

        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max

        # Find maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_drawdown = drawdown.min()

        # Find start of drawdown (last peak before max drawdown)
        peak_before_dd = running_max[:max_dd_idx]
        dd_start_idx = peak_before_dd[peak_before_dd == peak_before_dd.iloc[-1]].index[-1]

        # Find recovery date (when portfolio reaches new high after drawdown)
        values_after_dd = portfolio_values[max_dd_idx:]
        peak_value = portfolio_values.loc[dd_start_idx]
        recovery_mask = values_after_dd >= peak_value

        recovery_date = None
        if recovery_mask.any():
            recovery_date = recovery_mask[recovery_mask].index[0]

        # Calculate durations
        dd_duration = (max_dd_idx - dd_start_idx).days if hasattr(max_dd_idx - dd_start_idx, 'days') else len(portfolio_values[dd_start_idx:max_dd_idx])
        recovery_time = (recovery_date - max_dd_idx).days if recovery_date and hasattr(recovery_date - max_dd_idx, 'days') else None

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': dd_duration,
            'recovery_time': recovery_time,
            'start_date': dd_start_idx,
            'end_date': max_dd_idx,
            'recovery_date': recovery_date
        }

    def value_at_risk(self,
                     confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical' or 'parametric'

        Returns:
            VaR as a positive number (loss amount)
        """
        if len(self.daily_returns) == 0:
            return 0.0

        if method == 'historical':
            return -np.percentile(self.daily_returns, (1 - confidence) * 100)

        elif method == 'parametric':
            if not HAS_SCIPY:
                # Fallback to historical method if scipy not available
                return self.value_at_risk(confidence, 'historical')

            # Assume normal distribution
            mean_return = self.daily_returns.mean()
            std_return = self.daily_returns.std()
            var_cutoff = stats.norm.ppf(1 - confidence, mean_return, std_return)
            return -var_cutoff

        else:
            raise ValueError("Method must be 'historical' or 'parametric'")

    def conditional_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) - expected loss beyond VaR.

        Args:
            confidence: Confidence level

        Returns:
            CVaR as a positive number
        """
        if len(self.daily_returns) == 0:
            return 0.0

        var_threshold = -self.value_at_risk(confidence, method='historical')
        tail_losses = self.daily_returns[self.daily_returns <= var_threshold]

        if len(tail_losses) == 0:
            return self.value_at_risk(confidence, method='historical')

        return -tail_losses.mean()

    def downside_deviation(self, target_return: float = 0.0) -> float:
        """
        Calculate downside deviation.

        Args:
            target_return: Target return threshold

        Returns:
            Annualized downside deviation
        """
        if len(self.daily_returns) == 0:
            return 0.0

        downside_returns = self.daily_returns[self.daily_returns < target_return]

        if len(downside_returns) == 0:
            return 0.0

        downside_variance = np.mean((downside_returns - target_return)**2)
        daily_downside_dev = np.sqrt(downside_variance)

        return daily_downside_dev * np.sqrt(self.trading_days)

    def omega_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Omega ratio (probability weighted ratio of gains vs losses).

        Args:
            target_return: Target return threshold

        Returns:
            Omega ratio
        """
        if len(self.daily_returns) == 0:
            return 0.0

        gains = self.daily_returns[self.daily_returns > target_return] - target_return
        losses = target_return - self.daily_returns[self.daily_returns < target_return]

        if len(losses) == 0:
            return float('inf') if len(gains) > 0 else 1.0

        if len(gains) == 0:
            return 0.0

        return gains.sum() / losses.sum()

    def ulcer_index(self) -> float:
        """
        Calculate Ulcer Index - measure of drawdown risk.

        Returns:
            Ulcer Index
        """
        if len(self.portfolio_df) == 0:
            return 0.0

        portfolio_values = self.portfolio_df['total_value']
        running_max = portfolio_values.expanding().max()
        drawdown_pct = ((portfolio_values - running_max) / running_max) * 100

        # Calculate squared drawdowns
        squared_drawdowns = drawdown_pct**2
        ulcer_index = np.sqrt(squared_drawdowns.mean())

        return ulcer_index

    def tail_ratio(self, percentile: float = 0.95) -> float:
        """
        Calculate tail ratio (average of top percentile / average of bottom percentile).

        Args:
            percentile: Percentile threshold

        Returns:
            Tail ratio
        """
        if len(self.daily_returns) == 0:
            return 0.0

        top_threshold = np.percentile(self.daily_returns, percentile * 100)
        bottom_threshold = np.percentile(self.daily_returns, (1 - percentile) * 100)

        top_returns = self.daily_returns[self.daily_returns >= top_threshold]
        bottom_returns = self.daily_returns[self.daily_returns <= bottom_threshold]

        if len(bottom_returns) == 0 or bottom_returns.mean() == 0:
            return float('inf') if len(top_returns) > 0 and top_returns.mean() > 0 else 0.0

        return abs(top_returns.mean() / bottom_returns.mean())

    def get_all_drawdowns(self, min_duration: int = 1) -> pd.DataFrame:
        """
        Get all drawdown periods with details.

        Args:
            min_duration: Minimum duration in days to include drawdown

        Returns:
            DataFrame with drawdown details
        """
        if len(self.portfolio_df) == 0:
            return pd.DataFrame()

        portfolio_values = self.portfolio_df['total_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_starts = in_drawdown & (~in_drawdown.shift(1, fill_value=False))
        drawdown_ends = (~in_drawdown) & in_drawdown.shift(1, fill_value=False)

        drawdowns = []
        start_indices = drawdown_starts[drawdown_starts].index
        end_indices = drawdown_ends[drawdown_ends].index

        # Handle case where we end in a drawdown
        if len(start_indices) > len(end_indices):
            end_indices = end_indices.append(pd.Index([portfolio_values.index[-1]]))

        for start_idx, end_idx in zip(start_indices, end_indices):
            dd_period = drawdown[start_idx:end_idx]
            duration = len(dd_period)

            if duration >= min_duration:
                max_dd = dd_period.min()
                max_dd_date = dd_period.idxmin()

                drawdowns.append({
                    'start_date': start_idx,
                    'end_date': end_idx,
                    'duration': duration,
                    'max_drawdown': max_dd,
                    'max_drawdown_date': max_dd_date
                })

        return pd.DataFrame(drawdowns).sort_values('max_drawdown')

    def calculate_all_metrics(self, risk_free_rate: float = 0.02) -> dict:
        """
        Calculate all risk metrics at once.

        Args:
            risk_free_rate: Risk-free rate for calculations

        Returns:
            Dictionary with all risk metrics
        """
        metrics = {}

        try:
            metrics['sortino_ratio'] = self.sortino_ratio(risk_free_rate=risk_free_rate)
            metrics['calmar_ratio'] = self.calmar_ratio()
            metrics['var_95'] = self.value_at_risk(0.95, 'historical')
            metrics['var_99'] = self.value_at_risk(0.99, 'historical')
            metrics['cvar_95'] = self.conditional_var(0.95)
            metrics['cvar_99'] = self.conditional_var(0.99)
            metrics['downside_deviation'] = self.downside_deviation()
            metrics['omega_ratio'] = self.omega_ratio()
            metrics['ulcer_index'] = self.ulcer_index()
            metrics['tail_ratio'] = self.tail_ratio()

            # Max drawdown details
            dd_details = self.max_drawdown_details()
            metrics.update(dd_details)

        except Exception as e:
            warnings.warn(f"Error calculating some risk metrics: {e}")
            # Fill with default values for failed calculations
            for key in ['sortino_ratio', 'calmar_ratio', 'var_95', 'var_99',
                       'cvar_95', 'cvar_99', 'downside_deviation', 'omega_ratio',
                       'ulcer_index', 'tail_ratio']:
                if key not in metrics:
                    metrics[key] = 0.0

        return metrics