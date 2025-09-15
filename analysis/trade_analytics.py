"""Detailed trade analysis and statistics for backtesting results."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
from core.types import BacktestResult, Trade


class TradeAnalytics:
    """Comprehensive trade analysis for backtesting results."""

    def __init__(self, result: BacktestResult):
        """
        Initialize trade analytics.

        Args:
            result: BacktestResult object containing trade data
        """
        self.result = result
        self.trades_df = self._prepare_trades_data()

    def _prepare_trades_data(self) -> pd.DataFrame:
        """Prepare trades data for analysis."""
        if not self.result.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.result.trades:
            trade_dict = {
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'slippage': getattr(trade, 'slippage', 0),
                'spread_cost': getattr(trade, 'spread_cost', 0),
                'total_cost': getattr(trade, 'total_cost', trade.commission),
                'trade_id': trade.trade_id
            }
            trades_data.append(trade_dict)

        df = pd.DataFrame(trades_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def get_trade_pairs(self) -> pd.DataFrame:
        """
        Match buy and sell trades to create trade pairs for P&L analysis.

        Returns:
            DataFrame with matched trade pairs
        """
        if len(self.trades_df) == 0:
            return pd.DataFrame()

        buy_trades = self.trades_df[self.trades_df['side'] == 'BUY'].copy()
        sell_trades = self.trades_df[self.trades_df['side'] == 'SELL'].copy()

        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return pd.DataFrame()

        pairs = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades.iloc[i]
            sell_trade = sell_trades.iloc[i]

            # Calculate P&L
            gross_pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['quantity']
            total_costs = buy_trade['total_cost'] + sell_trade['total_cost']
            net_pnl = gross_pnl - total_costs

            # Calculate holding period
            holding_period = (sell_trade['timestamp'] - buy_trade['timestamp']).total_seconds() / 3600  # hours

            # Calculate returns
            total_investment = buy_trade['value'] + buy_trade['total_cost']
            return_pct = net_pnl / total_investment if total_investment > 0 else 0

            pairs.append({
                'pair_id': i + 1,
                'entry_date': buy_trade['timestamp'],
                'exit_date': sell_trade['timestamp'],
                'entry_price': buy_trade['price'],
                'exit_price': sell_trade['price'],
                'quantity': buy_trade['quantity'],
                'holding_period_hours': holding_period,
                'holding_period_days': holding_period / 24,
                'gross_pnl': gross_pnl,
                'total_costs': total_costs,
                'net_pnl': net_pnl,
                'return_pct': return_pct,
                'is_winner': net_pnl > 0,
                'entry_slippage': buy_trade['slippage'],
                'exit_slippage': sell_trade['slippage'],
                'total_slippage': buy_trade['slippage'] + sell_trade['slippage']
            })

        return pd.DataFrame(pairs)

    def get_win_loss_streaks(self) -> Dict[str, Any]:
        """
        Analyze winning and losing streaks.

        Returns:
            Dictionary with streak statistics
        """
        pairs_df = self.get_trade_pairs()

        if len(pairs_df) == 0:
            return {
                'max_winning_streak': 0,
                'max_losing_streak': 0,
                'current_streak': 0,
                'current_streak_type': None,
                'avg_winning_streak': 0,
                'avg_losing_streak': 0,
                'streak_details': []
            }

        # Create streak analysis
        is_winner = pairs_df['is_winner'].values
        streaks = []
        current_streak = 1
        current_type = is_winner[0]

        for i in range(1, len(is_winner)):
            if is_winner[i] == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_streak = 1
                current_type = is_winner[i]

        # Add the last streak
        streaks.append((current_type, current_streak))

        # Analyze streaks
        winning_streaks = [length for is_win, length in streaks if is_win]
        losing_streaks = [length for is_win, length in streaks if not is_win]

        return {
            'max_winning_streak': max(winning_streaks) if winning_streaks else 0,
            'max_losing_streak': max(losing_streaks) if losing_streaks else 0,
            'current_streak': current_streak,
            'current_streak_type': 'win' if current_type else 'loss',
            'avg_winning_streak': np.mean(winning_streaks) if winning_streaks else 0,
            'avg_losing_streak': np.mean(losing_streaks) if losing_streaks else 0,
            'num_winning_streaks': len(winning_streaks),
            'num_losing_streaks': len(losing_streaks),
            'streak_details': streaks
        }

    def get_holding_periods(self) -> Dict[str, float]:
        """
        Analyze holding periods by outcome.

        Returns:
            Dictionary with holding period statistics
        """
        pairs_df = self.get_trade_pairs()

        if len(pairs_df) == 0:
            return {
                'avg_holding_period_hours': 0,
                'avg_holding_period_days': 0,
                'avg_winning_holding_period': 0,
                'avg_losing_holding_period': 0,
                'median_holding_period': 0,
                'max_holding_period': 0,
                'min_holding_period': 0
            }

        winning_trades = pairs_df[pairs_df['is_winner']]
        losing_trades = pairs_df[~pairs_df['is_winner']]

        return {
            'avg_holding_period_hours': pairs_df['holding_period_hours'].mean(),
            'avg_holding_period_days': pairs_df['holding_period_days'].mean(),
            'avg_winning_holding_period': winning_trades['holding_period_hours'].mean() if len(winning_trades) > 0 else 0,
            'avg_losing_holding_period': losing_trades['holding_period_hours'].mean() if len(losing_trades) > 0 else 0,
            'median_holding_period_hours': pairs_df['holding_period_hours'].median(),
            'max_holding_period_hours': pairs_df['holding_period_hours'].max(),
            'min_holding_period_hours': pairs_df['holding_period_hours'].min()
        }

    def time_based_performance(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze performance by time periods (day of week, month, hour).

        Returns:
            Dictionary with performance by different time periods
        """
        if len(self.trades_df) == 0:
            return {}

        pairs_df = self.get_trade_pairs()
        if len(pairs_df) == 0:
            return {}

        analysis = {}

        # Day of week analysis (entry day)
        pairs_df['entry_day_of_week'] = pairs_df['entry_date'].dt.day_name()
        day_of_week = pairs_df.groupby('entry_day_of_week').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'return_pct': 'mean',
            'is_winner': 'mean'
        }).round(4)
        day_of_week.columns = ['count', 'total_pnl', 'avg_pnl', 'avg_return', 'win_rate']
        analysis['day_of_week'] = day_of_week

        # Month analysis
        pairs_df['entry_month'] = pairs_df['entry_date'].dt.strftime('%B')
        month_analysis = pairs_df.groupby('entry_month').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'return_pct': 'mean',
            'is_winner': 'mean'
        }).round(4)
        month_analysis.columns = ['count', 'total_pnl', 'avg_pnl', 'avg_return', 'win_rate']
        analysis['month'] = month_analysis

        # Hour analysis (if intraday data)
        pairs_df['entry_hour'] = pairs_df['entry_date'].dt.hour
        if pairs_df['entry_hour'].nunique() > 1:  # Only if we have intraday data
            hour_analysis = pairs_df.groupby('entry_hour').agg({
                'net_pnl': ['count', 'sum', 'mean'],
                'return_pct': 'mean',
                'is_winner': 'mean'
            }).round(4)
            hour_analysis.columns = ['count', 'total_pnl', 'avg_pnl', 'avg_return', 'win_rate']
            analysis['hour'] = hour_analysis

        return analysis

    def calculate_mae_mfe(self) -> pd.DataFrame:
        """
        Calculate Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
        Note: This requires tick-by-tick or at least high-frequency data during trades.

        Returns:
            DataFrame with MAE/MFE analysis (placeholder implementation)
        """
        pairs_df = self.get_trade_pairs()

        if len(pairs_df) == 0:
            return pd.DataFrame()

        # Placeholder implementation - in a real system, you'd need price data
        # during the holding period to calculate true MAE/MFE
        mae_mfe_data = []

        for _, trade in pairs_df.iterrows():
            # Simplified calculation based on entry/exit prices
            # In reality, you'd track the price movement during the holding period
            price_change = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']

            # Simplified MAE/MFE (assume worst case scenarios)
            if trade['is_winner']:
                # For winning trades, assume some adverse movement
                mae = abs(price_change * 0.3)  # Assume 30% of move was adverse
                mfe = abs(price_change)
            else:
                # For losing trades
                mae = abs(price_change)
                mfe = abs(price_change * 0.2)  # Assume some favorable movement

            mae_mfe_data.append({
                'pair_id': trade['pair_id'],
                'net_pnl': trade['net_pnl'],
                'mae_pct': -mae,  # Negative because it's adverse
                'mfe_pct': mfe,
                'mae_dollar': -mae * trade['entry_price'] * trade['quantity'],
                'mfe_dollar': mfe * trade['entry_price'] * trade['quantity'],
                'is_winner': trade['is_winner']
            })

        return pd.DataFrame(mae_mfe_data)

    def slippage_analysis(self) -> Dict[str, float]:
        """
        Analyze slippage costs and their impact.

        Returns:
            Dictionary with slippage statistics
        """
        if len(self.trades_df) == 0:
            return {}

        total_slippage = self.trades_df['slippage'].sum()
        avg_slippage_per_trade = self.trades_df['slippage'].mean()
        max_slippage = self.trades_df['slippage'].max()

        # Slippage as percentage of trade value
        self.trades_df['slippage_pct'] = self.trades_df['slippage'] / self.trades_df['value']
        avg_slippage_pct = self.trades_df['slippage_pct'].mean() * 100

        # Slippage by trade size
        small_trades = self.trades_df[self.trades_df['value'] < self.trades_df['value'].median()]
        large_trades = self.trades_df[self.trades_df['value'] >= self.trades_df['value'].median()]

        return {
            'total_slippage': total_slippage,
            'avg_slippage_per_trade': avg_slippage_per_trade,
            'max_slippage': max_slippage,
            'avg_slippage_pct': avg_slippage_pct,
            'small_trade_avg_slippage': small_trades['slippage'].mean() if len(small_trades) > 0 else 0,
            'large_trade_avg_slippage': large_trades['slippage'].mean() if len(large_trades) > 0 else 0,
            'slippage_impact_on_returns': total_slippage / self.result.initial_capital * 100
        }

    def risk_reward_analysis(self) -> Dict[str, Any]:
        """
        Analyze risk/reward ratios and expectancy.

        Returns:
            Dictionary with risk/reward statistics
        """
        pairs_df = self.get_trade_pairs()

        if len(pairs_df) == 0:
            return {}

        winning_trades = pairs_df[pairs_df['is_winner']]
        losing_trades = pairs_df[~pairs_df['is_winner']]

        # Basic statistics
        avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['net_pnl'].mean()) if len(losing_trades) > 0 else 0
        win_rate = len(winning_trades) / len(pairs_df) if len(pairs_df) > 0 else 0

        # Profit factor
        gross_profit = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Risk/reward ratio
        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Kelly criterion
        kelly_pct = win_rate - ((1 - win_rate) / risk_reward_ratio) if risk_reward_ratio > 0 else 0

        return {
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_rate': win_rate,
            'loss_rate': 1 - win_rate,
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward_ratio,
            'expectancy': expectancy,
            'kelly_percentage': max(0, kelly_pct),  # Don't allow negative Kelly
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'largest_win': winning_trades['net_pnl'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['net_pnl'].min() if len(losing_trades) > 0 else 0
        }

    def trade_clustering_analysis(self) -> Dict[str, Any]:
        """
        Analyze clustering of trades over time.

        Returns:
            Dictionary with clustering statistics
        """
        if len(self.trades_df) == 0:
            return {}

        # Calculate time between trades
        trade_times = self.trades_df['timestamp'].sort_values()
        time_diffs = trade_times.diff().dt.total_seconds() / 3600  # hours

        # Identify trade clusters (trades within 1 hour of each other)
        cluster_threshold = 1.0  # 1 hour
        clusters = []
        current_cluster = [0]

        for i in range(1, len(time_diffs)):
            if time_diffs.iloc[i] <= cluster_threshold:
                current_cluster.append(i)
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [i]

        if len(current_cluster) > 1:
            clusters.append(current_cluster)

        # Analyze clusters
        cluster_sizes = [len(cluster) for cluster in clusters]

        return {
            'num_trade_clusters': len(clusters),
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'clustered_trades_pct': sum(cluster_sizes) / len(self.trades_df) * 100 if len(self.trades_df) > 0 else 0,
            'avg_time_between_trades_hours': time_diffs.mean() if len(time_diffs) > 1 else 0,
            'median_time_between_trades_hours': time_diffs.median() if len(time_diffs) > 1 else 0
        }

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get all trade analytics in one comprehensive report.

        Returns:
            Dictionary with all trade analysis results
        """
        analysis = {}

        try:
            analysis['trade_pairs'] = self.get_trade_pairs()
            analysis['win_loss_streaks'] = self.get_win_loss_streaks()
            analysis['holding_periods'] = self.get_holding_periods()
            analysis['time_based_performance'] = self.time_based_performance()
            analysis['mae_mfe'] = self.calculate_mae_mfe()
            analysis['slippage_analysis'] = self.slippage_analysis()
            analysis['risk_reward'] = self.risk_reward_analysis()
            analysis['clustering'] = self.trade_clustering_analysis()

            # Summary statistics
            analysis['summary'] = {
                'total_trades': len(self.trades_df),
                'total_trade_pairs': len(analysis['trade_pairs']),
                'total_volume': self.trades_df['value'].sum() if len(self.trades_df) > 0 else 0,
                'avg_trade_size': self.trades_df['value'].mean() if len(self.trades_df) > 0 else 0,
                'total_costs': self.trades_df['total_cost'].sum() if len(self.trades_df) > 0 else 0,
                'cost_per_trade': self.trades_df['total_cost'].mean() if len(self.trades_df) > 0 else 0
            }

        except Exception as e:
            warnings.warn(f"Error in comprehensive trade analysis: {e}")

        return analysis