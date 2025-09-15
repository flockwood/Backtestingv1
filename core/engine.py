from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import time
import logging

from .portfolio import Portfolio
from .types import Order, OrderSide, BacktestResult
from .costs import CostModel, ZeroCostModel
from strategies.base import BaseStrategy
from data.base import BaseDataLoader

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine that orchestrates the entire simulation.
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 position_size: float = 1.0,
                 cost_model: Optional[CostModel] = None,
                 save_to_db: bool = False,
                 db_manager=None,
                 repository=None):
        """
        Initialize the backtesting engine.

        Args:
            initial_capital: Starting capital
            commission: Commission rate (as decimal, e.g., 0.001 = 0.1%)
            position_size: Fraction of capital to risk per trade (0.0 to 1.0)
            cost_model: Cost model for transaction costs (if None, uses ZeroCostModel)
            save_to_db: Whether to save results to database
            db_manager: Database manager instance
            repository: Database repository instance
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        self.cost_model = cost_model if cost_model else ZeroCostModel()
        self.portfolio = Portfolio(initial_capital, cost_model=self.cost_model, commission=commission)
        self.save_to_db = save_to_db
        self.db_manager = db_manager
        self.repository = repository
        
    def run_backtest(self, 
                    data_loader: BaseDataLoader,
                    strategy: BaseStrategy,
                    symbol: str,
                    start_date: str,
                    end_date: str) -> BacktestResult:
        """
        Run a complete backtest.
        
        Args:
            data_loader: Data loader instance
            strategy: Trading strategy instance
            symbol: Stock symbol to trade
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult with complete results
        """
        start_time = time.time()
        print(f"Running backtest for {symbol} from {start_date} to {end_date}")
        print(f"Strategy: {strategy}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")

        # Load data
        data = data_loader.load_data(symbol, start_date, end_date)
        if data.empty:
            raise ValueError("No data loaded for backtest")

        # Generate signals
        print(f"Generating signals...")
        signals_data = strategy.generate_signals(data)

        # Run simulation
        print(f"Running simulation...")
        self._simulate_trades(signals_data, symbol)

        # Calculate results
        run_duration = time.time() - start_time
        result = self._calculate_results(strategy, symbol, start_date, end_date)

        # Save to database if enabled
        if self.save_to_db and self.repository:
            try:
                self._save_to_database(strategy, symbol, start_date, end_date,
                                     data_loader.__class__.__name__, run_duration, result)
                print("Results saved to database")
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")
                print(f"Warning: Failed to save to database: {e}")

        print(f"Backtest completed!")
        return result
    
    def _simulate_trades(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Simulate trading based on signals.

        Args:
            data: DataFrame with price data and signals
            symbol: Stock symbol being traded
        """
        current_position = 0  # 0 = no position, 1 = long position

        for timestamp, row in data.iterrows():
            current_price = row['Close']
            position_change = row.get('Position', 0)
            volume = row.get('Volume', None)  # Get volume data for slippage calculation

            # Record portfolio value
            self.portfolio.record_portfolio_value(
                timestamp, {symbol: current_price}
            )

            # Handle position changes
            if position_change == 1 and current_position == 0:
                # Enter long position
                self._enter_long_position(symbol, current_price, timestamp, volume)
                current_position = 1

            elif position_change == -1 and current_position == 1:
                # Exit long position
                self._exit_long_position(symbol, current_price, timestamp, volume)
                current_position = 0
    
    def _enter_long_position(self, symbol: str, price: float, timestamp: datetime, volume: Optional[float] = None) -> None:
        """Enter a long position."""
        # Calculate position size based on available cash
        # First estimate costs to adjust position size
        test_quantity = self.portfolio.cash * self.position_size / price
        costs = self.cost_model.calculate_costs(
            side=OrderSide.BUY,
            quantity=test_quantity,
            price=price,
            volume=volume
        )

        # Adjust for total costs
        available_cash = self.portfolio.cash * self.position_size
        quantity = available_cash / (price + costs.total_cost / test_quantity)

        if quantity > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                timestamp=timestamp
            )

            trade = self.portfolio.execute_order(order, price, timestamp, volume)
            if trade:
                print(f"  BUY: {trade.quantity:.2f} shares at ${trade.price:.2f} (costs: ${trade.total_cost:.2f})")
    
    def _exit_long_position(self, symbol: str, price: float, timestamp: datetime, volume: Optional[float] = None) -> None:
        """Exit a long position."""
        position = self.portfolio.positions.get(symbol)
        if position and position.quantity > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                timestamp=timestamp
            )

            trade = self.portfolio.execute_order(order, price, timestamp, volume)
            if trade:
                print(f"  SELL: {trade.quantity:.2f} shares at ${trade.price:.2f} (costs: ${trade.total_cost:.2f})")
    
    def _calculate_results(self, 
                          strategy: BaseStrategy,
                          symbol: str,
                          start_date: str,
                          end_date: str) -> BacktestResult:
        """Calculate backtest performance metrics."""
        
        # Basic metrics
        final_value = self.portfolio.total_value
        total_return = self.portfolio.total_return / 100
        
        # Calculate additional metrics
        portfolio_df = pd.DataFrame(self.portfolio.portfolio_values)
        
        if len(portfolio_df) > 1:
            returns = portfolio_df['return'].pct_change().dropna()
            annualized_return = self._calculate_annualized_return(portfolio_df)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(portfolio_df)
        else:
            annualized_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade analysis
        trades_df = self.portfolio.get_trades_df()
        win_rate = self._calculate_win_rate(trades_df) if not trades_df.empty else 0

        # Get cumulative costs
        cumulative_costs = self.portfolio.cumulative_costs

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            num_trades=len(self.portfolio.trades),
            win_rate=win_rate,
            trades=self.portfolio.trades,
            portfolio_values=self.portfolio.portfolio_values,
            total_commission=cumulative_costs['commission'],
            total_slippage=cumulative_costs['slippage'],
            total_spread_cost=cumulative_costs['spread'],
            total_transaction_costs=cumulative_costs['total'],
            metadata={
                'symbol': symbol,
                'strategy': str(strategy),
                'start_date': start_date,
                'end_date': end_date,
                'commission': self.commission,
                'cost_model': str(self.cost_model.__class__.__name__)
            }
        )
    
    def _calculate_annualized_return(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate annualized return."""
        if len(portfolio_df) < 2:
            return 0
        
        start_value = portfolio_df.iloc[0]['total_value']
        end_value = portfolio_df.iloc[-1]['total_value']
        
        start_date = portfolio_df.iloc[0]['timestamp']
        end_date = portfolio_df.iloc[-1]['timestamp']
        days = (end_date - start_date).days
        
        if days == 0:
            return 0
        
        years = days / 365.25
        return ((end_value / start_value) ** (1 / years)) - 1
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        
        excess_return = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        return excess_return / returns.std() * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_df) < 2:
            return 0
        
        values = portfolio_df['total_value']
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()
    
    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate from trades."""
        if trades_df.empty:
            return 0
        
        # Group trades by symbol to calculate P&L per round trip
        buy_trades = trades_df[trades_df['side'] == 'BUY']
        sell_trades = trades_df[trades_df['side'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0
        
        winning_trades = 0
        total_pairs = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_pairs):
            buy_price = buy_trades.iloc[i]['price']
            sell_price = sell_trades.iloc[i]['price']
            
            if sell_price > buy_price:
                winning_trades += 1
        
        return winning_trades / total_pairs if total_pairs > 0 else 0

    def _save_to_database(self, strategy, symbol: str, start_date: str,
                         end_date: str, data_source: str, run_duration: float,
                         result: BacktestResult):
        """Save backtest results to database."""
        from database.models import (
            Strategy as StrategyModel,
            BacktestRun,
            BacktestResultModel,
            TradeModel,
            DailyPerformance
        )

        # Save strategy
        strategy_model = StrategyModel(
            name=str(strategy),
            strategy_type=strategy.__class__.__name__,
            parameters=strategy.parameters,
            description=strategy.get_description() if hasattr(strategy, 'get_description') else None
        )
        strategy_id = self.repository.save_strategy(strategy_model)

        # Save backtest run
        cost_model_info = {
            'type': self.cost_model.__class__.__name__,
            'commission': self.commission
        }
        if hasattr(self.cost_model, '__dict__'):
            cost_model_info.update(self.cost_model.__dict__)

        run = BacktestRun(
            strategy_id=strategy_id,
            symbol=symbol,
            start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
            end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
            initial_capital=self.initial_capital,
            data_source=data_source,
            cost_model=cost_model_info,
            run_duration_seconds=run_duration,
            status='completed'
        )
        run_id = self.repository.save_backtest_run(run)

        # Calculate additional metrics
        trades_df = self.portfolio.get_trades_df()
        winning_trades = 0
        losing_trades = 0
        total_wins = 0
        total_losses = 0

        if not trades_df.empty:
            # Pair buy and sell trades to calculate PnL
            buy_trades = trades_df[trades_df['side'] == 'BUY']
            sell_trades = trades_df[trades_df['side'] == 'SELL']

            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades.iloc[i]['price']
                sell_price = sell_trades.iloc[i]['price']
                quantity = buy_trades.iloc[i]['quantity']

                pnl = (sell_price - buy_price) * quantity
                if pnl > 0:
                    winning_trades += 1
                    total_wins += pnl
                else:
                    losing_trades += 1
                    total_losses += abs(pnl)

        # Save backtest result
        result_model = BacktestResultModel(
            run_id=run_id,
            final_value=result.final_value,
            total_return=result.total_return,
            annualized_return=result.annualized_return,
            sharpe_ratio=result.sharpe_ratio if not pd.isna(result.sharpe_ratio) else None,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            num_trades=result.num_trades,
            num_winning_trades=winning_trades,
            num_losing_trades=losing_trades,
            total_commission=result.total_commission,
            total_slippage=result.total_slippage,
            total_spread_cost=result.total_spread_cost,
            total_transaction_costs=result.total_transaction_costs,
            avg_win=total_wins / winning_trades if winning_trades > 0 else None,
            avg_loss=total_losses / losing_trades if losing_trades > 0 else None
        )
        self.repository.save_backtest_result(result_model)

        # Save trades
        if result.trades:
            trade_models = []
            for trade in result.trades:
                trade_model = TradeModel(
                    run_id=run_id,
                    trade_date=trade.timestamp,
                    action=trade.side.value,
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    price=trade.price,
                    commission=trade.commission,
                    slippage=trade.slippage,
                    spread_cost=trade.spread_cost,
                    total_cost=trade.total_cost,
                    trade_value=trade.value,
                    cash_after=self.portfolio.cash  # Current cash after trade
                )
                trade_models.append(trade_model)
            self.repository.save_trades(trade_models)

        # Save daily performance
        if result.portfolio_values:
            perf_models = []
            for pv in result.portfolio_values:
                perf_model = DailyPerformance(
                    run_id=run_id,
                    date=pv['timestamp'].date() if isinstance(pv['timestamp'], datetime) else pv['timestamp'],
                    portfolio_value=pv['total_value'],
                    cash=pv.get('cash'),
                    positions_value=pv.get('positions_value'),
                    daily_return=pv.get('return'),
                    num_positions=len(self.portfolio.positions)
                )
                perf_models.append(perf_model)
            self.repository.save_daily_performance(perf_models)