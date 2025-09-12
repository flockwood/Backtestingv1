from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .portfolio import Portfolio
from .types import Order, OrderSide, BacktestResult
from strategies.base import BaseStrategy
from data.base import BaseDataLoader


class BacktestEngine:
    """
    Backtesting engine that orchestrates the entire simulation.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 position_size: float = 1.0):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (as decimal, e.g., 0.001 = 0.1%)
            position_size: Fraction of capital to risk per trade (0.0 to 1.0)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        self.portfolio = Portfolio(initial_capital, commission)
        
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
        result = self._calculate_results(strategy, symbol, start_date, end_date)
        
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
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            position_change = row.get('Position', 0)
            
            # Record portfolio value
            self.portfolio.record_portfolio_value(
                timestamp, {symbol: current_price}
            )
            
            # Handle position changes
            if position_change == 1 and current_position == 0:
                # Enter long position
                self._enter_long_position(symbol, current_price, timestamp)
                current_position = 1
                
            elif position_change == -1 and current_position == 1:
                # Exit long position
                self._exit_long_position(symbol, current_price, timestamp)
                current_position = 0
    
    def _enter_long_position(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Enter a long position."""
        # Calculate position size based on available cash
        available_cash = self.portfolio.cash * self.position_size
        quantity = available_cash / (price * (1 + self.commission))
        
        if quantity > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                timestamp=timestamp
            )
            
            trade = self.portfolio.execute_order(order, price, timestamp)
            if trade:
                print(f"  BUY: {trade.quantity:.2f} shares at ${trade.price:.2f}")
    
    def _exit_long_position(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Exit a long position."""
        position = self.portfolio.positions.get(symbol)
        if position and position.quantity > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                timestamp=timestamp
            )
            
            trade = self.portfolio.execute_order(order, price, timestamp)
            if trade:
                print(f"  SELL: {trade.quantity:.2f} shares at ${trade.price:.2f}")
    
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
            metadata={
                'symbol': symbol,
                'strategy': str(strategy),
                'start_date': start_date,
                'end_date': end_date,
                'commission': self.commission
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