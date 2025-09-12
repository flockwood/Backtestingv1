from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from .types import Order, Position, Trade, OrderSide, OrderStatus


class Portfolio:
    """
    Portfolio class that manages cash, positions, and trade execution.
    """
    
    def __init__(self, initial_capital: float, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Dict] = []
        self._trade_id_counter = 0
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage."""
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100
    
    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in current_prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
    
    def can_execute_order(self, order: Order, current_price: float) -> bool:
        """Check if an order can be executed."""
        if order.side == OrderSide.BUY:
            required_cash = order.quantity * current_price * (1 + self.commission)
            return self.cash >= required_cash
        else:  # SELL
            position = self.positions.get(order.symbol)
            return position is not None and position.quantity >= order.quantity
    
    def execute_order(self, order: Order, current_price: float, timestamp: datetime) -> Optional[Trade]:
        """
        Execute an order and update portfolio state.
        
        Args:
            order: Order to execute
            current_price: Current market price
            timestamp: Execution timestamp
            
        Returns:
            Trade object if executed, None if execution failed
        """
        if not self.can_execute_order(order, current_price):
            return None
        
        # Calculate commission
        commission = order.quantity * current_price * self.commission
        
        # Create trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=current_price,
            timestamp=timestamp,
            trade_id=str(self._trade_id_counter),
            commission=commission
        )
        self._trade_id_counter += 1
        
        # Update portfolio
        if order.side == OrderSide.BUY:
            self._execute_buy(trade)
        else:
            self._execute_sell(trade)
        
        self.trades.append(trade)
        order.status = OrderStatus.FILLED
        
        return trade
    
    def _execute_buy(self, trade: Trade) -> None:
        """Execute a buy order."""
        total_cost = trade.value + trade.commission
        self.cash -= total_cost
        
        if trade.symbol in self.positions:
            # Update existing position
            pos = self.positions[trade.symbol]
            total_quantity = pos.quantity + trade.quantity
            total_cost_basis = pos.cost_basis + trade.value
            pos.avg_price = total_cost_basis / total_quantity
            pos.quantity = total_quantity
        else:
            # Create new position
            self.positions[trade.symbol] = Position(
                symbol=trade.symbol,
                quantity=trade.quantity,
                avg_price=trade.price,
                current_price=trade.price
            )
    
    def _execute_sell(self, trade: Trade) -> None:
        """Execute a sell order."""
        total_proceeds = trade.value - trade.commission
        self.cash += total_proceeds
        
        if trade.symbol in self.positions:
            pos = self.positions[trade.symbol]
            pos.quantity -= trade.quantity
            
            # Remove position if fully sold
            if pos.quantity <= 0:
                del self.positions[trade.symbol]
    
    def record_portfolio_value(self, timestamp: datetime, current_prices: Dict[str, float]) -> None:
        """Record current portfolio value for tracking."""
        self.update_positions(current_prices)
        
        portfolio_record = {
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': sum(pos.market_value for pos in self.positions.values()),
            'total_value': self.total_value,
            'return': self.total_return
        }
        
        self.portfolio_values.append(portfolio_record)
    
    def get_portfolio_summary(self) -> Dict:
        """Get a summary of the current portfolio state."""
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'positions_value': sum(pos.market_value for pos in self.positions.values()),
            'total_value': self.total_value,
            'total_return': self.total_return,
            'num_positions': len(self.positions),
            'num_trades': len(self.trades)
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission
            })
        
        return pd.DataFrame(trades_data)