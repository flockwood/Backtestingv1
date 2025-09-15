from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from .types import Order, Position, Trade, OrderSide, OrderStatus
from .costs import CostModel, ZeroCostModel


class Portfolio:
    """
    Portfolio class that manages cash, positions, and trade execution.
    """
    
    def __init__(self, initial_capital: float, cost_model: Optional[CostModel] = None, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.cost_model = cost_model if cost_model else ZeroCostModel()
        self.commission = commission  # Keep for backward compatibility
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Dict] = []
        self._trade_id_counter = 0
        self.cumulative_costs = {
            'commission': 0.0,
            'slippage': 0.0,
            'spread': 0.0,
            'total': 0.0
        }
    
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
    
    def can_execute_order(self, order: Order, current_price: float, volume: Optional[float] = None) -> bool:
        """Check if an order can be executed."""
        if order.side == OrderSide.BUY:
            # Calculate costs to check if we have enough cash
            costs = self.cost_model.calculate_costs(
                side=order.side,
                quantity=order.quantity,
                price=current_price,
                volume=volume
            )
            required_cash = (order.quantity * current_price) + costs.total_cost
            return self.cash >= required_cash
        else:  # SELL
            position = self.positions.get(order.symbol)
            return position is not None and position.quantity >= order.quantity
    
    def execute_order(self, order: Order, current_price: float, timestamp: datetime, volume: Optional[float] = None) -> Optional[Trade]:
        """
        Execute an order and update portfolio state.

        Args:
            order: Order to execute
            current_price: Current market price
            timestamp: Execution timestamp
            volume: Average daily volume for slippage calculation

        Returns:
            Trade object if executed, None if execution failed
        """
        if not self.can_execute_order(order, current_price, volume):
            return None

        # Calculate transaction costs
        costs = self.cost_model.calculate_costs(
            side=order.side,
            quantity=order.quantity,
            price=current_price,
            volume=volume
        )

        # Adjust execution price for slippage
        if order.side == OrderSide.BUY:
            # Buy at higher price due to slippage
            execution_price = current_price * (1 + costs.slippage / (order.quantity * current_price))
        else:
            # Sell at lower price due to slippage
            execution_price = current_price * (1 - costs.slippage / (order.quantity * current_price))

        # Create trade
        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=timestamp,
            trade_id=str(self._trade_id_counter),
            commission=costs.commission,
            slippage=costs.slippage,
            spread_cost=costs.spread_cost,
            total_cost=costs.total_cost
        )
        self._trade_id_counter += 1

        # Update cumulative costs
        self.cumulative_costs['commission'] += costs.commission
        self.cumulative_costs['slippage'] += costs.slippage
        self.cumulative_costs['spread'] += costs.spread_cost
        self.cumulative_costs['total'] += costs.total_cost

        # Update portfolio
        if order.side == OrderSide.BUY:
            self._execute_buy(trade, costs)
        else:
            self._execute_sell(trade, costs)

        self.trades.append(trade)
        order.status = OrderStatus.FILLED

        return trade
    
    def _execute_buy(self, trade: Trade, costs) -> None:
        """Execute a buy order."""
        # Total cost includes trade value plus all transaction costs
        total_cost = trade.value + costs.total_cost
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
    
    def _execute_sell(self, trade: Trade, costs) -> None:
        """Execute a sell order."""
        # Total proceeds is trade value minus all transaction costs
        total_proceeds = trade.value - costs.total_cost
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
            'num_trades': len(self.trades),
            'cumulative_costs': self.cumulative_costs
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
                'commission': trade.commission,
                'slippage': trade.slippage,
                'spread_cost': trade.spread_cost,
                'total_cost': trade.total_cost
            })
        
        return pd.DataFrame(trades_data)