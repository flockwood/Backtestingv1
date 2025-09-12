from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # None for market orders
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_price) * self.quantity
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price


@dataclass
class Trade:
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    trade_id: Optional[str] = None
    commission: float = 0.0
    
    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class BacktestResult:
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    trades: list
    portfolio_values: list
    metadata: Dict[str, Any]