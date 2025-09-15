"""Database models for backtesting results."""

from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import json


@dataclass
class Strategy:
    """Strategy configuration model."""

    name: str
    strategy_type: str
    parameters: Dict[str, Any]
    description: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            'name': self.name,
            'strategy_type': self.strategy_type,
            'parameters': json.dumps(self.parameters),
            'description': self.description
        }


@dataclass
class BacktestRun:
    """Backtest run model."""

    strategy_id: int
    symbol: str
    start_date: date
    end_date: date
    initial_capital: float
    data_source: Optional[str] = None
    cost_model: Optional[Dict[str, Any]] = None
    run_duration_seconds: Optional[float] = None
    status: str = "completed"
    error_message: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            'strategy_id': self.strategy_id,
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'data_source': self.data_source,
            'cost_model': json.dumps(self.cost_model) if self.cost_model else None,
            'run_duration_seconds': self.run_duration_seconds,
            'status': self.status,
            'error_message': self.error_message
        }


@dataclass
class BacktestResultModel:
    """Backtest result summary model."""

    run_id: int
    final_value: float
    total_return: float
    num_trades: int
    annualized_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    num_winning_trades: Optional[int] = None
    num_losing_trades: Optional[int] = None
    total_commission: Optional[float] = None
    total_slippage: Optional[float] = None
    total_spread_cost: Optional[float] = None
    total_transaction_costs: Optional[float] = None
    avg_trade_duration_hours: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        # Remove id if None for insertion
        if data.get('id') is None:
            data.pop('id', None)
        return data


@dataclass
class TradeModel:
    """Individual trade model."""

    run_id: int
    trade_date: datetime
    action: str  # BUY or SELL
    symbol: str
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    spread_cost: float = 0.0
    total_cost: float = 0.0
    trade_value: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    position_size: Optional[float] = None
    cash_after: Optional[float] = None
    id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        if data.get('id') is None:
            data.pop('id', None)
        return data


@dataclass
class DailyPerformance:
    """Daily portfolio performance model."""

    run_id: int
    date: date
    portfolio_value: float
    cash: Optional[float] = None
    positions_value: Optional[float] = None
    daily_return: Optional[float] = None
    cumulative_return: Optional[float] = None
    drawdown: Optional[float] = None
    positions: Optional[Dict[str, Any]] = None
    num_positions: int = 0
    id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        data = asdict(self)
        if data.get('id') is None:
            data.pop('id', None)
        if data.get('positions'):
            data['positions'] = json.dumps(data['positions'])
        return data


@dataclass
class OptimizationResult:
    """Strategy optimization result model."""

    strategy_type: str
    symbol: str
    start_date: date
    end_date: date
    parameters: Dict[str, Any]
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    num_trades: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            'strategy_type': self.strategy_type,
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'parameters': json.dumps(self.parameters),
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'num_trades': self.num_trades
        }