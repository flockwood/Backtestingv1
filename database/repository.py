"""Repository layer for database operations."""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from .db_manager import DatabaseManager
from .models import (
    Strategy, BacktestRun, BacktestResultModel,
    TradeModel, DailyPerformance, OptimizationResult
)


logger = logging.getLogger(__name__)


class BacktestRepository:
    """Repository for backtest data operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize repository with database manager."""
        self.db = db_manager

    # Strategy operations
    def save_strategy(self, strategy: Strategy) -> int:
        """Save strategy and return its ID."""
        query = """
            INSERT INTO strategies (name, strategy_type, parameters, description)
            VALUES (%(name)s, %(strategy_type)s, %(parameters)s, %(description)s)
            ON CONFLICT DO NOTHING
            RETURNING id
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, strategy.to_dict())
            result = cursor.fetchone()

            if result:
                return result['id']

            # If insert was skipped due to conflict, get existing ID
            select_query = """
                SELECT id FROM strategies
                WHERE name = %(name)s AND strategy_type = %(strategy_type)s
                AND parameters = %(parameters)s
            """
            cursor.execute(select_query, strategy.to_dict())
            result = cursor.fetchone()
            return result['id'] if result else None

    def get_strategy(self, strategy_id: int) -> Optional[Strategy]:
        """Get strategy by ID."""
        query = "SELECT * FROM strategies WHERE id = %s"

        with self.db.get_cursor() as cursor:
            cursor.execute(query, (strategy_id,))
            result = cursor.fetchone()

            if result:
                return Strategy(
                    id=result['id'],
                    name=result['name'],
                    strategy_type=result['strategy_type'],
                    parameters=result['parameters'],
                    description=result['description'],
                    created_at=result['created_at']
                )
            return None

    def find_strategies(self, strategy_type: Optional[str] = None) -> List[Strategy]:
        """Find strategies by type."""
        if strategy_type:
            query = "SELECT * FROM strategies WHERE strategy_type = %s ORDER BY created_at DESC"
            params = (strategy_type,)
        else:
            query = "SELECT * FROM strategies ORDER BY created_at DESC"
            params = None

        results = self.db.execute_query(query, params)

        return [Strategy(
            id=r['id'],
            name=r['name'],
            strategy_type=r['strategy_type'],
            parameters=r['parameters'],
            description=r['description'],
            created_at=r['created_at']
        ) for r in results]

    # Backtest run operations
    def save_backtest_run(self, run: BacktestRun) -> int:
        """Save backtest run and return its ID."""
        query = """
            INSERT INTO backtest_runs
            (strategy_id, symbol, start_date, end_date, initial_capital,
             data_source, cost_model, run_duration_seconds, status, error_message)
            VALUES (%(strategy_id)s, %(symbol)s, %(start_date)s, %(end_date)s,
                    %(initial_capital)s, %(data_source)s, %(cost_model)s,
                    %(run_duration_seconds)s, %(status)s, %(error_message)s)
            ON CONFLICT (strategy_id, symbol, start_date, end_date, initial_capital)
            DO UPDATE SET
                run_duration_seconds = EXCLUDED.run_duration_seconds,
                status = EXCLUDED.status,
                created_at = CURRENT_TIMESTAMP
            RETURNING id
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, run.to_dict())
            result = cursor.fetchone()
            return result['id'] if result else None

    # Backtest result operations
    def save_backtest_result(self, result: BacktestResultModel) -> int:
        """Save backtest result."""
        query = """
            INSERT INTO backtest_results
            (run_id, final_value, total_return, annualized_return, sharpe_ratio,
             sortino_ratio, max_drawdown, win_rate, num_trades, num_winning_trades,
             num_losing_trades, total_commission, total_slippage, total_spread_cost,
             total_transaction_costs, avg_trade_duration_hours, profit_factor,
             avg_win, avg_loss, largest_win, largest_loss)
            VALUES (%(run_id)s, %(final_value)s, %(total_return)s, %(annualized_return)s,
                    %(sharpe_ratio)s, %(sortino_ratio)s, %(max_drawdown)s, %(win_rate)s,
                    %(num_trades)s, %(num_winning_trades)s, %(num_losing_trades)s,
                    %(total_commission)s, %(total_slippage)s, %(total_spread_cost)s,
                    %(total_transaction_costs)s, %(avg_trade_duration_hours)s,
                    %(profit_factor)s, %(avg_win)s, %(avg_loss)s,
                    %(largest_win)s, %(largest_loss)s)
            ON CONFLICT (run_id) DO UPDATE SET
                final_value = EXCLUDED.final_value,
                total_return = EXCLUDED.total_return,
                num_trades = EXCLUDED.num_trades
            RETURNING id
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, result.to_dict())
            result = cursor.fetchone()
            return result['id'] if result else None

    # Trade operations
    def save_trades(self, trades: List[TradeModel]) -> int:
        """Save multiple trades."""
        if not trades:
            return 0

        query = """
            INSERT INTO trades
            (run_id, trade_date, action, symbol, quantity, price,
             commission, slippage, spread_cost, total_cost, trade_value,
             pnl, pnl_percent, position_size, cash_after)
            VALUES (%(run_id)s, %(trade_date)s, %(action)s, %(symbol)s,
                    %(quantity)s, %(price)s, %(commission)s, %(slippage)s,
                    %(spread_cost)s, %(total_cost)s, %(trade_value)s,
                    %(pnl)s, %(pnl_percent)s, %(position_size)s, %(cash_after)s)
        """

        trades_data = [trade.to_dict() for trade in trades]
        with self.db.get_cursor(dict_cursor=False) as cursor:
            cursor.executemany(query, trades_data)
            return cursor.rowcount

    # Daily performance operations
    def save_daily_performance(self, performances: List[DailyPerformance]) -> int:
        """Save daily performance data."""
        if not performances:
            return 0

        query = """
            INSERT INTO daily_performance
            (run_id, date, portfolio_value, cash, positions_value,
             daily_return, cumulative_return, drawdown, positions, num_positions)
            VALUES (%(run_id)s, %(date)s, %(portfolio_value)s, %(cash)s,
                    %(positions_value)s, %(daily_return)s, %(cumulative_return)s,
                    %(drawdown)s, %(positions)s, %(num_positions)s)
            ON CONFLICT (run_id, date) DO UPDATE SET
                portfolio_value = EXCLUDED.portfolio_value,
                positions = EXCLUDED.positions
        """

        perf_data = [perf.to_dict() for perf in performances]
        with self.db.get_cursor(dict_cursor=False) as cursor:
            cursor.executemany(query, perf_data)
            return cursor.rowcount

    # Query operations
    def get_backtest_runs(self,
                         symbol: Optional[str] = None,
                         strategy_id: Optional[int] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get backtest runs with filters."""
        conditions = ["br.status = 'completed'"]
        params = []

        if symbol:
            conditions.append("br.symbol = %s")
            params.append(symbol)

        if strategy_id:
            conditions.append("br.strategy_id = %s")
            params.append(strategy_id)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                br.*,
                s.name as strategy_name,
                s.strategy_type,
                res.total_return,
                res.sharpe_ratio,
                res.max_drawdown,
                res.num_trades
            FROM backtest_runs br
            JOIN strategies s ON br.strategy_id = s.id
            LEFT JOIN backtest_results res ON br.id = res.run_id
            WHERE {where_clause}
            ORDER BY br.created_at DESC
            LIMIT %s
        """

        params.append(limit)
        return self.db.execute_query(query, tuple(params))

    def get_best_strategies(self,
                           symbol: Optional[str] = None,
                           min_trades: int = 5,
                           limit: int = 20) -> List[Dict[str, Any]]:
        """Get best performing strategies."""
        conditions = ["br.status = 'completed'", "res.num_trades >= %s"]
        params = [min_trades]

        if symbol:
            conditions.append("br.symbol = %s")
            params.append(symbol)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                s.name,
                s.strategy_type,
                s.parameters,
                br.symbol,
                AVG(res.total_return) as avg_return,
                AVG(res.sharpe_ratio) as avg_sharpe,
                AVG(res.max_drawdown) as avg_drawdown,
                COUNT(*) as num_runs,
                SUM(res.num_trades) as total_trades
            FROM strategies s
            JOIN backtest_runs br ON s.id = br.strategy_id
            JOIN backtest_results res ON br.id = res.run_id
            WHERE {where_clause}
            GROUP BY s.id, s.name, s.strategy_type, s.parameters, br.symbol
            ORDER BY avg_sharpe DESC NULLS LAST
            LIMIT %s
        """

        params.append(limit)
        return self.db.execute_query(query, tuple(params))

    def get_trades_for_run(self, run_id: int) -> List[Dict[str, Any]]:
        """Get all trades for a specific run."""
        query = """
            SELECT * FROM trades
            WHERE run_id = %s
            ORDER BY trade_date
        """
        return self.db.execute_query(query, (run_id,))

    def get_daily_performance(self, run_id: int) -> List[Dict[str, Any]]:
        """Get daily performance for a specific run."""
        query = """
            SELECT * FROM daily_performance
            WHERE run_id = %s
            ORDER BY date
        """
        return self.db.execute_query(query, (run_id,))

    # Optimization operations
    def save_optimization_result(self, result: OptimizationResult) -> int:
        """Save optimization result."""
        query = """
            INSERT INTO optimization_results
            (strategy_type, symbol, start_date, end_date, parameters,
             total_return, sharpe_ratio, max_drawdown, num_trades)
            VALUES (%(strategy_type)s, %(symbol)s, %(start_date)s, %(end_date)s,
                    %(parameters)s, %(total_return)s, %(sharpe_ratio)s,
                    %(max_drawdown)s, %(num_trades)s)
            RETURNING id
        """

        with self.db.get_cursor() as cursor:
            cursor.execute(query, result.to_dict())
            result = cursor.fetchone()
            return result['id'] if result else None

    def get_optimization_results(self,
                                strategy_type: str,
                                symbol: str,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get optimization results for a strategy and symbol."""
        query = """
            SELECT * FROM optimization_results
            WHERE strategy_type = %s AND symbol = %s
            ORDER BY sharpe_ratio DESC NULLS LAST
            LIMIT %s
        """
        return self.db.execute_query(query, (strategy_type, symbol, limit))

    # Statistics
    def get_strategy_statistics(self, strategy_id: int) -> Dict[str, Any]:
        """Get aggregated statistics for a strategy."""
        query = "SELECT * FROM calculate_strategy_stats(%s)"
        results = self.db.execute_query(query, (strategy_id,))
        return results[0] if results else None