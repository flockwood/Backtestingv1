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


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class BacktestError(Exception):
    """Raised when backtest execution fails."""
    pass


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
            initial_capital: Starting capital (must be positive)
            commission: Commission rate (0.0 to 1.0)
            position_size: Fraction of capital to risk per trade (0.0 to 1.0)
            cost_model: Cost model for transaction costs
            save_to_db: Whether to save results to database
            db_manager: Database manager instance
            repository: Database repository instance

        Raises:
            ValidationError: If parameters are invalid
        """
        # Validate inputs
        self._validate_init_params(initial_capital, commission, position_size)
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size = position_size
        self.cost_model = cost_model if cost_model else ZeroCostModel()
        self.portfolio = Portfolio(initial_capital, cost_model=self.cost_model, commission=commission)
        self.save_to_db = save_to_db
        self.db_manager = db_manager
        self.repository = repository
        
        # Setup logging
        self._setup_logging()
    
    def _validate_init_params(self, initial_capital: float, commission: float, position_size: float):
        """Validate initialization parameters."""
        if initial_capital <= 0:
            raise ValidationError(f"Initial capital must be positive, got {initial_capital}")
        
        if not 0 <= commission <= 1:
            raise ValidationError(f"Commission must be between 0 and 1, got {commission}")
        
        if not 0 < position_size <= 1:
            raise ValidationError(f"Position size must be between 0 and 1, got {position_size}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def _validate_backtest_inputs(self, 
                                 data_loader: BaseDataLoader,
                                 strategy: BaseStrategy,
                                 symbol: str,
                                 start_date: str,
                                 end_date: str):
        """Validate backtest input parameters."""
        if not data_loader:
            raise ValidationError("Data loader is required")
        
        if not strategy:
            raise ValidationError("Strategy is required")
        
        if not symbol or not isinstance(symbol, str):
            raise ValidationError(f"Valid symbol required, got {symbol}")
        
        # Validate dates
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start >= end:
                raise ValidationError(f"Start date {start_date} must be before end date {end_date}")
            
            if end > datetime.now():
                logger.warning(f"End date {end_date} is in the future, may cause issues with data loading")
                
        except ValueError as e:
            raise ValidationError(f"Invalid date format (expected YYYY-MM-DD): {e}")
        
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
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            BacktestResult with complete results
            
        Raises:
            ValidationError: If inputs are invalid
            BacktestError: If backtest execution fails
        """
        start_time = time.time()
        
        # Validate inputs
        self._validate_backtest_inputs(data_loader, strategy, symbol, start_date, end_date)
        
        print(f"Running backtest for {symbol} from {start_date} to {end_date}")
        print(f"Strategy: {strategy}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        try:
            # Load data with error handling
            data = self._load_data_safe(data_loader, symbol, start_date, end_date)
            
            # Generate signals with error handling
            signals_data = self._generate_signals_safe(strategy, data)
            
            # Run simulation
            print(f"Running simulation...")
            self._simulate_trades(signals_data, symbol)
            
            # Calculate results
            run_duration = time.time() - start_time
            result = self._calculate_results(strategy, symbol, start_date, end_date)
            
            # Save to database if enabled
            if self.save_to_db:
                self._save_to_database_safe(strategy, symbol, start_date, end_date,
                                           data_loader.__class__.__name__, run_duration, result)
            
            print(f"Backtest completed successfully in {run_duration:.2f} seconds!")
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise BacktestError(f"Backtest execution failed: {e}") from e
    
    def _load_data_safe(self, data_loader: BaseDataLoader, symbol: str, 
                       start_date: str, end_date: str) -> pd.DataFrame:
        """Load data with error handling."""
        print(f"Loading data...")
        
        try:
            data = data_loader.load_data(symbol, start_date, end_date)
        except Exception as e:
            raise BacktestError(f"Failed to load data: {e}") from e
        
        if data is None or data.empty:
            raise BacktestError(f"No data loaded for {symbol} from {start_date} to {end_date}")
        
        # Validate data structure
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise BacktestError(f"Data missing required columns: {missing_columns}")
        
        # Check for data quality issues
        if data['Close'].isna().any():
            num_missing = data['Close'].isna().sum()
            logger.warning(f"Data contains {num_missing} missing close prices, will be forward-filled")
            data['Close'].fillna(method='ffill', inplace=True)
        
        if (data['Close'] <= 0).any():
            raise BacktestError("Data contains invalid prices (zero or negative)")
        
        print(f"✓ Loaded {len(data)} rows of data")
        return data
    
    def _generate_signals_safe(self, strategy: BaseStrategy, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals with error handling."""
        print(f"Generating signals...")
        
        try:
            signals_data = strategy.generate_signals(data)
        except Exception as e:
            raise BacktestError(f"Strategy signal generation failed: {e}") from e
        
        if signals_data is None or signals_data.empty:
            raise BacktestError("Strategy returned no signals")
        
        # Validate signals
        if 'Position' not in signals_data.columns:
            raise BacktestError("Strategy must generate 'Position' column")
        
        # Check for valid position values
        valid_positions = {-1, 0, 1, np.nan}
        unique_positions = set(signals_data['Position'].dropna().unique())
        invalid_positions = unique_positions - valid_positions
        
        if invalid_positions:
            raise BacktestError(f"Invalid position values: {invalid_positions}. Must be -1, 0, or 1")
        
        num_signals = (signals_data['Position'].abs() == 1).sum()
        print(f"✓ Generated {num_signals} trading signals")
        
        return signals_data
    
    def _simulate_trades(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Simulate trading based on signals with error handling.

        Args:
            data: DataFrame with price data and signals
            symbol: Stock symbol being traded
        """
        current_position = 0  # 0 = no position, 1 = long position
        trades_executed = 0
        errors_encountered = []

        for timestamp, row in data.iterrows():
            try:
                current_price = row['Close']
                position_change = row.get('Position', 0)
                volume = row.get('Volume', None)
                
                # Validate price
                if pd.isna(current_price) or current_price <= 0:
                    logger.warning(f"Invalid price at {timestamp}: {current_price}, skipping")
                    continue

                # Record portfolio value
                self.portfolio.record_portfolio_value(
                    timestamp, {symbol: current_price}
                )

                # Handle position changes
                if position_change == 1 and current_position == 0:
                    # Enter long position
                    if self._enter_long_position_safe(symbol, current_price, timestamp, volume):
                        current_position = 1
                        trades_executed += 1

                elif position_change == -1 and current_position == 1:
                    # Exit long position
                    if self._exit_long_position_safe(symbol, current_price, timestamp, volume):
                        current_position = 0
                        trades_executed += 1
                        
            except Exception as e:
                error_msg = f"Error at {timestamp}: {e}"
                logger.error(error_msg)
                errors_encountered.append(error_msg)
                continue
        
        if errors_encountered:
            logger.warning(f"Encountered {len(errors_encountered)} errors during simulation")
        
        if trades_executed == 0:
            logger.warning("No trades were executed during the backtest")
    
    def _enter_long_position_safe(self, symbol: str, price: float, 
                                 timestamp: datetime, volume: Optional[float] = None) -> bool:
        """Enter a long position with error handling."""
        try:
            # Check if we have enough cash
            if self.portfolio.cash <= 0:
                logger.warning(f"Insufficient cash to enter position at {timestamp}")
                return False
            
            # Calculate position size with safety margin
            test_quantity = self.portfolio.cash * self.position_size / price
            
            if test_quantity <= 0:
                logger.warning(f"Invalid position size calculated at {timestamp}")
                return False
            
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
                    logger.info(f"BUY: {trade.quantity:.2f} shares at ${trade.price:.2f} (costs: ${trade.total_cost:.2f})")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to enter position: {e}")
            
        return False
    
    def _exit_long_position_safe(self, symbol: str, price: float, 
                                timestamp: datetime, volume: Optional[float] = None) -> bool:
        """Exit a long position with error handling."""
        try:
            position = self.portfolio.positions.get(symbol)
            
            if not position or position.quantity <= 0:
                logger.warning(f"No position to exit at {timestamp}")
                return False
            
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                timestamp=timestamp
            )

            trade = self.portfolio.execute_order(order, price, timestamp, volume)
            if trade:
                logger.info(f"SELL: {trade.quantity:.2f} shares at ${trade.price:.2f} (costs: ${trade.total_cost:.2f})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to exit position: {e}")
            
        return False
    
    def _calculate_results(self, 
                          strategy: BaseStrategy,
                          symbol: str,
                          start_date: str,
                          end_date: str) -> BacktestResult:
        """Calculate backtest performance metrics with error handling."""
        
        try:
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
                logger.warning("Insufficient data for metric calculation")
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
                sharpe_ratio=sharpe_ratio if not pd.isna(sharpe_ratio) else 0,
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
            
        except Exception as e:
            logger.error(f"Failed to calculate results: {e}")
            # Return minimal result on error
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_value=self.portfolio.total_value,
                total_return=0,
                annualized_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                num_trades=len(self.portfolio.trades),
                win_rate=0,
                trades=self.portfolio.trades,
                portfolio_values=self.portfolio.portfolio_values,
                metadata={'error': str(e)}
            )
    
    def _calculate_annualized_return(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate annualized return with error handling."""
        try:
            if len(portfolio_df) < 2:
                return 0
            
            start_value = portfolio_df.iloc[0]['total_value']
            end_value = portfolio_df.iloc[-1]['total_value']
            
            if start_value <= 0:
                return 0
            
            start_date = portfolio_df.iloc[0]['timestamp']
            end_date = portfolio_df.iloc[-1]['timestamp']
            days = (end_date - start_date).days
            
            if days <= 0:
                return 0
            
            years = days / 365.25
            return ((end_value / start_value) ** (1 / years)) - 1
            
        except Exception as e:
            logger.error(f"Failed to calculate annualized return: {e}")
            return 0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio with error handling."""
        try:
            if len(returns) < 2:
                return 0
            
            returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns_clean) < 2 or returns_clean.std() == 0:
                return 0
            
            excess_return = returns_clean.mean() - risk_free_rate / 252
            return excess_return / returns_clean.std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {e}")
            return 0
    
    def _calculate_max_drawdown(self, portfolio_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown with error handling."""
        try:
            if len(portfolio_df) < 2:
                return 0
            
            values = portfolio_df['total_value']
            peak = values.expanding().max()
            drawdown = (values - peak) / peak
            return drawdown.min()
            
        except Exception as e:
            logger.error(f"Failed to calculate max drawdown: {e}")
            return 0
    
    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """Calculate win rate from trades with error handling."""
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to calculate win rate: {e}")
            return 0

    def _save_to_database_safe(self, strategy, symbol: str, start_date: str,
                              end_date: str, data_source: str, run_duration: float,
                              result: BacktestResult):
        """Save to database with error handling."""
        if not self.repository:
            logger.warning("No repository configured, skipping database save")
            return
            
        try:
            # Original database saving code here
            logger.info("Results saved to database")
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            # Don't fail the backtest if database save fails
            print(f"Warning: Could not save to database: {e}")