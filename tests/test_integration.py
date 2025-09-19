"""
Comprehensive integration tests for the backtesting system.
Tests error handling, validation, and end-to-end workflow.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import BacktestEngine
from core.types import BacktestResult, OrderSide
from core.costs import SimpleCostModel, ZeroCostModel
from strategies import SMAStrategy, BaseStrategy
from data.base import BaseDataLoader


# ==============================================================================
# Mock Components for Testing
# ==============================================================================

class MockDataLoader(BaseDataLoader):
    """Mock data loader for testing."""
    
    def __init__(self, data=None, should_fail=False):
        super().__init__("Mock")
        self.data = data
        self.should_fail = should_fail
    
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Return mock data or raise error if configured."""
        if self.should_fail:
            raise Exception("Mock data loader failure")
        
        if self.data is not None:
            return self.data
        
        # Generate synthetic data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Create realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n)
        prices = 100 * (1 + returns).cumprod()
        
        data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.99, 1.01, n),
            'High': prices * np.random.uniform(1.01, 1.03, n),
            'Low': prices * np.random.uniform(0.97, 0.99, n),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n),
            'Adj Close': prices
        }, index=dates)
        
        return data


class BrokenStrategy(BaseStrategy):
    """Strategy that raises errors for testing."""
    
    def __init__(self):
        super().__init__("Broken", {})
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise Exception("Strategy is broken")


class InvalidSignalStrategy(BaseStrategy):
    """Strategy that generates invalid signals."""
    
    def __init__(self):
        super().__init__("Invalid", {})
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = data.copy()
        signals['Position'] = 999  # Invalid position value
        return signals


# ==============================================================================
# Test Validation
# ==============================================================================

def test_engine_initialization_validation():
    """Test BacktestEngine initialization parameter validation."""
    
    # Test invalid initial capital
    with pytest.raises(Exception) as exc_info:
        engine = BacktestEngine(initial_capital=-1000)
    assert "Initial capital must be positive" in str(exc_info.value)
    
    # Test invalid commission
    with pytest.raises(Exception) as exc_info:
        engine = BacktestEngine(initial_capital=10000, commission=1.5)
    assert "Commission must be between 0 and 1" in str(exc_info.value)
    
    # Test invalid position size
    with pytest.raises(Exception) as exc_info:
        engine = BacktestEngine(initial_capital=10000, position_size=1.5)
    assert "Position size must be between 0 and 1" in str(exc_info.value)
    
    # Test valid initialization
    engine = BacktestEngine(initial_capital=10000, commission=0.001, position_size=0.95)
    assert engine.initial_capital == 10000
    assert engine.commission == 0.001
    assert engine.position_size == 0.95


def test_backtest_input_validation():
    """Test backtest input parameter validation."""
    
    engine = BacktestEngine(initial_capital=10000)
    data_loader = MockDataLoader()
    strategy = SMAStrategy()
    
    # Test missing data loader
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=None,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    assert "Data loader is required" in str(exc_info.value)
    
    # Test missing strategy
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=None,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    assert "Strategy is required" in str(exc_info.value)
    
    # Test invalid symbol
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    assert "Valid symbol required" in str(exc_info.value)
    
    # Test invalid date format
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='01/01/2023',  # Wrong format
            end_date='2023-12-31'
        )
    assert "Invalid date format" in str(exc_info.value)
    
    # Test start date after end date
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-12-31',
            end_date='2023-01-01'
        )
    assert "Start date" in str(exc_info.value) and "before end date" in str(exc_info.value)


# ==============================================================================
# Test Error Handling
# ==============================================================================

def test_data_loader_failure():
    """Test handling of data loader failures."""
    
    engine = BacktestEngine(initial_capital=10000)
    data_loader = MockDataLoader(should_fail=True)
    strategy = SMAStrategy()
    
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    assert "Failed to load data" in str(exc_info.value)


def test_empty_data_handling():
    """Test handling of empty data."""
    
    engine = BacktestEngine(initial_capital=10000)
    empty_data = pd.DataFrame()
    data_loader = MockDataLoader(data=empty_data)
    strategy = SMAStrategy()
    
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    assert "No data loaded" in str(exc_info.value)


def test_missing_columns():
    """Test handling of data with missing columns."""
    
    engine = BacktestEngine(initial_capital=10000)
    
    # Data missing required columns
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    bad_data = pd.DataFrame({
        'Price': [100] * len(dates)  # Wrong column name
    }, index=dates)
    
    data_loader = MockDataLoader(data=bad_data)
    strategy = SMAStrategy()
    
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
    assert "missing required columns" in str(exc_info.value)


def test_invalid_price_data():
    """Test handling of invalid price data."""
    
    engine = BacktestEngine(initial_capital=10000)
    
    # Data with invalid prices
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    bad_data = pd.DataFrame({
        'Open': [100] * len(dates),
        'High': [110] * len(dates),
        'Low': [90] * len(dates),
        'Close': [-50] * len(dates),  # Negative prices
        'Volume': [1000000] * len(dates)
    }, index=dates)
    
    data_loader = MockDataLoader(data=bad_data)
    strategy = SMAStrategy()
    
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
    assert "invalid prices" in str(exc_info.value).lower()


def test_strategy_failure():
    """Test handling of strategy failures."""
    
    engine = BacktestEngine(initial_capital=10000)
    data_loader = MockDataLoader()
    strategy = BrokenStrategy()
    
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
    assert "Strategy" in str(exc_info.value) and "failed" in str(exc_info.value)


def test_invalid_signals():
    """Test handling of invalid strategy signals."""
    
    engine = BacktestEngine(initial_capital=10000)
    data_loader = MockDataLoader()
    strategy = InvalidSignalStrategy()
    
    with pytest.raises(Exception) as exc_info:
        engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='AAPL',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
    assert "Invalid position values" in str(exc_info.value)


# ==============================================================================
# Test End-to-End Workflow
# ==============================================================================

def test_successful_backtest():
    """Test a successful end-to-end backtest."""
    
    engine = BacktestEngine(initial_capital=10000, commission=0.001)
    data_loader = MockDataLoader()
    strategy = SMAStrategy(short_window=5, long_window=10)
    
    result = engine.run_backtest(
        data_loader=data_loader,
        strategy=strategy,
        symbol='TEST',
        start_date='2023-01-01',
        end_date='2023-03-31'
    )
    
    # Verify result structure
    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 10000
    assert result.final_value > 0
    assert isinstance(result.total_return, (int, float))
    assert isinstance(result.sharpe_ratio, (int, float))
    assert isinstance(result.max_drawdown, (int, float))
    assert result.num_trades >= 0
    assert 0 <= result.win_rate <= 1
    
    # Verify metadata
    assert result.metadata['symbol'] == 'TEST'
    assert 'SMA' in result.metadata['strategy']
    
    print(f"✓ Successful backtest: Return={result.total_return:.2%}, Trades={result.num_trades}")


def test_backtest_with_costs():
    """Test backtest with transaction costs."""
    
    cost_model = SimpleCostModel(
        fixed_fee=1.0,
        percentage_fee=0.001,
        base_slippage_bps=5,
        market_impact_coefficient=10,
        bid_ask_spread_bps=10
    )
    
    engine = BacktestEngine(
        initial_capital=10000,
        cost_model=cost_model
    )
    
    data_loader = MockDataLoader()
    strategy = SMAStrategy(short_window=5, long_window=10)
    
    result = engine.run_backtest(
        data_loader=data_loader,
        strategy=strategy,
        symbol='TEST',
        start_date='2023-01-01',
        end_date='2023-03-31'
    )
    
    # Verify costs were applied
    assert result.total_commission >= 0
    assert result.total_slippage >= 0
    assert result.total_spread_cost >= 0
    assert result.total_transaction_costs > 0
    
    print(f"✓ Backtest with costs: Total costs=${result.total_transaction_costs:.2f}")


def test_no_trades_scenario():
    """Test scenario where no trades are executed."""
    
    # Create data with no clear trend
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    flat_data = pd.DataFrame({
        'Open': [100] * len(dates),
        'High': [100.1] * len(dates),
        'Low': [99.9] * len(dates),
        'Close': [100] * len(dates),
        'Volume': [1000000] * len(dates)
    }, index=dates)
    
    engine = BacktestEngine(initial_capital=10000)
    data_loader = MockDataLoader(data=flat_data)
    strategy = SMAStrategy(short_window=10, long_window=20)
    
    result = engine.run_backtest(
        data_loader=data_loader,
        strategy=strategy,
        symbol='TEST',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # Should complete even with no trades
    assert result.num_trades == 0
    assert result.total_return == 0
    assert result.win_rate == 0
    
    print(f"✓ No trades scenario handled correctly")


# ==============================================================================
# Test Portfolio Management
# ==============================================================================

def test_insufficient_cash():
    """Test handling of insufficient cash for trades."""
    
    engine = BacktestEngine(initial_capital=100)  # Very small capital
    data_loader = MockDataLoader()
    strategy = SMAStrategy(short_window=5, long_window=10)
    
    # Should handle gracefully without crashing
    result = engine.run_backtest(
        data_loader=data_loader,
        strategy=strategy,
        symbol='TEST',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    assert result is not None
    print(f"✓ Insufficient cash handled correctly")


def test_position_sizing():
    """Test different position sizing configurations."""
    
    for position_size in [0.5, 0.75, 0.95, 1.0]:
        engine = BacktestEngine(
            initial_capital=10000,
            position_size=position_size
        )
        
        data_loader = MockDataLoader()
        strategy = SMAStrategy(short_window=5, long_window=10)
        
        result = engine.run_backtest(
            data_loader=data_loader,
            strategy=strategy,
            symbol='TEST',
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        assert result is not None
        print(f"✓ Position size {position_size} works correctly")


# ==============================================================================
# Run Tests
# ==============================================================================

def run_all_tests():
    """Run all integration tests."""
    
    print("="*60)
    print("INTEGRATION TESTS")
    print("="*60)
    
    test_functions = [
        test_engine_initialization_validation,
        test_backtest_input_validation,
        test_data_loader_failure,
        test_empty_data_handling,
        test_missing_columns,
        test_invalid_price_data,
        test_strategy_failure,
        test_invalid_signals,
        test_successful_backtest,
        test_backtest_with_costs,
        test_no_trades_scenario,
        test_insufficient_cash,
        test_position_sizing
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)