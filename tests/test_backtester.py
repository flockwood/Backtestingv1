import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def test_simple_backtest():
    """Test basic backtest functionality with synthetic data"""
    # Create synthetic price data
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    prices = [100 + i + np.sin(i/5)*10 for i in range(len(dates))]
    
    data = pd.DataFrame({
        'Close': prices,
        'signal': [0]*len(dates)
    }, index=dates)
    
    # Add a few manual signals
    data.iloc[5, data.columns.get_loc('signal')] = 1   # Buy on day 5
    data.iloc[20, data.columns.get_loc('signal')] = -1  # Sell on day 20
    
    # Import and run backtest
    from backtester import run_backtest
    results = run_backtest(data, initial_capital=10000)
    
    # Basic assertions
    assert results['initial_capital'] == 10000
    assert results['num_trades'] == 1
    assert len(results['trades']) == 2  # One buy, one sell
    assert results['final_value'] > 0
    
    print(f"Test passed! Return: {results['total_return']:.2f}%")

if __name__ == "__main__":
    test_simple_backtest()