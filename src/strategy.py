import pandas as pd
import numpy as np

def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()

def generate_sma_crossover_signals(data, short_window=10, long_window=20):
    """
    Generate buy/sell signals based on SMA crossover
    
    Args:
        data: DataFrame with price data (must have 'Close' column)
        short_window: Period for short SMA
        long_window: Period for long SMA
    
    Returns:
        DataFrame with signals column added
    """
    # Create a copy to avoid modifying original data
    signals = data.copy()
    
    # Calculate SMAs
    signals['SMA_short'] = calculate_sma(signals['Close'], short_window)
    signals['SMA_long'] = calculate_sma(signals['Close'], long_window)
    
    # Initialize signal column
    signals['signal'] = 0
    
    # Create position column (1 when short > long, 0 otherwise)
    signals['position'] = 0
    signals.loc[signals.index[long_window:], 'position'] = np.where(
        signals['SMA_short'][long_window:] > signals['SMA_long'][long_window:], 1, 0
    )
    
    # Generate trading signals (change in position)
    signals['signal'] = signals['position'].diff()
    
    # Count actual trading signals
    buy_signals = len(signals[signals['signal'] == 1])
    sell_signals = len(signals[signals['signal'] == -1])
    
    print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
    
    return signals