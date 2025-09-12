import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseStrategy


class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates buy signals when short SMA crosses above long SMA,
    and sell signals when short SMA crosses below long SMA.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 20):
        parameters = {
            'short_window': short_window,
            'long_window': long_window
        }
        super().__init__("SMA Crossover", parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate SMA crossover trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus signal columns
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying original data
        signals_data = data.copy()
        
        short_window = self.parameters['short_window']
        long_window = self.parameters['long_window']
        
        # Calculate moving averages
        signals_data['SMA_short'] = signals_data['Close'].rolling(window=short_window).mean()
        signals_data['SMA_long'] = signals_data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        signals_data['Signal'] = 0
        
        # Create boolean mask for buy signals (short MA > long MA)
        buy_condition = (signals_data['SMA_short'] > signals_data['SMA_long'])
        
        # Set signals: 1 for buy, 0 for hold/sell
        signals_data.loc[buy_condition, 'Signal'] = 1
        
        # Generate position changes (1 = enter position, -1 = exit position, 0 = hold)
        signals_data['Position'] = signals_data['Signal'].diff()
        
        # Clean up the first few rows where we don't have enough data for long MA
        signals_data.iloc[:long_window, signals_data.columns.get_loc('Position')] = 0
        
        return signals_data
    
    def get_description(self) -> str:
        """Get a human-readable description of the strategy."""
        short = self.parameters['short_window']
        long = self.parameters['long_window']
        return f"SMA Crossover Strategy ({short}/{long}): Buy when {short}-period SMA crosses above {long}-period SMA"