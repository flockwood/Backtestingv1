"""
Bollinger Bands Strategy

A volatility-based strategy that uses standard deviations to identify overbought/oversold conditions.
Trades on mean reversion when price touches the bands or breakouts when price breaks through.

Buy Signal: Price touches lower band (mean reversion) or breaks above upper band (breakout)
Sell Signal: Price touches upper band (mean reversion) or breaks below lower band (breakdown)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands trading strategy.
    
    Can be configured for:
    1. Mean Reversion: Trade bounces off the bands
    2. Breakout: Trade breaks through the bands
    """
    
    def __init__(self,
                 period: int = 20,
                 num_std: float = 2.0,
                 strategy_type: str = 'mean_reversion'):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            period: Moving average period (typically 20)
            num_std: Number of standard deviations for bands (typically 2)
            strategy_type: 'mean_reversion' or 'breakout'
        """
        if strategy_type not in ['mean_reversion', 'breakout']:
            raise ValueError("strategy_type must be 'mean_reversion' or 'breakout'")
            
        parameters = {
            'period': period,
            'num_std': num_std,
            'strategy_type': strategy_type
        }
        super().__init__("Bollinger Bands", parameters)
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band, band_width, percent_b)
        """
        period = self.parameters['period']
        num_std = self.parameters['num_std']
        
        # Calculate middle band (SMA)
        middle_band = data['Close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        std_dev = data['Close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        # Calculate band width (measure of volatility)
        band_width = (upper_band - lower_band) / middle_band
        
        # Calculate %B (position of price relative to bands)
        # %B = (Price - Lower Band) / (Upper Band - Lower Band)
        percent_b = (data['Close'] - lower_band) / (upper_band - lower_band)
        
        return middle_band, upper_band, lower_band, band_width, percent_b
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Bollinger Bands trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus signal columns
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying original data
        signals_data = data.copy()
        
        # Calculate Bollinger Bands
        middle, upper, lower, width, percent_b = self.calculate_bollinger_bands(signals_data)
        
        # Add indicators to dataframe
        signals_data['BB_Middle'] = middle
        signals_data['BB_Upper'] = upper
        signals_data['BB_Lower'] = lower
        signals_data['BB_Width'] = width
        signals_data['BB_PercentB'] = percent_b
        
        # Initialize signals
        signals_data['Signal'] = 0
        signals_data['Position'] = 0
        
        period = self.parameters['period']
        strategy_type = self.parameters['strategy_type']
        
        # Generate signals based on strategy type
        for i in range(period, len(signals_data)):
            current_price = signals_data['Close'].iloc[i]
            prev_price = signals_data['Close'].iloc[i-1]
            current_upper = signals_data['BB_Upper'].iloc[i]
            current_lower = signals_data['BB_Lower'].iloc[i]
            current_middle = signals_data['BB_Middle'].iloc[i]
            prev_upper = signals_data['BB_Upper'].iloc[i-1]
            prev_lower = signals_data['BB_Lower'].iloc[i-1]
            
            # Skip if any values are NaN
            if pd.isna(current_upper) or pd.isna(current_lower):
                continue
            
            if strategy_type == 'mean_reversion':
                # Mean Reversion Strategy
                # Buy when price touches/crosses below lower band
                if current_price <= current_lower and prev_price > prev_lower:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = 1
                    
                # Sell when price touches/crosses above upper band
                elif current_price >= current_upper and prev_price < prev_upper:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = -1
                    
                # Alternative exit: price crosses middle band
                elif prev_price < current_middle and current_price > current_middle:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = -1
                    
            else:  # breakout strategy
                # Breakout Strategy
                # Buy when price breaks above upper band
                if current_price > current_upper and prev_price <= prev_upper:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = 1
                    
                # Sell when price breaks below lower band or returns to middle
                elif current_price < current_lower and prev_price >= prev_lower:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = -1
                elif current_price < current_middle and prev_price >= current_middle:
                    # Exit on return to middle band
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = -1
        
        # Generate position changes
        in_position = False
        for i in range(len(signals_data)):
            if signals_data['Signal'].iloc[i] == 1:
                if not in_position:
                    signals_data.iloc[i, signals_data.columns.get_loc('Position')] = 1
                    in_position = True
            elif signals_data['Signal'].iloc[i] == -1:
                if in_position:
                    signals_data.iloc[i, signals_data.columns.get_loc('Position')] = -1
                    in_position = False
        
        # Add squeeze indicator (low volatility periods)
        # Squeeze = when BB width is at its lowest in the period
        signals_data['BB_Squeeze'] = signals_data['BB_Width'] < signals_data['BB_Width'].rolling(window=period).mean()
        
        return signals_data
    
    def get_description(self) -> str:
        """Get a human-readable description of the strategy."""
        period = self.parameters['period']
        num_std = self.parameters['num_std']
        strategy_type = self.parameters['strategy_type']
        
        if strategy_type == 'mean_reversion':
            return (f"Bollinger Bands({period}, {num_std}σ) Mean Reversion: "
                    f"Buy at lower band, Sell at upper band")
        else:
            return (f"Bollinger Bands({period}, {num_std}σ) Breakout: "
                    f"Buy on break above upper band, Sell on break below lower band")
    
    def get_indicators(self) -> Dict[str, Any]:
        """Return the indicators used by this strategy for plotting."""
        return {
            'BB_Upper': {
                'type': 'line',
                'panel': 'main',  # Plot on price chart
                'color': 'red',
                'style': 'dashed'
            },
            'BB_Middle': {
                'type': 'line',
                'panel': 'main',
                'color': 'blue',
                'style': 'solid'
            },
            'BB_Lower': {
                'type': 'line',
                'panel': 'main',
                'color': 'green',
                'style': 'dashed'
            },
            'BB_PercentB': {
                'type': 'line',
                'panel': 'bottom',  # Separate panel
                'color': 'orange',
                'levels': [0, 0.5, 1],  # Reference levels
            }
        }