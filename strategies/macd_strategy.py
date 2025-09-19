"""
MACD (Moving Average Convergence Divergence) Strategy

A momentum indicator that shows the relationship between two moving averages.
Trades on MACD line crossovers with the signal line and zero-line crosses.

Buy Signal: MACD crosses above signal line (bullish crossover)
Sell Signal: MACD crosses below signal line (bearish crossover)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseStrategy


class MACDStrategy(BaseStrategy):
    """
    MACD-based momentum trading strategy.
    
    MACD = 12-day EMA - 26-day EMA
    Signal Line = 9-day EMA of MACD
    Histogram = MACD - Signal Line
    """
    
    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 use_histogram: bool = False):
        """
        Initialize MACD strategy.
        
        Args:
            fast_period: Fast EMA period (typically 12)
            slow_period: Slow EMA period (typically 26)
            signal_period: Signal line EMA period (typically 9)
            use_histogram: If True, use histogram zero crosses for signals
        """
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'use_histogram': use_histogram
        }
        super().__init__("MACD Strategy", parameters)
    
    def calculate_macd(self, prices: pd.Series) -> tuple:
        """
        Calculate MACD, signal line, and histogram.
        
        Args:
            prices: Series of prices (typically Close)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        signal_period = self.parameters['signal_period']
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate MACD trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus signal columns
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying original data
        signals_data = data.copy()
        
        # Calculate MACD indicators
        macd, signal, histogram = self.calculate_macd(signals_data['Close'])
        
        # Add indicators to dataframe
        signals_data['MACD'] = macd
        signals_data['MACD_Signal'] = signal
        signals_data['MACD_Histogram'] = histogram
        
        # Initialize signals
        signals_data['Signal'] = 0
        signals_data['Position'] = 0
        
        # Minimum periods needed for calculation
        min_period = self.parameters['slow_period'] + self.parameters['signal_period']
        
        if self.parameters['use_histogram']:
            # Strategy based on histogram zero crosses
            for i in range(min_period, len(signals_data)):
                current_hist = signals_data['MACD_Histogram'].iloc[i]
                prev_hist = signals_data['MACD_Histogram'].iloc[i-1]
                
                # Skip if values are NaN
                if pd.isna(current_hist) or pd.isna(prev_hist):
                    continue
                
                # Buy when histogram crosses above zero (momentum turning positive)
                if prev_hist <= 0 and current_hist > 0:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = 1
                    
                # Sell when histogram crosses below zero (momentum turning negative)
                elif prev_hist >= 0 and current_hist < 0:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = -1
        else:
            # Traditional MACD crossover strategy
            for i in range(min_period, len(signals_data)):
                current_macd = signals_data['MACD'].iloc[i]
                current_signal = signals_data['MACD_Signal'].iloc[i]
                prev_macd = signals_data['MACD'].iloc[i-1]
                prev_signal = signals_data['MACD_Signal'].iloc[i-1]
                
                # Skip if values are NaN
                if pd.isna(current_macd) or pd.isna(current_signal):
                    continue
                
                # Bullish crossover: MACD crosses above signal line
                if prev_macd <= prev_signal and current_macd > current_signal:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = 1
                    
                # Bearish crossover: MACD crosses below signal line
                elif prev_macd >= prev_signal and current_macd < current_signal:
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
        
        # Add divergence detection (advanced feature)
        signals_data['MACD_Divergence'] = self.detect_divergence(
            signals_data['Close'], 
            signals_data['MACD_Histogram']
        )
        
        return signals_data
    
    def detect_divergence(self, prices: pd.Series, histogram: pd.Series, lookback: int = 20) -> pd.Series:
        """
        Detect bullish and bearish divergences.
        
        Args:
            prices: Price series
            histogram: MACD histogram
            lookback: Period to look for divergences
            
        Returns:
            Series with divergence signals (1 = bullish, -1 = bearish, 0 = none)
        """
        divergence = pd.Series(0, index=prices.index)
        
        if len(prices) < lookback * 2:
            return divergence
        
        for i in range(lookback, len(prices) - lookback):
            # Get local highs and lows
            price_window = prices.iloc[i-lookback:i+lookback+1]
            hist_window = histogram.iloc[i-lookback:i+lookback+1]
            
            # Skip if NaN values
            if hist_window.isna().any():
                continue
            
            # Check for bullish divergence (price makes lower low, MACD makes higher low)
            if prices.iloc[i] == price_window.min():
                # This is a price low
                prev_lows_idx = price_window.iloc[:-lookback].idxmin()
                if prev_lows_idx != prices.index[i]:
                    prev_price_low = prices.loc[prev_lows_idx]
                    prev_hist_low = histogram.loc[prev_lows_idx]
                    
                    if prices.iloc[i] < prev_price_low and histogram.iloc[i] > prev_hist_low:
                        divergence.iloc[i] = 1  # Bullish divergence
            
            # Check for bearish divergence (price makes higher high, MACD makes lower high)
            if prices.iloc[i] == price_window.max():
                # This is a price high
                prev_highs_idx = price_window.iloc[:-lookback].idxmax()
                if prev_highs_idx != prices.index[i]:
                    prev_price_high = prices.loc[prev_highs_idx]
                    prev_hist_high = histogram.loc[prev_highs_idx]
                    
                    if prices.iloc[i] > prev_price_high and histogram.iloc[i] < prev_hist_high:
                        divergence.iloc[i] = -1  # Bearish divergence
        
        return divergence
    
    def get_description(self) -> str:
        """Get a human-readable description of the strategy."""
        fast = self.parameters['fast_period']
        slow = self.parameters['slow_period']
        signal = self.parameters['signal_period']
        use_hist = self.parameters['use_histogram']
        
        if use_hist:
            return (f"MACD({fast},{slow},{signal}) Histogram Strategy: "
                    f"Buy when histogram > 0, Sell when histogram < 0")
        else:
            return (f"MACD({fast},{slow},{signal}) Crossover Strategy: "
                    f"Buy on bullish crossover, Sell on bearish crossover")
    
    def get_indicators(self) -> Dict[str, Any]:
        """Return the indicators used by this strategy for plotting."""
        return {
            'MACD': {
                'type': 'line',
                'panel': 'bottom',  # Separate panel below price
                'color': 'blue',
                'style': 'solid'
            },
            'MACD_Signal': {
                'type': 'line',
                'panel': 'bottom',
                'color': 'red',
                'style': 'dashed'
            },
            'MACD_Histogram': {
                'type': 'histogram',
                'panel': 'bottom',
                'color': 'gray',
                'alpha': 0.3
            },
            'MACD_Divergence': {
                'type': 'marker',
                'panel': 'main',
                'color_bullish': 'green',
                'color_bearish': 'red',
                'marker': '^'  # Triangle up for bullish, down for bearish
            }
        }