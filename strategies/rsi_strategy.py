"""
RSI (Relative Strength Index) Strategy

A momentum oscillator that measures the speed and magnitude of price changes.
Trades on overbought/oversold conditions for mean reversion.

Buy Signal: RSI < 30 (oversold)
Sell Signal: RSI > 70 (overbought)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    
    The RSI oscillates between 0 and 100:
    - RSI < 30: Oversold (potential buy)
    - RSI > 70: Overbought (potential sell)
    - RSI 30-70: Neutral
    """
    
    def __init__(self, 
                 period: int = 14,
                 oversold: float = 30,
                 overbought: float = 70,
                 use_signal_line: bool = False):
        """
        Initialize RSI strategy.
        
        Args:
            period: Lookback period for RSI calculation (typically 14)
            oversold: RSI level indicating oversold condition (typically 30)
            overbought: RSI level indicating overbought condition (typically 70)
            use_signal_line: If True, use 9-period EMA of RSI as signal line
        """
        parameters = {
            'period': period,
            'oversold': oversold,
            'overbought': overbought,
            'use_signal_line': use_signal_line
        }
        super().__init__("RSI Strategy", parameters)
        
    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate the Relative Strength Index.
        
        Args:
            prices: Series of prices (typically Close)
            period: RSI period
            
        Returns:
            Series of RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Alternative calculation for better accuracy (Wilder's smoothing)
        # First values use simple average
        avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RSI-based trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with original data plus signal columns
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying original data
        signals_data = data.copy()
        
        # Get strategy parameters
        period = self.parameters['period']
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']
        use_signal_line = self.parameters['use_signal_line']
        
        # Calculate RSI
        signals_data['RSI'] = self.calculate_rsi(signals_data['Close'], period)
        
        # Optional: Calculate signal line (9-period EMA of RSI)
        if use_signal_line:
            signals_data['RSI_Signal'] = signals_data['RSI'].ewm(span=9, adjust=False).mean()
        
        # Initialize signals
        signals_data['Signal'] = 0
        signals_data['Position'] = 0
        
        # Generate signals based on RSI levels
        for i in range(period, len(signals_data)):
            current_rsi = signals_data['RSI'].iloc[i]
            prev_rsi = signals_data['RSI'].iloc[i-1] if i > period else current_rsi
            
            # Skip if RSI is NaN
            if pd.isna(current_rsi):
                continue
            
            if use_signal_line and 'RSI_Signal' in signals_data.columns:
                signal_line = signals_data['RSI_Signal'].iloc[i]
                
                # Buy when RSI crosses above signal line in oversold zone
                if current_rsi < oversold and current_rsi > signal_line and prev_rsi <= signal_line:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = 1
                    
                # Sell when RSI crosses below signal line in overbought zone
                elif current_rsi > overbought and current_rsi < signal_line and prev_rsi >= signal_line:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = -1
            else:
                # Simple RSI strategy
                # Buy when RSI crosses above oversold level
                if prev_rsi <= oversold and current_rsi > oversold:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = 1
                    
                # Sell when RSI crosses below overbought level
                elif prev_rsi >= overbought and current_rsi < overbought:
                    signals_data.iloc[i, signals_data.columns.get_loc('Signal')] = -1
        
        # Generate position changes
        # Position is 1 when we should be long, 0 when we should be out
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
        
        return signals_data
    
    def get_description(self) -> str:
        """Get a human-readable description of the strategy."""
        period = self.parameters['period']
        oversold = self.parameters['oversold']
        overbought = self.parameters['overbought']
        
        return (f"RSI({period}) Mean Reversion Strategy: "
                f"Buy when RSI < {oversold} (oversold), "
                f"Sell when RSI > {overbought} (overbought)")
    
    def get_indicators(self) -> Dict[str, Any]:
        """Return the indicators used by this strategy for plotting."""
        return {
            'RSI': {
                'type': 'line',
                'panel': 'bottom',  # Plot in separate panel
                'color': 'purple',
                'levels': [self.parameters['oversold'], self.parameters['overbought']],
                'fill_between': (0, 100)  # Valid RSI range
            }
        }