from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement the generate_signals method which takes
    price data and returns trading signals.
    """
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on price data.
        
        Args:
            data: DataFrame with OHLCV data, indexed by date
        
        Returns:
            DataFrame with original data plus signal columns:
            - 'Signal': 1 for buy, 0 for hold, -1 for sell
            - 'Position': Position changes (1 for enter, -1 for exit, 0 for hold)
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return the strategy parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, **kwargs) -> None:
        """Update strategy parameters."""
        self.parameters.update(kwargs)
    
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"