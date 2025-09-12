from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Optional


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    All data loaders must implement the load_data method which retrieves
    historical price data for a given symbol and date range.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical price data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        
        Returns:
            DataFrame with OHLCV data indexed by date:
            - 'Open': Opening price
            - 'High': High price
            - 'Low': Low price
            - 'Close': Closing price
            - 'Volume': Trading volume
            - 'Adj Close': Adjusted closing price
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data has the required columns and format.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if data.empty:
            return False
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False
            
        if not isinstance(data.index, pd.DatetimeIndex):
            print("Index must be DatetimeIndex")
            return False
            
        return True
    
    def __str__(self) -> str:
        return f"{self.name} Data Loader"