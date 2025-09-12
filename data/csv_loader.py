import pandas as pd
import os
from .base import BaseDataLoader


class CSVDataLoader(BaseDataLoader):
    """
    Data loader for CSV files.
    """
    
    def __init__(self, data_directory: str = "data"):
        super().__init__("CSV")
        self.data_directory = data_directory
    
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical stock data from CSV file.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        # Try different CSV file naming conventions
        possible_files = [
            f"{symbol}.csv",
            f"{symbol}_alpaca.csv",
            f"{symbol}_data.csv",
            f"{symbol.lower()}.csv"
        ]
        
        filepath = None
        for filename in possible_files:
            test_path = os.path.join(self.data_directory, filename)
            if os.path.exists(test_path):
                filepath = test_path
                break
        
        if filepath is None:
            print(f"No CSV file found for {symbol} in {self.data_directory}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Missing required columns in CSV: {missing_columns}")
                return pd.DataFrame()
            
            # Add Adj Close if it doesn't exist
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # Filter by date range (handle timezone-aware dates)
            start = pd.to_datetime(start_date, utc=True)
            end = pd.to_datetime(end_date, utc=True)
            
            # Make sure index is timezone-aware if needed
            if df.index.tz is not None:
                start = start.tz_convert(df.index.tz)
                end = end.tz_convert(df.index.tz)
            elif start.tz is not None:
                start = start.tz_localize(None)
                end = end.tz_localize(None)
            
            df = df[(df.index >= start) & (df.index <= end)]
            
            print(f"✓ Loaded {len(df)} rows from {filepath}")
            return df
            
        except Exception as e:
            print(f"Error loading CSV file {filepath}: {e}")
            return pd.DataFrame()