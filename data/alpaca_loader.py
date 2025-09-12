import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

from .base import BaseDataLoader

try:
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
except ImportError:
    raise ImportError("Please create config.py with your Alpaca API credentials")


class AlpacaDataLoader(BaseDataLoader):
    """
    Data loader for Alpaca Markets API.
    """
    
    def __init__(self):
        super().__init__("Alpaca")
        self.logger = self._setup_logger()
        
        try:
            self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            self.logger.info("✓ StockHistoricalDataClient created successfully")
        except Exception as e:
            self.logger.error(f"✗ Failed to create StockHistoricalDataClient: {e}")
            raise
            
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def _setup_logger(self):
        """Setup dedicated logger for Alpaca API issues"""
        logger = logging.getLogger('alpaca_data_loader')
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical stock data from Alpaca Markets.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data indexed by date
        """
        self.logger.info(f"Loading {symbol} data from {start_date} to {end_date}")
        
        # Parse dates and ensure they're not in the future
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        today = datetime.now()
        
        if end > today:
            self.logger.info(f"End date {end_date} is in the future, adjusted to today")
            end = today
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        try:
            self.logger.info("Calling Alpaca API...")
            bars = self.client.get_stock_bars(request)
            
            if symbol not in bars.data or not bars.data[symbol]:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_list = []
            for bar in bars.data[symbol]:
                data_list.append({
                    'Date': bar.timestamp,
                    'Open': bar.open,
                    'High': bar.high,
                    'Low': bar.low,
                    'Close': bar.close,
                    'Volume': bar.volume
                })
            
            df = pd.DataFrame(data_list)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['Adj Close'] = df['Close']  # Alpaca doesn't provide adjusted close
            
            self.logger.info(f"✓ Successfully loaded {len(df)} bars for {symbol}")
            
            # Save to CSV for caching
            filepath = os.path.join(self.data_dir, f"{symbol}_alpaca.csv")
            df.to_csv(filepath)
            self.logger.info(f"✓ Data cached to {filepath}")
            
            return df
            
        except APIError as e:
            error_msg = str(e)
            self.logger.error(f"✗ Alpaca API Error: {error_msg}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"✗ Unexpected error: {type(e).__name__}: {e}")
            return pd.DataFrame()