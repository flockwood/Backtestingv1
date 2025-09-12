import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import os
import logging
import requests
from alpaca.common.exceptions import APIError

try:
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
except ImportError:
    raise ImportError("Please create config.py with your Alpaca API credentials")

class AlpacaDataLoader:
    def __init__(self):
        # Setup persistent logging for 403 errors
        self.logger = self._setup_logger()
        
        self.logger.info(f"Initializing AlpacaDataLoader with API key: {ALPACA_API_KEY[:10]}...")
        
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
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            # File handler for persistent logging
            os.makedirs('test', exist_ok=True)
            file_handler = logging.FileHandler('test/alpaca_data_loader.log')
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Detailed formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def download_stock_data(self, symbol: str, start_date: str, end_date: str):
        self.logger.info(f"Starting download: {symbol} from {start_date} to {end_date}")
        print(f"Downloading {symbol} from {start_date} to {end_date}...")
        
        # Parse dates and ensure they're not in the future
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        today = datetime.now()
        
        if end > today:
            self.logger.info(f"End date {end_date} is in the future, adjusted to today")
            print(f"End date {end_date} is in the future, using today's date")
            end = today
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        self.logger.debug(f"API request parameters: symbol={symbol}, timeframe=Day, start={start}, end={end}")
        
        try:
            self.logger.info("Calling get_stock_bars API...")
            bars = self.client.get_stock_bars(request)
            
            if symbol not in bars.data or not bars.data[symbol]:
                self.logger.warning(f"No data returned for {symbol}")
                print(f"No data returned for {symbol}")
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
            df['Adj Close'] = df['Close']
            
            self.logger.info(f"✓ Successfully downloaded {len(df)} bars for {symbol}")
            print(f"Downloaded {len(df)} bars")
            
            # Save to CSV
            filepath = os.path.join(self.data_dir, f"{symbol}_alpaca.csv")
            df.to_csv(filepath)
            self.logger.info(f"✓ Data saved to {filepath}")
            print(f"Saved to {filepath}")
            
            return df
            
        except APIError as e:
            error_msg = str(e)
            self.logger.error(f"✗ Alpaca API Error: {error_msg}")
            
            # Check for 403 specifically
            if "403" in error_msg or "forbidden" in error_msg.lower():
                self.logger.error("=== 403 FORBIDDEN ERROR DETECTED ===")
                self.logger.error("This indicates an authentication/authorization issue")
                self.logger.error("Recommendations:")
                self.logger.error("1. Check API key validity in Alpaca dashboard")
                self.logger.error("2. Verify account is approved and active")
                self.logger.error("3. Ensure correct environment (paper vs live)")
                self.logger.error("4. Try regenerating API keys")
                self.logger.error("=== END 403 ERROR DETAILS ===")
                
                # Also test basic connectivity to help with diagnosis
                self._test_basic_auth()
            
            print(f"API Error: {e}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"✗ Unexpected error: {type(e).__name__}: {e}")
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def _test_basic_auth(self):
        """Test basic authentication when 403 error occurs"""
        self.logger.info("Testing basic authentication after 403 error...")
        
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        
        test_url = "https://paper-api.alpaca.markets/v2/account"
        
        try:
            response = requests.get(test_url, headers=headers, timeout=10)
            self.logger.info(f"Basic auth test: Status {response.status_code}")
            
            if response.status_code == 403:
                self.logger.error("✗ Basic auth test also returns 403 - credential issue confirmed")
                self.logger.error(f"Response: {response.text}")
            elif response.status_code == 200:
                self.logger.info("✓ Basic auth test successful - issue may be data-specific")
            else:
                self.logger.warning(f"? Basic auth test returned {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Basic auth test failed: {e}")

if __name__ == "__main__":
    loader = AlpacaDataLoader()
    
    # Use dates that are definitely in the past
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Testing with date range: {start_date} to {end_date}")
    
    data = loader.download_stock_data('AAPL', start_date, end_date)
    if len(data) > 0:
        print("\nSuccess! First 5 rows:")
        print(data.head())