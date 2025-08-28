import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import os

try:
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
except ImportError:
    raise ImportError("Please create config.py with your Alpaca API credentials")

class AlpacaDataLoader:
    def __init__(self):
        self.client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        self.data_dir = 'data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def download_stock_data(self, symbol: str, start_date: str, end_date: str):
        print(f"Downloading {symbol} from {start_date} to {end_date}...")
        
        # Parse dates and ensure they're not in the future
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        today = datetime.now()
        
        if end > today:
            print(f"End date {end_date} is in the future, using today's date")
            end = today
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        try:
            bars = self.client.get_stock_bars(request)
            
            if symbol not in bars.data or not bars.data[symbol]:
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
            
            print(f"Downloaded {len(df)} bars")
            
            # Save to CSV
            filepath = os.path.join(self.data_dir, f"{symbol}_alpaca.csv")
            df.to_csv(filepath)
            print(f"Saved to {filepath}")
            
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()

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