import yfinance as yf
import pandas as pd
from datetime import datetime

def download_stock_data(symbol, start_date, end_date):
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date string (e.g., '2023-01-01')
        end_date: End date string (e.g., '2023-12-31')
    
    Returns:
        DataFrame with stock data
    """
    print(f"Downloading {symbol} data from {start_date} to {end_date}...")
    
    # Download data with auto_adjust explicitly set
    data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # If MultiIndex columns, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"Downloaded {len(data)} days of data")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    return data