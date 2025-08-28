import pandas as pd
import os
from datetime import datetime
from typing import Optional

def download_stock_data(symbol: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None, source: str = 'csv',
                       data_dir: str = 'data'):
    """
    Load stock data from CSV file or Alpaca API
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        source: Data source ('csv' or 'alpaca')
        data_dir: Directory for CSV files
    
    Returns:
        DataFrame with stock data
    """
    if source == 'alpaca':
        try:
            from alpaca_data_loader import AlpacaDataLoader
            loader = AlpacaDataLoader()
            
            # Use provided dates or default to last year
            if not start_date:
                from datetime import timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            return loader.download_stock_data(symbol, start_date, end_date, save_csv=True)
            
        except ImportError:
            print("Alpaca not available, falling back to CSV")
            source = 'csv'
    
    if source == 'csv':
        # Original CSV loading logic
        filename = f"{symbol}_data.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Try alternative filenames
        if not os.path.exists(filepath):
            alt_filepath = os.path.join(data_dir, f"{symbol}_alpaca.csv")
            if os.path.exists(alt_filepath):
                filepath = alt_filepath
            else:
                alt_filepath = os.path.join(data_dir, f"{symbol}.csv")
                if os.path.exists(alt_filepath):
                    filepath = alt_filepath
                else:
                    raise FileNotFoundError(f"No data file found for {symbol}")
        
        print(f"Loading {symbol} data from {filepath}...")
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        
        print(f"Loaded {len(data)} days of data")
        if len(data) > 0:
            print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        return data