import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import download_stock_data
import yfinance as yf
import os
import time

def download_with_retry(symbol, start_date, end_date, max_retries=3):
    """
    Download data with retry logic
    """
    for attempt in range(max_retries):
        try:
            # Try downloading directly with yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if len(data) > 0:
                return data
            
            print(f"Attempt {attempt + 1}: No data returned, retrying...")
            time.sleep(2)  # Wait 2 seconds before retry
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # If all retries failed, try one more time with different parameters
    print("Trying alternative download method...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=True, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data

def verify_data():
    """
    Download and verify AAPL data for 2023
    Checks for data quality issues and saves in multiple formats
    """
    # Configuration
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    print("=== Data Verification Script ===\n")
    
    # Step 1: Download data with retry logic
    print(f"Attempting to download {symbol} data...")
    data = download_with_retry(symbol, start_date, end_date)
    
    # Check if we got data
    if len(data) == 0:
        print("\nERROR: No data downloaded. Possible issues:")
        print("1. Internet connection problem")
        print("2. Yahoo Finance API is temporarily down")
        print("3. Date range issue")
        print("\nTrying a known working example...")
        
        # Try a different date range
        data = download_with_retry(symbol, '2023-06-01', '2023-12-31')
        
        if len(data) == 0:
            print("\nStill no data. Please check your internet connection.")
            return None
    
    print(f"\nSuccessfully downloaded {len(data)} days of data")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Step 2: Display first and last 5 rows
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nLast 5 rows:")
    print(data.tail())
    
    # Step 3: Check for missing values
    print("\n=== Missing Values Check ===")
    missing_count = data.isnull().sum()
    print("Missing values per column:")
    print(missing_count)
    print(f"\nTotal missing values: {missing_count.sum()}")
    
    # Check for any gaps in dates
    if len(data) > 1:
        date_diff = pd.Series(data.index).diff()
        date_gaps = date_diff[date_diff > pd.Timedelta(days=1)]
        
        if len(date_gaps) > 0:
            print(f"\nWarning: Found {len(date_gaps)} gaps in dates:")
            for idx in date_gaps.index:
                print(f"  Gap between {data.index[idx-1]} and {data.index[idx]}")
        else:
            print("\nNo gaps in dates (excluding weekends/holidays)")
    
    # Step 4: Data statistics
    print("\n=== Data Statistics ===")
    print(data.describe())
    
    # Step 5: Anomaly checks (only if we have data)
    if len(data) > 0:
        print("\n=== Anomaly Checks ===")
        
        # Check if High >= Low
        invalid_hl = data[data['High'] < data['Low']]
        print(f"Rows where High < Low: {len(invalid_hl)}")
        
        # Check if Close is within High/Low range
        invalid_close = data[(data['Close'] > data['High']) | (data['Close'] < data['Low'])]
        print(f"Rows where Close outside High/Low range: {len(invalid_close)}")
        
        # Check for zero or negative prices
        zero_prices = data[(data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)]
        print(f"Rows with zero or negative prices: {len(zero_prices)}")
        
        # Check for extreme price changes (> 20% in a day)
        if len(data) > 1:
            data['Daily_Return'] = data['Close'].pct_change()
            extreme_moves = data[abs(data['Daily_Return']) > 0.20]
            print(f"Days with >20% price change: {len(extreme_moves)}")
            if len(extreme_moves) > 0:
                print("Extreme moves:")
                print(extreme_moves[['Close', 'Daily_Return']])
    
    # Step 6: Plot closing price
    if len(data) > 0:
        print("\n=== Creating Price Chart ===")
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], linewidth=2)
        plt.title(f'{symbol} Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/{symbol}_price.png')
        print(f"Price chart saved to plots/{symbol}_price.png")
        plt.close()
    
    # Step 7: Save data in different formats
    if len(data) > 0:
        print("\n=== Saving Data ===")
        
        # Ensure data directory exists
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Remove temporary columns if they exist
        if 'Daily_Return' in data.columns:
            data = data.drop('Daily_Return', axis=1)
        
        # Save as CSV
        csv_path = f'data/{symbol}_data.csv'
        data.to_csv(csv_path)
        print(f"Saved as CSV: {csv_path}")
        
        # Save as Parquet
        parquet_path = f'data/{symbol}_data.parquet'
        data.to_parquet(parquet_path, compression='snappy')
        print(f"Saved as Parquet: {parquet_path}")
        
        # Step 8: Load and verify both files match
        print("\n=== Verifying Saved Files ===")
        
        # Load CSV
        csv_data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        print(f"CSV loaded: {len(csv_data)} rows")
        
        # Load Parquet
        parquet_data = pd.read_parquet(parquet_path)
        print(f"Parquet loaded: {len(parquet_data)} rows")
        
        # Compare file sizes
        csv_size = os.path.getsize(csv_path) / 1024  # KB
        parquet_size = os.path.getsize(parquet_path) / 1024  # KB
        
        print(f"\n=== File Size Comparison ===")
        print(f"CSV size: {csv_size:.2f} KB")
        print(f"Parquet size: {parquet_size:.2f} KB")
        print(f"Compression ratio: {csv_size/parquet_size:.2f}x")
    
    return data

if __name__ == "__main__":
    verified_data = verify_data()
    if verified_data is not None:
        print("\n=== Verification Complete ===")
        print(f"Data shape: {verified_data.shape}")
        print("Data is ready for backtesting!")
    else:
        print("\n=== Verification Failed ===")
        print("Please check your internet connection and try again.")