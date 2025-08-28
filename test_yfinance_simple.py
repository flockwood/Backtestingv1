import yfinance as yf
import time

print("Testing Yahoo Finance connection with rate limit handling...\n")

# Method 1: Download with period instead of dates
print("Method 1: Downloading last month of AAPL data...")
try:
    data = yf.download("AAPL", period="1mo", interval="1d", progress=False)
    print(f"✓ Success! Downloaded {len(data)} days of data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Wait to avoid rate limit
print("\nWaiting 2 seconds to avoid rate limit...")
time.sleep(2)

# Method 2: Use Ticker.history() instead
print("\nMethod 2: Using Ticker.history()...")
try:
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1mo")
    print(f"✓ Success! Downloaded {len(data)} days of data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Save the working data for our backtester
if len(data) > 0:
    print("\nSaving data to use in backtester...")
    data.to_csv("data/AAPL_recent.csv")
    print("✓ Saved to data/AAPL_recent.csv")