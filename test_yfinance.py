# test_yfinance.py
import yfinance as yf

# Test basic download
print("Testing Yahoo Finance connection...")
aapl = yf.Ticker("AAPL")
info = aapl.info
print(f"Company: {info.get('longName', 'N/A')}")
print(f"Current Price: ${info.get('currentPrice', 'N/A')}")

# Try downloading recent data
print("\nTrying to download recent data...")
data = yf.download("AAPL", period="1mo", progress=True)
print(f"Downloaded {len(data)} days of recent data")