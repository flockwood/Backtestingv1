import requests
import json
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Test different endpoints
endpoints = {
    "Paper Trading": "https://paper-api.alpaca.markets",
    "Live Trading": "https://api.alpaca.markets",
}

print("Testing Alpaca Authentication...\n")
print(f"API Key: {ALPACA_API_KEY[:10]}...")
print(f"Secret: {ALPACA_SECRET_KEY[:10]}...\n")

for name, base_url in endpoints.items():
    print(f"Testing {name} endpoint: {base_url}")
    
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    
    try:
        response = requests.get(f"{base_url}/v2/account", headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            account = response.json()
            print(f"✓ SUCCESS - Account Status: {account.get('status')}")
            print(f"  Buying Power: ${account.get('buying_power')}")
        else:
            print(f"✗ FAILED - Response: {response.text}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    print("-" * 50)

# Also test data endpoint
print("\nTesting Market Data Endpoint...")
data_url = "https://data.alpaca.markets/v2/stocks/AAPL/bars/latest"
headers = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

response = requests.get(data_url, headers=headers)
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    print("✓ Market data access works")
else:
    print(f"✗ Market data failed: {response.text}")