import requests
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

def test_alpaca_auth():
    """Test Alpaca authentication"""
    
    # Test account endpoint
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
    }
    
    print("Testing Alpaca authentication...")
    print(f"API Key: {ALPACA_API_KEY[:10]}...")
    print(f"Base URL: {ALPACA_BASE_URL}")
    
    # Test account info
    response = requests.get(f"{ALPACA_BASE_URL}/account", headers=headers)
    
    if response.status_code == 200:
        print("\n✓ Authentication successful!")
        account = response.json()
        print(f"Account Status: {account.get('status')}")
        print(f"Buying Power: ${account.get('buying_power')}")
    else:
        print(f"\n✗ Authentication failed: {response.status_code}")
        print(f"Response: {response.text}")
    
    # Test market data endpoint (different base URL)
    print("\n\nTesting market data access...")
    data_url = "https://data.alpaca.markets/v2/stocks/AAPL/bars"
    params = {
        'start': '2024-01-01T00:00:00Z',
        'end': '2024-01-31T00:00:00Z',
        'timeframe': '1Day',
        'limit': 10
    }
    
    response = requests.get(data_url, headers=headers, params=params)
    
    if response.status_code == 200:
        print("✓ Market data access successful!")
        data = response.json()
        print(f"Retrieved {len(data.get('bars', []))} bars")
    else:
        print(f"✗ Market data access failed: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    test_alpaca_auth()