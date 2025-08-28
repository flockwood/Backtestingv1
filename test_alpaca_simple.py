from alpaca.trading.client import TradingClient
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

# Test trading client connection
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

try:
    # Get account information
    account = trading_client.get_account()
    print("Connection successful!")
    print(f"Account status: {account.status}")
    print(f"Buying power: ${account.buying_power}")
except Exception as e:
    print(f"Connection failed: {e}")