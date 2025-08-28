import pandas as pd
import numpy as np

def run_backtest(data, initial_capital=10000):
    """
    Run a simple backtest
    
    Args:
        data: DataFrame with price data and signals
        initial_capital: Starting cash amount
    
    Returns:
        Dictionary with results
    """
    # Initialize tracking variables
    cash = initial_capital
    shares = 0
    trades = []
    portfolio_values = []
    
    print(f"\nStarting backtest with ${initial_capital:,.2f}")
    
    for index, row in data.iterrows():
        current_price = row['Close']
        signal = row['signal']
        date = index  # Date is in the index
        
        # Skip rows with NaN signal
        if pd.isna(signal):
            portfolio_value = cash + (shares * current_price)
            portfolio_values.append(portfolio_value)
            continue
        
        # Execute trades based on signals
        if signal == 1.0 and cash > 0:  # Buy signal and we have cash
            shares_to_buy = cash / current_price
            shares += shares_to_buy
            cash = 0
            trades.append({
                'date': date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares_to_buy,
                'cash_after': cash
            })
            print(f"{date.strftime('%Y-%m-%d')}: BUY {shares_to_buy:.2f} shares at ${current_price:.2f}")
            
        elif signal == -1.0 and shares > 0:  # Sell signal and we have shares
            cash_from_sale = shares * current_price
            cash += cash_from_sale
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'cash_after': cash
            })
            print(f"{date.strftime('%Y-%m-%d')}: SELL {shares:.2f} shares at ${current_price:.2f} for ${cash_from_sale:.2f}")
            shares = 0
        
        # Calculate portfolio value
        portfolio_value = cash + (shares * current_price)
        portfolio_values.append(portfolio_value)
    
    # Calculate final metrics
    final_value = cash + (shares * data.iloc[-1]['Close'])
    total_return = (final_value - initial_capital) / initial_capital * 100
    num_trades = len([t for t in trades if t['action'] == 'BUY'])
    
    results = {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'num_trades': num_trades,
        'trades': trades,
        'portfolio_values': portfolio_values
    }
    
    return results