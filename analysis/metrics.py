import pandas as pd
import numpy as np
from typing import List, Dict, Any
from core.types import BacktestResult, Trade


class PerformanceAnalyzer:
    """
    Analyze backtest performance and generate metrics.
    """
    
    @staticmethod
    def print_results(result: BacktestResult) -> None:
        """Print formatted backtest results."""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        
        # Basic Performance
        print(f"Initial Capital:      ${result.initial_capital:,.2f}")
        print(f"Final Value:          ${result.final_value:,.2f}")
        print(f"Total Return:         {result.total_return:.2%}")
        print(f"Annualized Return:    {result.annualized_return:.2%}")
        
        # Risk Metrics
        print(f"Sharpe Ratio:         {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:         {result.max_drawdown:.2%}")
        
        # Trade Statistics
        print(f"Number of Trades:     {result.num_trades}")
        print(f"Win Rate:             {result.win_rate:.2%}")
        
        # Strategy Info
        print(f"Symbol:               {result.metadata.get('symbol', 'N/A')}")
        print(f"Strategy:             {result.metadata.get('strategy', 'N/A')}")
        print(f"Period:               {result.metadata.get('start_date')} to {result.metadata.get('end_date')}")
        print(f"Commission:           {result.metadata.get('commission', 0):.3%}")
        
        print("="*50)
    
    @staticmethod
    def print_trade_summary(result: BacktestResult, show_all: bool = False) -> None:
        """Print trade summary."""
        if not result.trades:
            print("No trades executed.")
            return
        
        print(f"\nTRADE SUMMARY ({len(result.trades)} trades)")
        print("-" * 60)
        
        trades_to_show = result.trades if show_all else result.trades[-10:]
        
        for i, trade in enumerate(trades_to_show, 1):
            print(f"{i:2d}. {trade.timestamp.date()} | "
                  f"{trade.side.value:4s} | "
                  f"{trade.quantity:8.2f} @ ${trade.price:7.2f} | "
                  f"${trade.value:9.2f}")
        
        if not show_all and len(result.trades) > 10:
            print(f"    ... and {len(result.trades) - 10} more trades")
    
    @staticmethod
    def calculate_monthly_returns(result: BacktestResult) -> pd.DataFrame:
        """Calculate monthly returns from portfolio values."""
        if not result.portfolio_values:
            return pd.DataFrame()
        
        portfolio_df = pd.DataFrame(result.portfolio_values)
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Resample to monthly and calculate returns
        monthly = portfolio_df['total_value'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        return monthly_returns
    
    @staticmethod
    def calculate_trade_analysis(result: BacktestResult) -> Dict[str, Any]:
        """Analyze individual trades for detailed statistics."""
        if not result.trades:
            return {}
        
        # Convert trades to DataFrame for analysis
        trades_data = []
        buy_price = None
        
        for trade in result.trades:
            if trade.side.value == "BUY":
                buy_price = trade.price
            elif trade.side.value == "SELL" and buy_price is not None:
                pnl = (trade.price - buy_price) * trade.quantity - trade.commission
                pnl_pct = (trade.price - buy_price) / buy_price
                
                trades_data.append({
                    'buy_price': buy_price,
                    'sell_price': trade.price,
                    'quantity': trade.quantity,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_days': (trade.timestamp - trades_data[-1]['buy_timestamp'] 
                                    if trades_data else pd.Timedelta(days=0)).days,
                    'buy_timestamp': trade.timestamp
                })
                buy_price = None
        
        if not trades_data:
            return {}
        
        trades_df = pd.DataFrame(trades_data)
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        analysis = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'avg_win_pct': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss_pct': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_trade_duration': trades_df['duration_days'].mean(),
            'profit_factor': (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum()) 
                            if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf)
        }
        
        return analysis
    
    @staticmethod
    def print_detailed_analysis(result: BacktestResult) -> None:
        """Print detailed trade analysis."""
        analysis = PerformanceAnalyzer.calculate_trade_analysis(result)
        
        if not analysis:
            print("No completed trade pairs for detailed analysis.")
            return
        
        print(f"\nDETAILED TRADE ANALYSIS")
        print("-" * 40)
        print(f"Total Trade Pairs:    {analysis['total_trades']}")
        print(f"Winning Trades:       {analysis['winning_trades']} ({analysis['win_rate']:.1%})")
        print(f"Losing Trades:        {analysis['losing_trades']}")
        print(f"Average Win:          ${analysis['avg_win']:.2f} ({analysis['avg_win_pct']:.2%})")
        print(f"Average Loss:         ${analysis['avg_loss']:.2f} ({analysis['avg_loss_pct']:.2%})")
        print(f"Largest Win:          ${analysis['largest_win']:.2f}")
        print(f"Largest Loss:         ${analysis['largest_loss']:.2f}")
        print(f"Profit Factor:        {analysis['profit_factor']:.2f}")
        print(f"Avg Trade Duration:   {analysis['avg_trade_duration']:.1f} days")