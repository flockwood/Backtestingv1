# 📈 Modern Backtesting System

A professional-grade backtesting framework for quantitative trading strategies with clean architecture, advanced analytics, and comprehensive risk management.

## ✨ Features

### Core Capabilities
- **Clean Architecture**: Modular design with separation of concerns
- **Multiple Data Sources**: Support for Alpaca Markets API and CSV files
- **Flexible Strategy Framework**: Easy-to-extend base classes for custom strategies
- **Realistic Transaction Costs**: Comprehensive cost modeling including slippage, spread, and commissions
- **Portfolio Management**: Sophisticated position tracking and cash management

### Advanced Analytics
- **Risk Metrics**: Sortino ratio, Calmar ratio, VaR, CVaR, Ulcer Index, Omega ratio
- **Rolling Analysis**: Time-varying performance metrics and volatility
- **Trade Analytics**: Win/loss streaks, holding periods, MAE/MFE analysis
- **Performance Reporting**: HTML tearsheets with charts and comprehensive metrics

### Database Integration
- **PostgreSQL Storage**: Persist all backtest results for historical analysis
- **Strategy Comparison**: Compare performance across different strategies
- **Parameter Optimization**: Store and analyze parameter sweep results
- **Query Interface**: Built-in tools for analyzing stored results

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/backtesting-system.git
cd backtesting-system

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced analytics dependencies
pip install scipy matplotlib openpyxl
```

### Basic Usage

```python
from data import AlpacaDataLoader
from strategies import SMAStrategy
from core import BacktestEngine
from analysis import PerformanceAnalyzer

# Initialize components
data_loader = AlpacaDataLoader()
strategy = SMAStrategy(short_window=10, long_window=20)
engine = BacktestEngine(initial_capital=10000)

# Run backtest
result = engine.run_backtest(
    data_loader=data_loader,
    strategy=strategy,
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Analyze results
PerformanceAnalyzer.print_results(result)
```

## 📊 Advanced Features

### Transaction Cost Modeling

```python
from core.costs import SimpleCostModel

# Create realistic cost model
cost_model = SimpleCostModel(
    fixed_fee=0,
    percentage_fee=0.001,  # 0.1% commission
    base_slippage_bps=5,    # 5 basis points slippage
    market_impact_coefficient=10,
    bid_ask_spread_bps=10
)

engine = BacktestEngine(
    initial_capital=10000,
    cost_model=cost_model
)
```

### Comprehensive Risk Analysis

```python
from analysis import RiskMetrics, TradeAnalytics, PerformanceReport

# Advanced risk metrics
risk_metrics = RiskMetrics(result)
sortino = risk_metrics.sortino_ratio()
var_95 = risk_metrics.value_at_risk(0.95)
all_metrics = risk_metrics.calculate_all_metrics()

# Trade analytics
trade_analyzer = TradeAnalytics(result)
win_streaks = trade_analyzer.get_win_loss_streaks()
risk_reward = trade_analyzer.risk_reward_analysis()

# Generate comprehensive report
report = PerformanceReport(result)
report.create_tearsheet('performance_report.html')
```

### Database Storage

```python
from database import DatabaseManager, BacktestRepository

# Initialize database
db_manager = DatabaseManager()
repository = BacktestRepository(db_manager)

# Create engine with database support
engine = BacktestEngine(
    initial_capital=10000,
    save_to_db=True,
    db_manager=db_manager,
    repository=repository
)

# Results automatically saved to PostgreSQL
result = engine.run_backtest(...)

# Query stored results
best_strategies = repository.get_best_strategies(symbol='AAPL')
```

## 🏗️ Project Structure

```
backtesting-system/
├── core/                   # Core backtesting engine
│   ├── engine.py          # Main backtesting orchestrator
│   ├── portfolio.py       # Portfolio management
│   ├── costs.py           # Transaction cost models
│   └── types.py           # Data types and structures
│
├── strategies/            # Trading strategies
│   ├── base.py           # Abstract strategy interface
│   └── sma_crossover.py  # SMA crossover implementation
│
├── data/                  # Data loading and management
│   ├── base.py           # Abstract data loader
│   ├── alpaca_loader.py  # Alpaca Markets integration
│   └── csv_loader.py     # CSV file support
│
├── analysis/              # Performance analytics
│   ├── metrics.py         # Basic performance metrics
│   ├── risk_metrics.py    # Advanced risk analysis
│   ├── rolling_metrics.py # Rolling window analysis
│   ├── trade_analytics.py # Trade-level analysis
│   └── performance_report.py # Report generation
│
├── database/              # PostgreSQL integration
│   ├── schema.sql        # Database schema
│   ├── models.py         # Data models
│   ├── repository.py     # Data access layer
│   └── db_manager.py     # Connection management
│
└── tests/                # Test suite
    ├── test_advanced_metrics.py
    └── test_costs.py
```

## 📈 Supported Strategies

- **SMA Crossover**: Classic moving average crossover strategy
- **Custom Strategies**: Extend `BaseStrategy` to implement your own

Example custom strategy:
```python
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Implement your logic here
        pass
```

## 🔧 Configuration

### Environment Variables

```bash
# Alpaca API (required for market data)
export ALPACA_API_KEY=your_api_key
export ALPACA_SECRET_KEY=your_secret_key

# Database (optional)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=backtesting_db
export DB_USER=postgres
export DB_PASSWORD=your_password
```

### Database Setup

```bash
# Initialize PostgreSQL database
python database/init_db.py

# Query stored results
python database/query_results.py
```

## 📊 Performance Metrics

The system calculates a comprehensive set of metrics:

### Basic Metrics
- Total & Annualized Returns
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Number of Trades

### Advanced Risk Metrics
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return vs maximum drawdown
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Ulcer Index**: Drawdown pain measurement
- **Omega Ratio**: Probability-weighted gains vs losses

### Trade Analytics
- Profit Factor
- Risk/Reward Ratio
- Win/Loss Streaks
- Average Holding Period
- Kelly Criterion Position Sizing

## 🧪 Testing

```bash
# Run basic tests
python test_metrics_simple.py

# Test advanced metrics
python test_advanced_metrics.py

# Test transaction costs
python test_costs.py

# Run full test suite
pytest tests/
```

## 📚 Examples

### Run Multiple Strategies
```bash
# Basic backtest
python main.py

# With database storage
python main_with_db.py

# Advanced metrics demo
python demo_advanced_metrics.py
```

### Parameter Optimization
```python
# Test multiple parameter combinations
parameter_sets = [(5, 15), (10, 20), (20, 50)]

for short, long in parameter_sets:
    strategy = SMAStrategy(short_window=short, long_window=long)
    result = engine.run_backtest(...)
    print(f"SMA({short}/{long}): Return={result.total_return:.2%}")
```

## 📖 Documentation

### Key Components

**BacktestEngine**: Orchestrates the entire backtesting process
- Manages portfolio state
- Executes trades with realistic costs
- Tracks performance metrics

**Portfolio**: Manages positions and cash
- Tracks open positions
- Calculates portfolio value
- Records transaction history

**CostModel**: Models realistic trading costs
- Commission fees
- Market impact & slippage
- Bid-ask spread

**PerformanceAnalyzer**: Comprehensive analysis tools
- Statistical metrics
- Risk analysis
- Trade analytics

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional trading strategies
- More data source integrations
- Enhanced visualization capabilities
- Machine learning integration
- Real-time trading connections

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Alpaca Markets for market data API
- PostgreSQL for robust data storage
- NumPy/Pandas for numerical computing
- Matplotlib for visualization capabilities

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always perform due diligence and consider consulting with financial professionals before making investment decisions.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---
