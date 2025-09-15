# Database Integration for Backtesting System

This module provides PostgreSQL database integration for storing and analyzing backtest results.

## Features

- **Complete Data Persistence**: Store strategies, backtest runs, trades, and daily performance
- **Performance Analytics**: Track metrics like Sharpe ratio, returns, drawdowns across runs
- **Strategy Comparison**: Compare different strategies and parameter configurations
- **Historical Analysis**: Query past results and identify best-performing strategies
- **Optimization Storage**: Store parameter sweep results for systematic analysis

## Architecture

### Database Schema

The system uses a normalized schema with the following tables:

1. **strategies** - Strategy configurations and parameters
2. **backtest_runs** - Individual backtest executions
3. **backtest_results** - Summary metrics for each run
4. **trades** - Detailed trade records
5. **daily_performance** - Daily portfolio values
6. **optimization_results** - Parameter optimization results

### Components

- **DatabaseConfig** - Configuration management
- **DatabaseManager** - Connection pooling and query execution
- **BacktestRepository** - Data access layer with typed models
- **Models** - Type-safe data models for all entities

## Setup

### 1. Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start PostgreSQL
sudo service postgresql start  # Linux
brew services start postgresql  # macOS
```

### 2. Install Python Dependencies

```bash
pip install psycopg2-binary
# or
pip install -r requirements.txt
```

### 3. Configure Database Connection

Set environment variables (optional):
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=backtesting_db
export DB_USER=postgres
export DB_PASSWORD=yourpassword
```

Or use defaults (localhost:5432, postgres user, no password).

### 4. Initialize Database

```bash
python database/init_db.py
```

This will:
- Create the database if it doesn't exist
- Execute the schema to create tables
- Set up indexes and views

## Usage

### Running Backtest with Database Storage

```python
from database import DatabaseManager, BacktestRepository
from core import BacktestEngine

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

# Run backtest - results automatically saved
result = engine.run_backtest(...)
```

### Querying Results

```python
from database import DatabaseManager, BacktestRepository

db_manager = DatabaseManager()
repository = BacktestRepository(db_manager)

# Get best strategies
best = repository.get_best_strategies(symbol='AAPL', limit=10)

# Get recent runs
runs = repository.get_backtest_runs(limit=20)

# Get specific strategy statistics
stats = repository.get_strategy_statistics(strategy_id=1)
```

### Running Examples

```bash
# Run backtest with database storage
python main_with_db.py

# Query stored results
python database/query_results.py
```

## Query Examples

### Best Performing Strategies
```sql
SELECT * FROM best_strategies
WHERE symbol = 'AAPL'
ORDER BY sharpe_ratio DESC
LIMIT 10;
```

### Strategy Performance Over Time
```sql
SELECT
    date_trunc('day', created_at) as day,
    AVG(total_return) as avg_return,
    COUNT(*) as num_runs
FROM backtest_runs r
JOIN backtest_results res ON r.id = res.run_id
GROUP BY day
ORDER BY day;
```

### Trade Analysis
```sql
SELECT
    action,
    COUNT(*) as count,
    AVG(trade_value) as avg_value
FROM trades
WHERE run_id = 123
GROUP BY action;
```

## Database Maintenance

### Backup Database
```bash
pg_dump backtesting_db > backup.sql
```

### Restore Database
```bash
psql backtesting_db < backup.sql
```

### Clean Old Data
```sql
-- Delete runs older than 30 days
DELETE FROM backtest_runs
WHERE created_at < NOW() - INTERVAL '30 days';
```

## Performance Considerations

1. **Indexes**: Schema includes indexes on commonly queried fields
2. **Connection Pooling**: DatabaseManager uses connection pooling
3. **Batch Inserts**: Trades and daily performance use batch inserts
4. **JSONB Fields**: Flexible storage for parameters and positions

## Troubleshooting

### Connection Issues
- Verify PostgreSQL is running: `sudo service postgresql status`
- Check credentials and permissions
- Ensure database exists: `psql -l`

### Permission Errors
```sql
-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE backtesting_db TO your_user;
```

### Schema Updates
To update schema after changes:
```bash
python database/init_db.py  # This will recreate tables
```

## Advanced Features

### Custom Queries
The repository supports raw SQL queries:
```python
results = db_manager.execute_query(
    "SELECT * FROM strategies WHERE parameters->>'short_window' = %s",
    ('10',)
)
```

### Transaction Management
```python
with db_manager.get_connection() as conn:
    # Multiple operations in transaction
    cursor = conn.cursor()
    cursor.execute(...)
    cursor.execute(...)
    # Auto-commit on success, rollback on error
```

## Future Enhancements

- [ ] Add Redis caching layer
- [ ] Implement data archival strategy
- [ ] Add real-time performance monitoring
- [ ] Create web dashboard for results
- [ ] Add export to CSV/Excel functionality