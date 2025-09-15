-- Backtesting Database Schema
-- Stores strategy configurations, backtest results, trades, and performance metrics

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS daily_performance CASCADE;
DROP TABLE IF EXISTS trades CASCADE;
DROP TABLE IF EXISTS backtest_results CASCADE;
DROP TABLE IF EXISTS backtest_runs CASCADE;
DROP TABLE IF EXISTS strategies CASCADE;

-- 1. Strategies table - Store strategy configurations
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Backtest runs - Track each backtest execution
CREATE TABLE backtest_runs (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    symbol VARCHAR(10) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2) NOT NULL,
    data_source VARCHAR(50),
    cost_model JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_duration_seconds FLOAT,
    status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT,
    UNIQUE(strategy_id, symbol, start_date, end_date, initial_capital)
);

-- 3. Backtest results - Summary metrics for each run
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    final_value DECIMAL(15,2) NOT NULL,
    total_return DECIMAL(10,4) NOT NULL,
    annualized_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    win_rate DECIMAL(10,4),
    num_trades INTEGER NOT NULL,
    num_winning_trades INTEGER,
    num_losing_trades INTEGER,
    total_commission DECIMAL(15,2),
    total_slippage DECIMAL(15,2),
    total_spread_cost DECIMAL(15,2),
    total_transaction_costs DECIMAL(15,2),
    avg_trade_duration_hours FLOAT,
    profit_factor DECIMAL(10,4),
    avg_win DECIMAL(15,2),
    avg_loss DECIMAL(15,2),
    largest_win DECIMAL(15,2),
    largest_loss DECIMAL(15,2),
    UNIQUE(run_id)
);

-- 4. Trades - Individual trade details
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    trade_date TIMESTAMP NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(15,4) NOT NULL,
    price DECIMAL(15,4) NOT NULL,
    commission DECIMAL(15,4) DEFAULT 0,
    slippage DECIMAL(15,4) DEFAULT 0,
    spread_cost DECIMAL(15,4) DEFAULT 0,
    total_cost DECIMAL(15,4) DEFAULT 0,
    trade_value DECIMAL(15,2),
    pnl DECIMAL(15,4),
    pnl_percent DECIMAL(10,4),
    position_size DECIMAL(15,4),
    cash_after DECIMAL(15,2)
);

-- 5. Daily performance - Track daily portfolio values
CREATE TABLE daily_performance (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES backtest_runs(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    portfolio_value DECIMAL(15,2) NOT NULL,
    cash DECIMAL(15,2),
    positions_value DECIMAL(15,2),
    daily_return DECIMAL(10,6),
    cumulative_return DECIMAL(10,6),
    drawdown DECIMAL(10,6),
    positions JSONB,
    num_positions INTEGER DEFAULT 0,
    UNIQUE(run_id, date)
);

-- 6. Strategy optimization results (for parameter sweeps)
CREATE TABLE optimization_results (
    id SERIAL PRIMARY KEY,
    strategy_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    parameters JSONB NOT NULL,
    total_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    num_trades INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX idx_backtest_runs_symbol ON backtest_runs(symbol);
CREATE INDEX idx_backtest_runs_dates ON backtest_runs(start_date, end_date);
CREATE INDEX idx_backtest_runs_strategy ON backtest_runs(strategy_id);
CREATE INDEX idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX idx_trades_run_date ON trades(run_id, trade_date);
CREATE INDEX idx_trades_action ON trades(action);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_daily_performance_run_date ON daily_performance(run_id, date);
CREATE INDEX idx_results_return ON backtest_results(total_return DESC);
CREATE INDEX idx_results_sharpe ON backtest_results(sharpe_ratio DESC);
CREATE INDEX idx_results_drawdown ON backtest_results(max_drawdown);
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_optimization_params ON optimization_results USING GIN (parameters);

-- Views for analysis
CREATE OR REPLACE VIEW best_strategies AS
SELECT
    s.name,
    s.strategy_type,
    s.parameters,
    br.symbol,
    br.start_date,
    br.end_date,
    res.total_return,
    res.sharpe_ratio,
    res.max_drawdown,
    res.num_trades,
    res.win_rate
FROM strategies s
JOIN backtest_runs br ON s.id = br.strategy_id
JOIN backtest_results res ON br.id = res.run_id
WHERE br.status = 'completed'
ORDER BY res.sharpe_ratio DESC NULLS LAST;

CREATE OR REPLACE VIEW recent_trades AS
SELECT
    t.*,
    br.symbol as run_symbol,
    s.name as strategy_name
FROM trades t
JOIN backtest_runs br ON t.run_id = br.id
JOIN strategies s ON br.strategy_id = s.id
ORDER BY t.trade_date DESC
LIMIT 100;

-- Function to calculate strategy statistics
CREATE OR REPLACE FUNCTION calculate_strategy_stats(p_strategy_id INTEGER)
RETURNS TABLE (
    total_runs INTEGER,
    avg_return DECIMAL,
    avg_sharpe DECIMAL,
    best_return DECIMAL,
    worst_return DECIMAL,
    avg_trades INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_runs,
        AVG(res.total_return) as avg_return,
        AVG(res.sharpe_ratio) as avg_sharpe,
        MAX(res.total_return) as best_return,
        MIN(res.total_return) as worst_return,
        AVG(res.num_trades)::INTEGER as avg_trades
    FROM backtest_runs br
    JOIN backtest_results res ON br.id = res.run_id
    WHERE br.strategy_id = p_strategy_id
    AND br.status = 'completed';
END;
$$ LANGUAGE plpgsql;