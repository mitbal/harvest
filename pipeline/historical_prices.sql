-- SQL Schema for Historical Stock Prices
-- Run this in your Supabase SQL Editor once.

CREATE TABLE IF NOT EXISTS historical_prices (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    close FLOAT NOT NULL,
    PRIMARY KEY (symbol, date)
);

-- Index for faster queries by symbol or date
CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol ON historical_prices (symbol);
CREATE INDEX IF NOT EXISTS idx_historical_prices_date ON historical_prices (date);

-- Comment for documentation
COMMENT ON TABLE historical_prices IS 'Daily closing prices for stocks (IDX and SP500).';

-- View to compute the latest returns for all stocks exactly
-- This forces Postgres to do the math and keeps the REST API fast
CREATE OR REPLACE VIEW view_latest_returns AS
SELECT symbol, return_7d, return_1m, return_1y, return_10y
FROM (
    SELECT 
        symbol,
        date,
        (close / LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date)) - 1 AS return_7d,
        (close / LAG(close, 21) OVER (PARTITION BY symbol ORDER BY date)) - 1 AS return_1m,
        (close / LAG(close, 252) OVER (PARTITION BY symbol ORDER BY date)) - 1 AS return_1y,
        (close / LAG(close, 2520) OVER (PARTITION BY symbol ORDER BY date)) - 1 AS return_10y,
        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
    FROM historical_prices
) sub
WHERE rn = 1;
