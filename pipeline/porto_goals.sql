-- Per-user income goal for the Portfolio page (apps/porto/porto_overview.py).
-- Run once in the Supabase SQL editor. Safe to re-run.

alter table users add column if not exists target_income numeric;    -- IDR million a year
alter table users add column if not exists benchmark_yield numeric;  -- percent

-- A user can save a goal before they ever sync a portfolio, which inserts a row
-- with no portfolio yet.
alter table users alter column portfolio drop not null;
