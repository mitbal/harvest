# Premium Trading Section — Implementation Plan

Status: approved direction, ready for implementation. Last updated 4 Oct 2026.

This document is written to be executed by a coding agent in a fresh session with no prior
conversation context. Everything needed to start is here. Work **one phase per session**.

## How to use this document (for the agent)

Kickoff prompt the user will give you, for example:

> Read `docs/premium-trading-plan.md` and implement Phase 0, task 0.1. Follow the ground rules.

1. Read this whole file first, then the files listed under the task you are given.
2. Do only the requested phase or task. Do not start the next phase.
3. Tick the checkboxes in this file as tasks are completed, and add a short note under
   "Progress log" at the bottom (date, what was done, anything left open).
4. Stop and ask the user at every point marked **[ASK USER]**.

## Ground rules

- **Branches and commits**: work on a branch named `premium/phase-N-<short-name>`. Commit when a task's
  checks pass. Do not push, open PRs, or merge unless the user asks.
- **No production side effects without approval** [ASK USER]: running SQL against Supabase,
  deleting or migrating data, creating cloud buckets, changing Railway cron settings, turning on
  `PREMIUM_ENFORCED`. Write the SQL or script, then hand it to the user to run.
- **Secrets**: never hardcode keys. Read from `st.secrets` in the app and environment variables in
  the pipeline, matching the existing pattern in `harvest/data.py` and `pipeline/daily_data.py`.
  Note: `apps/assistant/omjin.ipynb` contains a leaked API key; do not copy it anywhere.
- **No behaviour change during extraction tasks**: when moving code into `harvest/`, the page must
  produce the same results. Run the existing tests before and after.
- **Match the codebase style**: plain functions, type hints, short docstrings, pandas. Pages are
  Streamlit scripts registered in `main.py` with `st.Page`.
- **Streamlit work**: load the `developing-with-streamlit` skill before editing pages.
- **User-facing copy**: English, sentence case, plain language. Prices in IDR with thousands separators.

## Project context

- App: Panen Dividen (panendividen.com), Streamlit app for Indonesian (IDX) dividend investors.
  Entry point `server.py` (Streamlit `st.App` plus Starlette SEO routes); navigation in `main.py`.
- Auth: Google login via `st.login('google')` and `st.user` (config in `.streamlit/secrets.toml`).
  Example usage: `apps/porto/porto_overview.py`.
- Storage today:
  - Supabase Postgres: `users` (portfolios), `historical_prices` (about 4.3M rows of daily closes,
    IDX and S&P 500, since 2010), materialized view `mat_view_latest_returns`, storage bucket
    `harvest_dividend`.
  - Redis: Parquet-serialized DataFrames written by the pipeline (`div_score_<exch>`,
    `div_cal_<exch>_<year>`), see `store_frames_to_redis` in `pipeline/daily_data.py`.
- **Problem that drives Phase 0**: the free Supabase project exceeded its usage limit. The price
  table and its indexes are about the size of the 500 MB free limit, and pages pull prices row by
  row through the REST API. Decision: move price history to Parquet files on Cloudflare R2 and
  keep Supabase for small user data only.
- Pipeline: `pipeline/daily_data.py` runs on Railway cron (`railway.pipeline-jkse.toml`,
  `0 16 * * 1-5` UTC = 23:00 WIB weekdays). Price fetching in `pipeline/historical_prices_core.py`.
- Trading pages (all in `apps/trading/`):
  - `dividend_play.py` (about 1,200 lines) — engine already extracted to `harvest/dividend_play.py`.
  - `day_trading.py` (about 2,300 lines, "Short Term Swing Trading") — engine inline in the page.
  - `backtester.py` (about 900 lines, "Long Term Position Trading", "Growth at a Discount"
    fundamental screen) — logic inline in the page.
  - Each page today: shortlist filters, then optimizer button, then single-ticker backtest, then
    scanner button. Results live in `st.session_state` and are lost on refresh.
- Donations: Trakteer (`trakteer.id/mitbal`) and Buy Me a Coffee (`buymeacoffee.com/mitbal`),
  linked from the sidebar in `main.py`.

### Running and testing

- Tests: `pytest` from the repo root. `tests/conftest.py` stubs Streamlit, Redis and Supabase at
  collection time so pages can be imported. Relevant existing tests:
  `tests/test_dividend_play.py`, `tests/apps/test_short_term_swing.py`,
  `tests/pipeline/test_daily_data.py`, `tests/pipeline/test_historical_prices_core.py`.
- Local app: `.claude/launch.json` config `harvest` (runs `streamlit run main.py` on port 8599).
  Use the preview tools to verify page changes.
- Live smoke tests (production, run by the user): `make run-test-live` (`tests/live/smoke.spec.js`).

## Decisions already made

| Topic | Decision |
|---|---|
| Page structure | Keep three separate tool pages, each with Today / Evidence / Lab views, plus a Trading Today hub and My Trades |
| Computation | Pages only read. Weekly optimize job and daily scan job in the pipeline |
| Price storage | Parquet on Cloudflare R2 (local file fallback for development) |
| User data | Supabase: `users`, `premium_users`, `donations`, `positions`, `user_settings`, `trading_*`, `alert_log` |
| Premium gate | Google login + `premium_users` table, enforced server-side with `st.stop()`, behind env flag `PREMIUM_ENFORCED` |
| Free tier | One full Today card per tool, Evidence for one sample ticker, everything else locked |
| Alerts | Telegram first |
| First launch | After Phase 1 (Dividend Play only) |

Still open (do not decide these yourself) [ASK USER]: prices and durations (placeholders
Rp 25rb / 60rb / 200rb), OJK regulatory framing, whether Trakteer webhooks include donor email,
FMP plan limits for OHLCV.

Design references: the paywall and Today view mockups were shown to the user in chat. Their
content is described in Phase 1 tasks below; follow those descriptions.

## Target structure

```
Trading (sidebar section in main.py)
  Today          apps/trading/today.py         hub (Phase 5)
  Dividend Play  apps/trading/dividend_play.py Today | Evidence | Lab
  Swing Trading  apps/trading/swing_trading.py Today | Evidence | Lab (renamed from day_trading.py)
  Long Term      apps/trading/backtester.py    Today | Evidence | Lab
  My Trades      apps/trading/my_trades.py     (Phase 2)
```

## Architecture

```
Railway cron                                  Storage                          Streamlit pages
------------                                  -------                          ---------------
pipeline/historical_prices_core.py (daily) -> R2: prices/<exch>.parquet  --->  harvest/prices.py
pipeline/daily_data.py (daily)             -> Redis: latest_returns_<exch> ->  market_watch, stock pages
pipeline/trading_optimize.py (weekly)      -> Supabase: trading_setups   -+
pipeline/trading_scan.py (daily)           -> Supabase: trading_signals   +->  Today / Evidence / hub
                                           -> Supabase: trading_runs     -+    (freshness banner)
                                           -> Telegram alerts
server.py webhooks (BMC, Trakteer, Telegram)-> Supabase: premium_users ------>  harvest/premium.py
"Track this trade" dialog                  -> Supabase: positions -------->  My Trades
```

### Module layout

| Module | Status | Purpose |
|---|---|---|
| `harvest/prices.py` | new (Phase 0) | `load_prices(exchange, symbols=None, start=None, columns=("close",))`, R2 read with local fallback, cached |
| `harvest/premium.py` | new (Phase 0) | `is_premium(email)`, `require_premium(teaser=None)`, `is_enforced()` |
| `harvest/trading_store.py` | new (Phase 0) | read/write `trading_*`, `positions`, freshness check |
| `harvest/swing.py` | extract (Phase 0) | strategy components, regime, fees, backtest, gates from `day_trading.py` |
| `harvest/dividend_play.py` | exists | add `scan_signals(...)` (Phase 1) |
| `harvest/position.py` | extract (Phase 4) | fundamental scan, scoring, projections from `backtester.py` |
| `harvest/alerts.py` | new (Phase 2) | Telegram send, dedupe via `alert_log` |
| `pipeline/trading_optimize.py` | new (Phase 1) | weekly job, one function per tool |
| `pipeline/trading_scan.py` | new (Phase 1) | daily job, one function per tool |

## Data model

Write all DDL to `pipeline/premium_trading.sql`. The user runs it in the Supabase SQL editor.

```sql
create table if not exists premium_users (
  email text primary key,
  source text not null,                 -- 'trakteer' | 'bmc' | 'manual'
  expires_at timestamptz,               -- null = lifetime
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists donations (  -- raw webhook log, idempotent
  provider text not null,
  provider_tx_id text not null,
  email text,
  amount_idr numeric,
  payload jsonb not null,
  received_at timestamptz default now(),
  primary key (provider, provider_tx_id)
);

create table if not exists trading_runs (
  id bigserial primary key,
  tool text not null,                   -- 'dividend_play' | 'swing' | 'long_term'
  kind text not null,                   -- 'optimize' | 'scan'
  as_of date not null,
  status text not null,                 -- 'running' | 'ok' | 'failed'
  stats jsonb,
  started_at timestamptz,
  finished_at timestamptz
);

create table if not exists trading_setups (
  tool text not null,
  scope text not null,                  -- 'global' or a symbol
  params jsonb not null,
  holdout jsonb not null,               -- trades, win_rate, avg_trade, profit_factor, t_stat
  run_id bigint references trading_runs(id),
  primary key (tool, scope)
);

create table if not exists trading_signals (
  tool text not null,
  symbol text not null,
  as_of date not null,
  status text not null,                 -- 'actionable' | 'opening_soon' | 'candidate'
  entry_price numeric,
  stop_price numeric,
  target_price numeric,
  exit_by date,
  key_date date,                        -- cum date for dividend play
  key_date_estimated boolean,
  setup_scope text,                     -- 'global' or symbol
  detail jsonb,                         -- card copy, progress, holdout stats
  primary key (tool, symbol, as_of)
);

create table if not exists positions (
  id bigserial primary key,
  email text not null,
  tool text not null,
  symbol text not null,
  entry_date date not null,
  entry_price numeric not null,
  lots int not null,
  exit_rule jsonb not null,             -- copied from the signal at tracking time
  exit_by date,
  closed_at date,
  exit_price numeric,
  created_at timestamptz default now()
);
create index if not exists positions_open_by_email on positions (email) where closed_at is null;

create table if not exists user_settings (
  email text primary key,
  buy_fee_pct numeric default 0.15,
  sell_fee_pct numeric default 0.25,
  telegram_chat_id text,
  telegram_link_token text,
  alert_prefs jsonb default '{}'
);

create table if not exists alert_log (
  email text not null,
  kind text not null,
  ref text not null,
  sent_at timestamptz default now(),
  primary key (email, kind, ref)        -- one alert per event
);
```

Rules:
- Never overwrite past `trading_signals` rows; they become the public track record (Phase 5).
- All access goes through the server-side Supabase connection. App code must always filter user
  tables by `st.user.email`.

### Price file format

- Path: `prices/<exchange>.parquet` in the R2 bucket (env/secret `R2_BUCKET`), and locally
  `data/<exchange>/historical_prices.parquet` as fallback.
- Columns: `symbol` (string/category), `date` (date), `close` (float32). Phase 3 adds `open`,
  `high`, `low` (float32) and `volume` (int64).
- Sorted by `symbol, date`, written with row groups per symbol batch so filtered reads are cheap.
- Credentials: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` in
  environment variables (pipeline) and `st.secrets["r2"]` (app). R2 speaks the S3 API; use
  `boto3` or `s3fs` with the endpoint `https://<account_id>.r2.cloudflarestorage.com`. Add the
  dependency to both `requirements.txt` and `requirements-pipeline.txt`.

## Phase 0: Foundations

Goal: get off the Supabase limit, ship the premium gate behind a flag, and make the swing engine
importable by the pipeline.

### 0.1 Price storage migration

Read first: `pipeline/historical_prices_core.py`, `pipeline/historical_prices.py`,
`pipeline/daily_data.py` (`refresh_returns_view`, `get_latest_returns_from_db`),
`apps/trading/day_trading.py` (`fetch_close_history`, single-ticker backtest around line 1958),
`apps/trading/dividend_play.py` (`fetch_price_history`), `apps/screener/market_watch.py` (around line 198),
and the related tests.

- [x] `harvest/prices.py`: `load_prices(exchange, symbols=None, start=None, columns=("close",))`
      returning a long DataFrame `symbol, date, <columns>`. Reads from R2, falls back to the
      local Parquet file, wrapped in `st.cache_data(ttl=6h)` at the call sites (keep the module
      itself Streamlit-free so the pipeline can use it).
- [x] `harvest/prices.py`: `write_prices(exchange, frame)` used by the pipeline.
- [x] `pipeline/historical_prices_core.py`: incremental mode loads the existing Parquet, appends
      and dedupes the last 7 days, writes back. Remove the Supabase upsert path behind a
      `--target {r2,supabase}` option defaulting to `r2`, so rollback is possible.
- [x] One-off script `pipeline/migrate_prices_to_parquet.py`: build the Parquet files from the
      local pickle cache (`data/<exch>/historical_prices.pkl`) and upload. [ASK USER] before running
      the upload; the user must create the R2 bucket and credentials first.
- [x] Latest returns: compute 7d / 1m / 1y / 10y returns in `pipeline/daily_data.py` from the
      Parquet data and store as Redis key `latest_returns_<exch>`. Replace reads of
      `mat_view_latest_returns` with the Redis key.
- [x] Replace all `historical_prices` Supabase reads in `day_trading.py`, `dividend_play.py`,
      `market_watch.py` with `load_prices`. Keep the live-API fallback in `dividend_play.py`.
- [x] Tests: `tests/test_prices.py` (local fallback, symbol and date filters, columns),
      update `tests/pipeline/test_historical_prices_core.py` and `test_daily_data.py`.
- [x] Hand the user the cleanup SQL, to run only after production reads are confirmed working:
      `drop index if exists idx_historical_prices_symbol;` now, and later
      `drop materialized view mat_view_latest_returns; drop table historical_prices;` [ASK USER].

Done when: no app or pipeline code queries `historical_prices` or `mat_view_latest_returns`;
pages load prices from Parquet; tests pass.

### 0.2 Premium gate

Read first: `main.py`, `apps/porto/porto_overview.py` (login pattern).

- [x] Write `pipeline/premium_trading.sql` with the DDL above. [ASK USER] to run it.
- [x] `harvest/premium.py`:
  - `is_enforced()` reads env `PREMIUM_ENFORCED` (default false).
  - `is_premium(email)` checks `premium_users` (`expires_at` null or in the future), cached 10 minutes.
  - `require_premium(teaser=None)`: if not enforced, return. If logged out, render the paywall
    with a Google login button and `st.stop()`. If logged in but not premium, render the paywall
    and `st.stop()`. `teaser` is an optional callable rendered above the paywall.
  - Paywall content: title "Unlock the trading lab"; one line on why (site stays free and
    ad-free, supporters cover data and servers); benefits list (full scanner, backtest any ticker,
    swing and long-term labs); tier cards from a config dict; three steps (log in with Google,
    donate on Trakteer or Buy Me a Coffee with the same email, access unlocks within minutes);
    buttons for login, Trakteer, Buy Me a Coffee; footnote for help via r/panendividen.
    When logged in, show "Logged in as <email>" so the user knows which email to donate with.
- [x] Tests with mocked `st.user`: enforced off, logged out, logged in free, premium, expired.

Done when: with `PREMIUM_ENFORCED=true` locally, the three trading pages are blocked for
non-premium users (call `require_premium()` at the top of each page for now).

### 0.3 Trading store

- [x] `harvest/trading_store.py`: `start_run`, `finish_run`, `latest_run(tool, kind)`,
      `write_setups`, `read_setups(tool)`, `write_signals`, `read_signals(tool, as_of=None)`,
      `is_stale(tool, max_trading_days=1)`, plus position CRUD. Supabase client passed in, so the
      pipeline (supabase-py) and the app (`st.connection`) can both use it.
- [x] Unit tests with a fake client.

### 0.4 Extract the swing engine

- [x] Move pure logic from `apps/trading/day_trading.py` into `harvest/swing.py`: strategy
      constants, `_wilder_rsi`, `_close_supertrend`, `_dividend_adjust`, `_idx_tick_size`,
      `_slippage_fraction`, `_build_trade_fees`, `_build_market_breadth`, `_strategy_components`,
      `_build_regime_signals`, `_run_regime_backtest`, regime helpers, summary and gate functions.
      Drop leading underscores for public functions.
- [x] Rename the page to `apps/trading/swing_trading.py` and update `main.py`.
- [x] `tests/apps/test_short_term_swing.py` passes unchanged in behaviour (update imports only);
      add `tests/test_swing.py` for the moved functions.
- [x] Verify the page in the local preview: optimizer and single-ticker backtest give the same
      numbers as before for one fixed ticker and date range.

## Phase 1: Dividend Play revamp (supporter launch)

Read first: `apps/trading/dividend_play.py`, `harvest/dividend_play.py`, `tests/test_dividend_play.py`.

### 1.1 Pipeline jobs

- [x] `pipeline/trading_optimize.py::dividend_play()`: default shortlist (same filters and defaults
      as the page), Balanced preset, existing holdout gates. Writes the global setup and
      per-stock setups to `trading_setups`, records a `trading_runs` row.
- [x] `pipeline/trading_scan.py::dividend_play()`: for each shortlist stock, use
      `estimate_next_ex_dates` and `entry_status` with the stored setup. Write `trading_signals`:
      `actionable` when inside the entry window (and dip rule satisfied if any), `opening_soon`
      when the window opens within 45 days. Fill `detail` with: company name, yield, setup
      description (`format_setup`), window day N of M, sessions left, holdout wins/trades and avg
      return after costs.
- [x] Announced vs estimated cum date: use announced dates from the existing dividend calendar
      data (`div_cal_*`) when present, set `key_date_estimated` accordingly.
- [x] CLI entry points (`python -m pipeline.trading_scan --tool dividend_play`). Pipeline tests
      with fixture prices and a fake store.
- [x] Railway: propose cron config (daily scan after `daily_data`, weekly optimize on Sunday).
      [ASK USER] before changing any `railway*.toml`.

### 1.2 Page restructure

- [x] View switch with `st.segmented_control` (Today, Evidence, Lab), persisted in
      `st.query_params["view"]`; `?stock=SYMBOL` opens Evidence for that stock.
- [x] **Today** (no inputs):
  - Status line: "Updated <weekday date>, <time> WIB after close", next update; stale warning via
    `is_stale`.
  - Summary metrics: entry windows open, opening in 14 days, your positions (with "N exits <day>").
  - "Entry window open now" cards, two per row on desktop, one on mobile. Each card: ticker,
    company, yield; badge "Cum date announced" or "Cum date estimated" (prefix estimated dates
    with "~"); entry at or below, sell by, cum date; progress bar "Day N of M in window" with
    sessions left (or "Dip rule active, X% below 10-day high"); one line of setup and holdout
    record; buttons "Track this trade" (Phase 2, hidden until then) and "See evidence".
  - "Opening soon" compact list: ticker, window opens (est. badge), historical run-up, "Alert me"
    (Phase 2, hidden until then).
  - Footer: "Historical backtest of past dividend events. Check the announced cum date with IDX
    before trading. Not financial advice."
- [x] **Evidence**: existing single-ticker section, prefilled from `trading_setups` (per-stock
      setup if present, else global), runs automatically on load. Keep the existing tabs
      (price and trades, typical run-up, trade log).
- [x] **Lab**: existing shortlist, optimizer and manual parameters, with a caption that results
      here are research only and do not change the published signals.
- [x] Free tier via `harvest/premium.py`: Today shows the first card in full and the remaining
      tickers as names only; Evidence allowed only for the sample ticker (config, default `BBRI.JK`);
      Lab gated.
- [x] AppTest coverage for the three views in logged out, free and premium states.
- [x] Verify in the local preview at desktop and mobile widths; Today must render in under 3
      seconds with data present.

Done when: the page opens on Today with stored signals and no button press; Evidence and Lab
work; the gate behaves per tier with the flag on.

### 1.3 Launch checklist [ASK USER for each]

- [ ] User runs the Supabase SQL, sets secrets, deploys pipeline and app.
- [ ] User adds early supporters to `premium_users` manually.
- [ ] User turns on `PREMIUM_ENFORCED` in production.
- [ ] Update `tests/live/smoke.spec.js` to expect the paywall on trading pages for anonymous visitors.

## Phase 2: Positions, alerts, donation webhooks

Read first: `server.py` (Starlette routes and middleware), `harvest/trading_store.py`.

- [x] "Track this trade" `st.dialog`: entry price (default from signal), lots, writes `positions`
      with the exit rule and `exit_by` copied from the signal.
- [x] `apps/trading/my_trades.py`: open positions sorted by `exit_by` with countdown in sessions
      (use IDX trading days; `holidays` is already a dependency), closed positions with results,
      settings form (fees, Telegram link, alert preferences). Register in `main.py`.
- [ ] Telegram:
  - Bot token in env/secrets (`TELEGRAM_BOT_TOKEN`). [ASK USER] to create the bot.
  - Linking: settings shows `https://t.me/<bot>?start=<token>`; `POST /webhooks/telegram` in
    `server.py` stores `chat_id` for the matching token.
  - `harvest/alerts.py` sends, called from the daily scan: new actionable signal (if opted in),
    "Alert me" window opening, exit reminder one trading session before `exit_by`. Dedupe with
    `alert_log`.
- [x] Donation webhooks in `server.py`:
  - `POST /webhooks/bmc`: verify signature header with `BMC_WEBHOOK_SECRET`, insert `donations`
    (ignore duplicates), upsert `premium_users` with duration from a config mapping amount to days.
  - `POST /webhooks/trakteer`: verify token `TRAKTEER_WEBHOOK_TOKEN`, same flow. If the payload has
    no email, try to read one from the supporter message; otherwise store with `email` null for
    manual matching.
  - Return 2xx quickly; never log full payloads containing personal data to stdout.
- [x] Tests: bad signature, duplicate transaction, unknown email, expiry extension (a new donation
      extends from the later of now and current expiry).

Done when: a test webhook unlocks access without manual steps, and a tracked position triggers an
exit reminder in Telegram (verified with a test chat).

## Phase 3: Swing Trading revamp

- [x] OHLCV: extend `historical_prices_core.py` and the Parquet schema with `open, high, low,
      volume`; backfill script. [ASK USER] to confirm FMP plan limits before the backfill.
- [ ] `harvest/swing.py`:
  - Stop and target fills using daily high and low, gaps fill at the open.
  - Walk-forward validation (rolling train/test windows) replacing the single holdout split.
  - Stricter gate accounting for the number of parameter combinations tried.
- [ ] Weekly optimize and daily scan functions for `swing`, same pattern as Phase 1.
- [ ] Page restructure into Today / Evidence / Lab. Today cards: entry, stop, target, max hold
      sessions, holdout record. Remove the P/E filter from the shortlist.
- [ ] Position sizing on cards: lots and IDR at risk from the user's risk percent (user setting),
      prices rounded to IDX ticks (`idx_tick_size`), respecting ARA/ARB limits.
- [ ] Tests for stop fills, gap handling, walk-forward splits, tick rounding.

Done when: published swing signals come from OHLCV and walk-forward validation; Today loads instantly.

## Phase 4: Long Term revamp

- [ ] Extract `harvest/position.py` from `apps/trading/backtester.py` (`_analyze_stock`, scan,
      scoring, projections). No behaviour change.
- [ ] Daily scan job storing ranked candidates as `trading_signals` (`status='candidate'`), with
      coverage stats in `trading_runs.stats` (analyzed, failed, skipped) instead of silently
      dropping failures.
- [ ] Quality checks: debt to equity, interest coverage, operating cash flow vs net income, ROE,
      share count change. Sector rule: price to book for banks instead of P/S.
- [ ] Historical backtest of the screen: yearly rebalance of the top N using financials available
      at the time (statement date plus reporting lag), compared with the JKSE index.
- [ ] Exit monitoring alerts for tracked positions: P/E back at historical mean, or new quarter
      turning TTM growth negative.
- [ ] Page restructure into Today (instant filters on stored candidates) / Evidence (current
      detail view) / Lab (current form).

## Phase 5: Hub and track record

- [ ] `apps/trading/today.py`: Act now (signals from all tools), Your positions, Coming up.
      Make it the first page in the Trading section in `main.py`; lock icons in page titles for
      non-supporters.
- [ ] Track record page from historical `trading_signals` and their realised outcomes.
- [ ] Portfolio simulation for Dividend Play and Swing: fixed capital, max open positions,
      100-share lots, liquidity cap, equity curve vs JKSE.
- [ ] Give the user real track-record numbers to replace placeholders in the promo animation.

## Testing checklist (every phase)

- `pytest` passes.
- New engine code has unit tests; new pipeline code has tests with fixtures and fake clients.
- Pages verified in the local preview (desktop and mobile width), including logged out, free
  and premium states where relevant.

## Risks

- Regulatory: paid buy/sell signals may count as investment advice under OJK rules. Copy should
  frame the product as a research and backtesting tool. The user decides.
- Weekly optimize runtime must fit Railway's 6-hour timeout; split per tool if needed.
- R2 read latency on cold start: cache aggressively and keep the local fallback.

## Progress log

- 2026-10-04: plan written. No implementation started.
- 2026-10-05: Phase 0, task 0.1 implemented on branch `premium/phase-0-price-storage`.
  - `harvest/prices.py`: `load_prices` (R2, then local file, then empty frame; also takes `end`),
    strict `read_prices` for the pipeline, `write_prices`, `merge_prices`. R2 access goes through
    `pyarrow.fs.S3FileSystem` (pyarrow is already in both requirements files), so no new
    dependency. Settings: `st.secrets["r2"]` keys `account_id`, `access_key_id`,
    `secret_access_key`, `bucket`, or env `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
    `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`. Closes stored as float32, returned as float64.
  - Pipeline: `run_historical_pipeline(target="r2")` merges the refetched days into the R2 file and
    refuses to write if the stored history can't be read or is empty (except in backfill mode).
    `--target supabase` (core CLI) / `--price-target supabase` (`daily_data`) restores the old
    upsert and materialized-view path. `daily_data` now requires the `R2_*` env vars and
    publishes `latest_returns_<exch>` to Redis.
  - Pages: `day_trading.py`, `dividend_play.py`, `market_watch.py` read through cached
    `load_prices` wrappers. The swing single-ticker backtest used to get at most 1,000 rows from
    the unpaged Supabase query; it now gets the full date range, so long backtests can differ.
  - Local Parquet built from the pickle caches (jkse 4.9 MB, sp500 6.4 MB; gitignored).
    Verified in the local preview: heatmap, swing single-ticker backtest, Dividend Play
    single-ticker backtest. Optimizer and scanner paths not clicked through.
  - Open [ASK USER]: create the R2 bucket and credentials, add them to Railway (pipeline env) and
    the app secrets, then run `python -m pipeline.migrate_prices_to_parquet --exchange jkse
    --fill-gap --upload` (same for sp500). The pickle caches end 2026-04-08/06, so `--fill-gap`
    refetches from FMP. Deploy the pipeline only after the upload, because incremental runs
    refuse to start without a stored history. Then run the cleanup SQL.
- 2026-10-05: Phase 0, task 0.2 implemented on branch `premium/phase-0-premium-gate`, stacked on
  `premium/phase-0-price-storage` (0.1).
  - `pipeline/premium_trading.sql`: the DDL above, plus row level security on every table. The
    app reads with the public key and signs in as the app account, so only that account (via the
    `app_server()` function, whose email must be filled in before running) and the service-role
    key get access. Without RLS anyone with the public key could insert themselves into
    `premium_users`.
  - `harvest/premium.py`: `is_enforced`, `is_premium` (10-minute cache; lookup errors count as
    not premium and are not cached), `require_premium(teaser=None)` with the paywall. Tier
    prices and durations are the placeholders in `TIERS`.
  - `require_premium()` called right after `st.set_page_config` in `backtester.py`,
    `day_trading.py`, `dividend_play.py`.
  - Verified locally with `PREMIUM_ENFORCED=true` (logged out): all three trading pages show only
    the paywall, Market Heatmap is unaffected, layout checked at desktop and mobile width. The
    logged-in free and supporter states are covered by unit tests only (no Google login in the
    preview).
  - Open [ASK USER]: replace the email in `app_server()` and run the SQL; confirm whether the
    pipeline's `SUPABASE_KEY` is the service-role key (needed for Phase 1 writes under RLS);
    final tier prices and durations.
- 2026-10-05: Phase 0, task 0.3 implemented on branch `premium/phase-0-trading-store`, stacked on
  `premium/phase-0-premium-gate`.
  - `harvest/trading_store.py` (Streamlit-free, client passed in): runs (`start_run` records
    `status='running'`, now in the SQL comment; `latest_run` defaults to `ok` runs), setups
    (`write_setups` replaces the tool's scopes, so dropped per-stock setups are deleted;
    `read_setups` returns `{scope: {params, holdout, run_id}}`), signals (`write_signals` refuses
    a day older than the latest stored one and replaces only its own day on rerun;
    `read_signals` returns a DataFrame for the latest or given day), `is_stale`, positions
    (`create_position`, `list_positions`, `update_position`, `close_position`,
    `delete_position`, all filtered by the owner's email), and `idx_trading_days`.
  - `is_stale`: the latest successful scan is stale once `max_trading_days` IDX sessions
    before today have no scan (today's session doesn't count until its nightly scan). Uses the
    same IDX closing rule as `apps/calendar/calendar.py::market_holidays`.
  - `tests/fake_supabase.py`: in-memory query builder for store and pipeline tests. The store's
    calls were also checked against the real supabase-py 2.31 builder (request URLs only, no
    network).
- 2026-10-05: Phase 0, task 0.4 implemented on branch `premium/phase-0-swing-engine`, stacked on
  `premium/phase-0-trading-store`. Phase 0 is complete apart from the [ASK USER] items above.
  - `harvest/swing.py`: constants (including `SEARCH_PRESETS` and the parameter labels and input
    bounds), regime helpers, indicators, dividend adjustment, IDX ticks and fees, breadth,
    `strategy_components`, `build_regime_signals`, `run_regime_backtest`, summaries, gates,
    stock selection, `scanner_strategy_plan`, `clean_price_series`, `is_near_setup`. Copied from
    the page by line range, then renamed. Leading underscores dropped except for internal helpers
    (`_align_to_prices`, `_weighted_mean`, `_score_stock_strategy_candidates`).
    `_test_stock_strategies` became `evaluate_stock_strategies` (a `test_` name would be collected
    by pytest). `is_near_setup` takes `regime=` instead of reading the page's `curr_regime`.
  - Page renamed to `apps/trading/swing_trading.py`; `main.py` keeps `url_path='day_trading'` so
    home page links, bookmarks and `tests/live/smoke.spec.js` keep working. Removed the stale
    commented-out `page_swing_trading` entry.
  - `tests/apps/test_short_term_swing.py`: import and name changes only, plus the regime
    argument, plus a smoke test that the page imports. New `tests/test_swing.py`.
  - Same numbers: a golden-master script on real local prices (12 IDX tickers, Balanced preset,
    7 strategies, 308 result rows) gave bit-identical results, summary, per-stock selections,
    scanner plan and single backtest before and after. In the preview, the old and new page gave
    identical single-ticker results for BBCA.JK (Regime RSI and Supertrend pullback). The
    optimizer could not run in the preview with either version: the shortlist needs the last 60
    sessions of prices and the local file ends 2026-04-08 (no FMP key locally to top it up).
- 2026-10-05: Phase 1, task 1.1 implemented on branch `premium/phase-1-dividend-play`, stacked on
  `premium/phase-0-swing-engine`. The R2 price files were uploaded the same day (jkse to
  2026-10-05, sp500 to 2026-10-02).
  - Shared logic moved from the page into `harvest/dividend_play.py` so the page and the jobs
    cannot drift: gates and defaults, `SEARCH_PRESETS`, `build_shortlist`, `holdout_split_date`,
    `trade_costs`, `play_data`, `setup_from_row`, `qualify_setups`, `next_play_event`,
    `scan_signal`. The page calls them; results are unchanged except two scanner fixes that apply
    to both: an estimated ex-date within 60 days of an announced one is dropped (the announced
    one wins), and a stock whose last dividend is over 550 days old gets no projected ex-date.
  - Announced vs estimated: taken from the dividend history itself (a future ex-date already in
    the history is announced, a rolled-forward one is estimated). `div_cal_*` is built from the
    same history, so reading it adds nothing.
  - `pipeline/trading_data.py` (loaders: `div_score_jkse` from Redis plus FMP profiles for
    company name and liquidity, dividend pickle from Supabase storage, prices strictly from R2),
    `pipeline/trading_optimize.py`, `pipeline/trading_scan.py`. Scan `as_of` is the latest
    price date. `actionable` = page "Buy Zone", `opening_soon` = window opens within 45 days;
    "in window, waiting for the dip" is not published (counted in run stats). Dip-rule entry
    price is the dip level (MA x (1 - dip%)). Key date is the cum date (last IDX session before
    the ex-date); `exit_by` is the cum date or the first session `hold_after_days` after ex.
  - Dry run on real data (in-memory store): optimize 16 s (129 stocks x 180 setups, global setup
    "Buy 90-30d before ex, dip 0% < MA20; sell 10d after ex", 91 per-stock setups), scan 6 s
    (14 actionable, 8 opening soon). The page's own optimizer and scanner gave the same global
    setup and the same 14 / 8 counts.
  - Added `holidays` to `requirements-pipeline.txt` (needed by `trading_store`).
  - Open [ASK USER]: Railway schedule (proposal in the session summary); per-stock setups are
    the best of 180 per stock on as few as 4 holdout trades, which flatters them (the page
    already says so); consider stricter per-stock gates before launch.
- 2026-10-05: Phase 1, task 1.2 implemented on the same branch.
  - Page opens on Today. `st.segmented_control(key='view', bind='query-params')`; the Evidence
    stock picker is bound to `?stock=` (full symbol, e.g. `BBRI.JK`; no `format_func`, because
    the bound URL value is the displayed label). Initial values go through session state, not
    `default`, so the in-page buttons don't trigger Streamlit's default-vs-session-state warning.
  - Today reads only the store (no prices or dividends): about 1 s on reload, 2.6 s cold. Cards
    sorted by holdout average; entry level rounded down to the IDX tick; "Window opens" shows the
    first trading day (windows are counted in calendar days). Dip copy follows the engine:
    "Dip rule active, X% below its N-day average" (the plan's "10-day high" doesn't match the
    rule). "Your positions" shows the open Dividend Play positions when logged in ("-" otherwise);
    Track this trade and Alert me stay hidden until Phase 2.
  - Evidence runs on load with the published per-stock setup, else the global one, else the
    baseline (with a note). `render_backtest` is shared with the Lab's manual backtest.
  - Lab = previous page, behind `require_premium()`, with the research-only caption.
  - Tiers: `harvest.premium.has_access()` / `current_email()`; `get_connection` is now public.
    Free: first card in full, the rest and the opening-soon list as names, an unlock prompt that
    opens the Lab paywall; Evidence only for `FREE_EVIDENCE_TICKER` (BBRI.JK).
  - AppTest suite in `apptests/` (14 tests: three views x premium/free/logged out, ?stock link,
    See evidence click, unlock click, store outage). It needs real Streamlit, so it runs in a
    subprocess from `tests/test_apptests.py`; plain `pytest` covers it.
  - Verified in the preview against the real-data dry-run signals (desktop and mobile, full and
    free tiers) and inside the real app navigation with no tables (graceful messages).
  - Known gap: after "See evidence" the URL shows `?view=Evidence` without `&stock=`; Streamlit
    doesn't push a bound value set before the widget exists. Inbound `?stock=` links work, and the
    URL updates once the user picks a stock.
- 2026-10-05: Railway schedule approved by the user (recommended option). `railway.pipeline-jkse.toml`
  now runs `trading_scan --tool dividend_play` after `daily_data` in the same job (`sh -c "... &&
  ..."`, scan timeout 1h), so it only runs on fresh prices. New `railway.trading-optimize.toml`:
  weekly optimize, Sunday 10:00 WIB (`0 3 * * 0`), 3h timeout. The user creates the optimize
  service in Railway with the same environment as the jkse pipeline and runs it once by hand
  before the first scan.
- 2026-10-05: Phase 2 built on branch `premium/phase-2-positions-alerts` (stacked on the Phase 1
  branch). Telegram is code-complete but unchecked above until the bot exists [ASK USER].
  - Trakteer webhooks carry no supporter email (fields: created_at, transaction_id, type,
    supporter_name, supporter_avatar, supporter_message, media, unit, unit_icon, quantity,
    price, net_amount; token in `X-Webhook-Token`). The email is read from the supporter
    message; otherwise the donation is stored with `email` null. The paywall now asks Trakteer
    donors to write their email in the message. BMC: hex HMAC-SHA256 of the raw body in
    `x-signature-sha256`; `data.supporter_email`, `data.amount`, `data.currency`.
  - `harvest/donations.py`: verification, parsing, `TIERS` (now also used by the paywall, with
    `min_idr` and `days`), `IDR_PER_UNIT` for BMC's USD (placeholder 16,000), idempotent
    `record_donation` (duplicate delivery changes nothing; extension from the later of now and
    the current expiry; lifetime stays lifetime). `is_enforced` / `is_active` moved here
    (Streamlit-free) and re-exported by `harvest/premium.py`.
  - `harvest/webhooks.py` + three POST routes in `server.py`. They use a service-role client from
    `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` (new app secret). Store errors return 500 so the
    provider retries; logs carry provider, transaction id and outcome only.
  - `harvest/alerts.py`: signal alerts (watched stock or "every new entry window" opt-in, only
    for users with access) and exit reminders (one session before `exit_by`, for anyone who
    tracked the trade), each once via `alert_log`; a failed send is un-logged so the next run
    retries. Called from the nightly scan; failures are recorded in run stats, never fail it.
    `/start <token>` links a chat, `/stop` unlinks. `pipeline/telegram_setup.py` registers the
    webhook (run by the user).
  - Today: "Track this trade" dialog (prefilled tick-rounded entry, lots, date; copies the
    published rule, cum date and dividend into `exit_rule`) and "Alert me" pills under the
    opening-soon table. My Trades: open positions with session countdown, close/remove dialogs,
    closed results after the user's fees (plus the dividend when held past the cum date),
    settings form and Telegram linking. Registered in `main.py`.
  - Tests: donations/webhooks/alerts (29), store additions, AppTests for My Trades and the Today
    controls (23 AppTests in total). AppTest can't replay clicks inside a dialog; the Track save
    was checked in the browser.
  - Open [ASK USER]: create the bot (BotFather) and set `TELEGRAM_BOT_TOKEN`,
    `TELEGRAM_BOT_USERNAME`, `TELEGRAM_WEBHOOK_SECRET`; set `SUPABASE_SERVICE_KEY`,
    `BMC_WEBHOOK_SECRET`, `TRAKTEER_WEBHOOK_TOKEN` in the app; register the BMC and Trakteer
    webhook URLs; run `pipeline/telegram_setup.py`; confirm tier prices, durations and the USD
    rate. "Done when" needs these.
- 2026-10-05: Phase 3 started on branch `premium/phase-3-swing` (stacked on Phase 2).
  - OHLCV: `fetch_stock_price` keeps open/high/low/close/volume (the FMP endpoint already
    returned them). `read_prices` returns requested columns the file lacks as NaN (S&P 500 stays
    close-only until its daily runs add OHLC to new rows). `harvest.prices.OHLCV_COLUMNS`.
  - Backfill approved by the user and run: `python -m pipeline.backfill_ohlcv --exchange jkse
    --upload`. 893 symbols refreshed from FMP since 2010 (none failed), 2,427,827 rows, all with
    OHLC. Previous file kept as `prices/jkse.backup-2026-10-05.parquet` in R2. 98.3% of
    overlapping closes are identical; 13 symbols changed by 80-96% (consistent split adjustment,
    e.g. MLPT, DSSA). 219 rows (0.009%) have a high/low that doesn't bracket open/close; the
    engine clamps them. 11.6% of rows have zero volume.
