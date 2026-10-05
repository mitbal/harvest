-- Premium trading tables (docs/premium-trading-plan.md, "Data model").
-- Run once in the Supabase SQL editor. Safe to re-run.
--
-- BEFORE RUNNING: replace app-account@example.com in app_server() below with the
-- EMAIL_ADDRESS the app signs in with ([connections.supabase] in secrets.toml).
--
-- Row level security is on for every table. The app reads with the public key,
-- so without it anyone could edit premium_users through the REST API. The
-- service-role key (pipeline, webhooks) bypasses RLS; the signed-in app account
-- is allowed through by the policies below. Nothing else gets access.

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


-- Access control ------------------------------------------------------------

create or replace function app_server() returns boolean
language sql stable as $$
  select coalesce(auth.jwt() ->> 'email', '') = 'app-account@example.com'
$$;

do $$
declare
  t text;
begin
  foreach t in array array[
    'premium_users', 'donations', 'trading_runs', 'trading_setups',
    'trading_signals', 'positions', 'user_settings', 'alert_log'
  ] loop
    execute format('alter table %I enable row level security', t);
    execute format('drop policy if exists app_server_all on %I', t);
    execute format(
      'create policy app_server_all on %I for all to authenticated '
      'using (app_server()) with check (app_server())',
      t
    );
  end loop;
end $$;


-- Manual supporter, until the donation webhooks exist (Phase 2):
-- insert into premium_users (email, source, expires_at)
-- values ('supporter@example.com', 'manual', now() + interval '30 days')
-- on conflict (email) do update
--   set source = excluded.source, expires_at = excluded.expires_at, updated_at = now();
