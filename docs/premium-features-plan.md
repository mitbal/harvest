# Premium Features, Round 2 — Implementation Plan

Status: proposal, not yet approved task by task. Written 12 Nov 2026.

This continues `docs/premium-trading-plan.md` (Phases 0 to 5, the trading section). It covers
what makes a visitor become a supporter and stay one: conversion fixes, alerts, supporter
features on the dividend pages, and a fuller trading journal.

It is written to be executed by a coding agent in a fresh session with no prior conversation
context. Work **one phase per session**.

## How to use this document (for the agent)

Kickoff prompt the user will give you, for example:

> Read `docs/premium-features-plan.md` and implement Phase A, task A.1. Follow the ground rules.

1. Read this whole file, then the "Project context" and "Ground rules" of
   `docs/premium-trading-plan.md`, then the files listed under your task.
2. Do only the requested phase or task. Do not start the next one.
3. Tick the checkboxes here as tasks are completed and add a note under "Progress log"
   (date, what was done, what is left open).
4. Stop and ask the user at every point marked **[ASK USER]**.

## Ground rules

The ground rules of `docs/premium-trading-plan.md` apply unchanged (branches named
`premium/<phase>-<short-name>`, no production side effects without approval, secrets from
`st.secrets` / environment, load the `developing-with-streamlit` skill before editing pages,
English sentence-case copy, IDR with thousands separators). In addition:

- **Never lock something that is free today.** Every dividend page stays fully usable for
  visitors. Supporter features are *added* sections. The rule for what goes behind the gate:
  free = anything a visitor can look up; supporter = personal, proactive or saved.
- **A locked section never stops the page.** `require_premium()` ends the script with
  `st.stop()`, which is right for the trading Lab but wrong inside a free page. Use the
  non-blocking helper from task A.0 instead.
- **No returns promises in copy.** The backtests are mixed (see "Why this plan" below) and paid
  signals carry OJK risk. Describe features as research, reminders and record keeping. Keep the
  "not financial advice" footers.
- **Alerts are sent at most once per event** through `store.log_alert` / `forget_alert`
  (`alert_log`), the same way `harvest/alerts.py` does it today. A failed send must never fail a
  pipeline run.
- **Pipeline jobs only read and write through `harvest/trading_store.py`-style functions** that
  take the Supabase client as their first argument, so they can be tested with
  `tests/fake_supabase.py`.
- **Symbols**: portfolios store IDX tickers without the suffix (`BBRI`); the trading tables and
  the dividend history use `BBRI.JK`. Convert at the boundary and say which one a function takes.
- The working tree may hold uncommitted changes from other work. Do not revert or reformat files
  you were not asked to touch.

## Why this plan

Findings from a review of the code on 12 Nov 2026:

| Finding | Where |
|---|---|
| Track Record is registered but commented out of the navigation, so nobody can open it | `main.py` (`# page_track_record,`) |
| The paywall lists three stale benefits | `harvest/premium.py` `BENEFITS` |
| Nothing shows a supporter when access ends, and nothing reminds them to renew | no `expires_at` use in any page |
| Unlocking means donating off-site and, on Trakteer, typing an email into the message | `harvest/donations.py` |
| Telegram alerts are built but the bot was never created; signal alerts cover Dividend Play only | `harvest/alerts.py`, plan log 2026-10-23 |
| The free tier shows the top-ranked card of each tool every day | `apps/trading/my_trades.py` (`rows.head(... if access else 1)`) |
| The journal has no current price, no edit, no manual entry, no export | `apps/trading/my_trades.py` |
| Supporter access covers the trading pages only; the audience is dividend investors | no `has_access` outside `apps/trading` |
| Backtests are mixed: Long Term +2.5% a year vs JKSE +4.0%; one swing family passes; a Dividend Play replay averaged −2.6% over 22 trades | `docs/premium-trading-plan.md` progress log |

So the pitch is: a nightly routine that saves time, an honest public record, and alerts and
reports about the supporter's own stocks.

## Decisions for the user [ASK USER before the first task that depends on each]

| # | Decision | Recommendation | Needed by |
|---|---|---|---|
| D1 | Free tier on Today views: keep "first card in full" or switch to "yesterday's cards in full, today's for supporters" | Switch to delayed | A.5 |
| D2 | Free trial: length, and whether it starts by itself or on a button | 7 days, on a button, once per email | A.4 |
| D3 | Final tier prices and durations (still placeholders in `harvest/donations.py` `TIERS`) | — | A.2 |
| D4 | Alert channels: Telegram only, or also email (needs a mail provider and its key) | Telegram first | B.1 |
| D5 | Create the Telegram bot and set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME`, `TELEGRAM_WEBHOOK_SECRET` | Required for Phase B | B.1 |
| D6 | Whether the pipeline's Supabase key may read the `users` table (portfolios), and its RLS state | Service-role key | C.3 |
| D7 | FMP plan limits for statements on the whole dividend universe | — | D.1 |
| D8 | A direct payment gateway (QRIS / e-wallet). The user opens the merchant account and holds the keys | Later, after Phase C | F.4 |
| D9 | Buying plan candidates: current holdings only, or also stocks the user does not own | Holdings by default, with an opt-in toggle that adds the top 10 of the screener's Income ranking. The ranking code lives in the page module `apps/screener/stock_picker.py` (`get_processed_df`); reuse it, do not copy the scoring | C.4 |
| D10 | Income goal basis: before or after the 10% final tax. The KPI card today compares the gross trailing dividend with the goal, while the `income_goal_idr` column is described as after tax | After tax everywhere, with the KPI card switched and labelled | C.4 |

## What exists (read before designing anything new)

- **Gate**: `harvest/premium.py` — `has_access()`, `is_premium(email)`, `current_email()`,
  `require_premium(teaser=None)`, `get_connection()`. `PREMIUM_ENFORCED` env flag; while it is
  off, `has_access()` is true for everyone. Tiers and donation parsing in `harvest/donations.py`.
- **Trading store**: `harvest/trading_store.py` — runs, setups, signals, outcomes, positions,
  `user_settings` (`get_settings`, `save_settings`, `SETTINGS_DEFAULTS`), `alert_log`
  (`log_alert`, `forget_alert`), `alert_subscribers`, IDX calendar (`idx_trading_days`,
  `sessions_until`). DDL in `pipeline/premium_trading.sql` (RLS on every table).
- **Alerts**: `harvest/alerts.py` — Telegram send, Dividend Play signal alerts, exit reminders,
  Long Term exit alerts, chat linking. Called from `pipeline/trading_scan.py`.
- **Nightly job**: `pipeline/trading_nightly.sh` — `daily_data`, three scans, `trading_track`.
  Weekly: `pipeline/trading_weekly.sh`.
- **Design kit**: `harvest/trading_ui.py` — `hero`, `section`, `card`, `cells`, `chip`,
  `progress`, `strip`, `timeline`, `empty_state`, `cta`, `unlock_prompt`.
- **Portfolio**: page `apps/porto/porto_overview.py`, maths in `harvest/portfolio.py`. A
  logged-in user's portfolio is the `portfolio` JSON column of the Supabase `users` table, keyed
  by `email`: records with `Symbol`, `Available Lot`, `Average Price` (`normalize_portfolio`).
  The yearly income target is a page widget (`key='porto_target'`), not saved anywhere.
- **Dividend data**: Redis `div_score_jkse` (scores, price, `lastDiv`, sector), Redis
  `latest_returns_jkse`, and the dividend history pickle `data/jkse/dividends.pkl` in the
  Supabase storage bucket `harvest_dividend` (`pipeline/trading_data.py::load_dividend_events`).
  The history has ex-dates and amounts only. **There is no payment date.**
- **Dividend engines to reuse**: `harvest/dividend_play.py` — `announced_dividends` (declared,
  future ex-dates), `estimate_next_ex_dates` (projected ones), `short_play_verdict`;
  `pipeline/trading_scan.py::cum_date`.
- **Fundamentals**: `harvest/position.py::quality_metrics` (debt to equity, interest cover, cash
  conversion, ROE, dilution) from FMP statements; known FMP unit errors are described in the
  first plan's Phase 4 log.
- **Tests**: `pytest` from the repo root. Engine tests in `tests/`, pipeline tests in
  `tests/pipeline/`, page AppTests in `apptests/` (run through `tests/test_apptests.py`).
  Local preview configs `harvest` (port 8599) and `harvest-premium` (gate on, port 8600) in
  `.claude/launch.json`.

## New data model

Write all DDL to a new file `pipeline/premium_features.sql`, in the style of
`pipeline/premium_trading.sql` (idempotent, RLS enabled, the `app_server_all` policy on each new
table). The user runs it. **[ASK USER]** before anything is run.

```sql
-- A.4 trial: one per email, enforced by the premium_users primary key.
alter table premium_users add column if not exists trial_started_at timestamptz;

-- C.4, F.3: goal, buying plan inputs and calendar feed live with the other settings.
alter table user_settings add column if not exists income_goal_idr numeric;   -- yearly; basis per D10
alter table user_settings add column if not exists income_goal_year int;
alter table user_settings add column if not exists goal_plan jsonb;           -- C.4 top-up and limits
alter table user_settings add column if not exists calendar_token text;

-- C.6 portfolio history: one small row per user per session, no per-stock detail.
create table if not exists portfolio_snapshots (
  email text not null,
  as_of date not null,
  invested numeric not null,
  market_value numeric not null,
  annual_dividend numeric not null,     -- trailing, before tax
  holdings int not null,
  primary key (email, as_of)
);

-- F.1 watchlists and saved screens.
create table if not exists watchlists (
  id bigserial primary key,
  email text not null,
  name text not null,
  symbols jsonb not null default '[]',  -- ['BBRI.JK', ...]
  screen jsonb,                         -- saved screener filters, null for a plain list
  created_at timestamptz default now()
);
create index if not exists watchlists_by_email on watchlists (email);
```

`alert_prefs` (jsonb in `user_settings`) grows by keys; old rows without them must keep working:

```json
{
  "watch": ["BBRI.JK"],            // exists: Dividend Play window alerts
  "new_signals": false,            // exists: every new Dividend Play window
  "digest": true,                  // B.2 nightly digest
  "swing": false,                  // B.3 swing plans for the next session
  "portfolio": true,               // C.3 cum dates, announcements and cuts on holdings
  "monthly_report": true,          // C.5
  "yield_targets": {"BBRI.JK": 6.0} // D.4 alert when the yield reaches this percent
}
```

`positions.tool` gains the value `manual` (E.1): add it to `store.TOOLS`-style validation for
positions only, not to the scan tools.

Job runs: `store.start_run` only accepts a tool in `store.TOOLS` and a kind in `store.RUN_KINDS`.
The new jobs record their runs as tool `account` with kinds `digest` (B.2) and `portfolio` (C.3).
Add a separate `RUN_TOOLS = TOOLS + ('account',)` for `start_run` / `latest_run` and leave `TOOLS`
alone, because signals, setups and outcomes must keep rejecting anything but the three trading
tools. `trading_runs.tool` and `kind` are plain text columns, so no DDL change is needed; update
the comments in the SQL.

## Phase A: Conversion foundations

Goal: a visitor can see the proof, understand what a supporter gets, try it and know when it ends.
No new data jobs.

### A.0 Non-blocking gate helper

Read first: `harvest/premium.py`, `harvest/trading_ui.py` (`unlock_prompt`, `cta`),
`tests/test_premium.py`.

- [ ] `harvest/premium.py`: `supporter_expiry(email) -> datetime | None | Literal['lifetime']`
      from the same cached `premium_users` lookup `is_premium` uses (select `expires_at`,
      `source`, `trial_started_at` once; keep the 10-minute cache and the "errors are not cached"
      behaviour).
- [ ] `harvest/trading_ui.py`: `supporter_section(key, title, body, preview=None) -> bool`.
      Returns True when `has_access()`. Otherwise renders one compact locked card (title, one
      line on what it does, optional `preview` callable drawn above it for a sample, and a button
      that opens the paywall page from A.2) and returns False. It never calls `st.stop()`.
      Usage: `if ui.supporter_section(...): render_the_feature()`.
- [ ] Tests: gate off, logged out, logged in free, supporter, expired.

### A.1 Track Record back in the navigation

Read first: `main.py`, `apps/trading/track_record.py`, `apptests/test_track_record.py`.

- [ ] Uncomment `page_track_record` in the Trading section of `main.py` (last entry, no lock icon:
      the page is public).
- [ ] `harvest/track_record.py`: `headline(outcomes) -> dict` for each tool (closed trades, wins,
      average after costs, JKSE over the same days) built on `summarize`. Return None when there
      are fewer than `MIN_HEADLINE_TRADES = 10` closed trades, so a tiny sample is never quoted.
- [ ] Link to the page from `ui.unlock_prompt` ("See the track record") and from the caption under
      each tool's Today view.
- [ ] Update `tests/live/smoke.spec.js` page list.

Done when: the page is reachable from the sidebar and from every unlock prompt.

### A.2 Supporter page and paywall rewrite

Read first: `harvest/premium.py` (`_render_paywall`), `harvest/donations.py` (`TIERS`), `home.py`
(the trading section around the "Short-Term Swing Trading Lab" card).

- [ ] New page `apps/supporter/supporter.py` ("Become a supporter", URL `/supporter`), registered
      in `main.py` under Home. It is the one place that explains the offer; `_render_paywall`
      renders the same content (move the body into a shared function so they cannot drift).
- [ ] Content, top to bottom: hero; the track record headline per tool when `headline` returns
      one (A.1), otherwise the count of cards published so far; benefits in two groups (below);
      tier cards; the three unlock steps; login / Trakteer / Buy Me a Coffee buttons; trial button
      (A.4); help footnote. When logged in, "Logged in as <email>" stays.
- [ ] Benefits, generated from one list in `harvest/premium.py` so later phases append to it. Only
      list a benefit once its feature is live (a `live: bool` field per item):
      - Trading: every card each night; evidence for any stock; the research lab; the journal with
        sell-by countdowns; Telegram alerts and the nightly digest (Phase B).
      - Dividends: paycheck forecast, cum-date and announcement alerts on your holdings, a buying
        plan towards your income goal, monthly income report, portfolio history (Phase C); safety
        grades and reasons (Phase D).
- [ ] **[ASK USER]** D3 before changing `TIERS`.
- The homepage is task A.6. It replaces the single call-to-action that was planned here.
- [ ] AppTests for the page in the three states. Check desktop and mobile in the preview.

### A.3 Supporter status and renewal

Read first: `main.py` (sidebar block at the bottom), `apps/trading/my_trades.py` (`render_journal`).

- [ ] Sidebar, above the donation buttons, only when logged in and the gate is on:
      "Supporter until 12 Jan 2027" / "Lifetime supporter" / "Trial ends in 3 days" / nothing for
      free users. Within 7 days of expiry the line becomes a warning with a "Renew" link to
      `/supporter`.
- [ ] The same line in the Settings section of My Trades.
- [ ] After expiry, the first trading page the user opens shows a one-time (per session) notice:
      "Your supporter access ended on <date>. Your journal and settings are kept."
- [ ] Confirm that nothing is deleted on expiry (positions, settings, snapshots stay).
- [ ] Tests for each state of the line.

### A.4 Free trial

**[ASK USER]** D2 first.

- [ ] `harvest/donations.py`: `start_trial(client, email, days=7, now=None) -> dict`. Inserts a
      `premium_users` row with `source='trial'`, `trial_started_at=now`, `expires_at=now+days`
      only when the email has **no row at all**. Any existing row (active, expired, trial used)
      returns `{'status': 'not_eligible'}`. A later donation extends from the later of now and the
      trial end through the existing `extended_expiry`; make sure `record_donation` keeps
      `trial_started_at` and sets `source` to the provider.
- [ ] "Start 7-day free trial" button on the supporter page and the paywall, shown only to
      logged-in users who are eligible. After it, clear the `_premium_rows` cache for that email
      and rerun.
- [ ] Tests: eligible, already supporter, expired supporter, trial used, donation during a trial.

### A.5 Free tier: delayed cards

**[ASK USER]** D1 first. This replaces "one full Today card per tool" in the first plan.

Read first: the `render_today` function of each tool page, `my_trades.py` ("Act now"),
`store.read_signals`, `store.scan_dates`, the free-tier AppTests in `apptests/`.

- [ ] Non-supporters see the **previous session's** actionable cards in full, labelled
      "Published <day>. Supporters saw this the night it was published", each with its result so
      far when `trading_outcomes` has a row for it (filled or not, return to the latest close).
      Tonight's cards show as a count plus the unlock prompt; no tickers of tonight's cards.
- [ ] Long Term has no daily turnover: keep its current rule (first candidate in full).
- [ ] Implement once in `harvest/trading_ui.py` or a small `harvest/free_tier.py`
      (`visible_signals(tool, access, client)`), used by the three tool pages and My Trades, so the
      rule cannot differ between pages.
- [ ] Track Record keeps its rule (open trades are names for supporters only).
- [ ] Update the free-tier AppTests.

### A.6 Homepage: show the supporter offer

Do this last in Phase A. It links to the pages and reuses the functions of A.0 to A.5.

The homepage stays a page about the free tools. The supporter offer is a second layer on it,
and every supporter link goes to `/supporter`. The design is fixed by the mockup
`docs/home-redesign/home-mockup.html`: open it with the preview config `home-mockup`
(port 8602, path `/docs/home-redesign/home-mockup.html`) or straight from disk. The bar at its
top switches between the four states below, and its "Design notes" tags repeat the rules of this
task. **Every number in the mockup is sample data.**

Read first: `home.py` (one `st.html` f-string plus the click router at the end), the mockup,
`main.py` (page titles and URLs), `harvest/premium.py`, `harvest/donations.py` (`TIERS`),
`harvest/track_record.py`, `apps/trading/my_trades.py` ("Act now" and "Coming up").

States, decided once at the top of `home.py`:

| State | Condition | Mockup button |
|---|---|---|
| Gate off | `not is_enforced()` | Gate off |
| Visitor | gate on and `not has_access()` (logged out, or logged in without access) | Visitor |
| Supporter | gate on and `has_access()` | Supporter |
| Renewal due | supporter whose `supporter_expiry` (A.0) is within 7 days; never for lifetime rows | Renewal due |

Use the same 7-day constant as the sidebar line of A.3, so the two cannot disagree.

- [ ] Split the `home.py` f-string into one function per section, each returning HTML, and
      assemble the page from the state. Every dynamic value goes through `html.escape`. The CSS
      and the unchanged sections stay as they are.
- [ ] **Gate off renders today's page**: the "Latest tools" section, the tagline in the masthead,
      the Trakteer "Support" link, and no supporter content anywhere. Two differences from today:
      the tool index below, and the context strip says "12 tools that work together".
- [ ] Masthead, right side: "Become a supporter" pill (visitor), "Supporter until <date>" or
      "Lifetime supporter" chip (supporter), amber "Supporter access ends in N days · Renew"
      (renewal due). The pill stays visible on mobile.
- [ ] Hero: one line under the buttons. Visitor: "Free and ad-free. Supporters get nightly trade
      cards, alerts and a trade journal", linking to `/supporter`. Supporter: "Welcome back. N new
      cards were published after <day>'s close", linking to `/my_trades`; left out when N is 0.
- [ ] Context strip: third item becomes "Free to use, funded by supporters".
- [ ] Supporter band (visitor), in place of "Latest tools":
  - Benefit rows come from the benefits list of A.2 and its `live` flags. The Telegram row
    appears only once Phase B is live.
  - Buttons: "Start N-day free trial" (N from the A.4 constant) and "See supporter plans". A
    logged-in visitor who is not eligible for a trial gets "See supporter plans" only.
  - Sample card: the first card of the previous session from `visible_signals` (A.5), in the
    fixed tool order Short Term Swing, Dividend Play, with its result so far and the label
    "Published <day>. Supporters saw this the night it was published". Below it, the count of
    cards from the latest session with no tickers. No previous card: leave the panel out and let
    the text column take the full width.
  - Track record strip: `headline(outcomes)` (A.1) for each tool. A tool under
    `MIN_HEADLINE_TRADES` shows "N closed trades so far. Results are shown from 10." A negative
    average is shown like a positive one. Link to `/track_record`.
- [ ] "Tonight for you" block (supporter, renewal due), in place of the band: new cards by tool,
      the supporter's positions to sell within two sessions, and the next date from the "Coming
      up" events of My Trades. Put the counting in a function outside the page module, so the
      nightly digest (B.2) reuses it and the two always agree. A tile with nothing to say is left
      out. Renewal due adds the notice "Your supporter access ends on <date>. Your journal and
      settings are kept." with a "Renew" link.
- [ ] Workflow step 3: the one-line "Supporters" extra appears only once the paycheck forecast
      and holding alerts (C.2, C.3) are marked live in the benefits list. Until then, nothing.
- [ ] Tool index: a "Trading" group replaces "Strategy research", with My Trades, Dividend Play,
      Short Term Swing, Long Term Position and Track Record, using the titles and URLs of
      `main.py`. The three gated pages carry a "Supporter" pill for visitors only.
- [ ] "Free to use. Better with support." section (visitor only): the two lists, and one price
      line built from the cheapest monthly and the yearly entry of `TIERS`. **[ASK USER]** D3
      before the price line ships; until then show the section without it.
- [ ] Closing block: second button "Become a supporter" (visitor only). Footer: "Support" goes
      to `/supporter` when the gate is on.
- [ ] `st.html` cannot run a login or start a trial, so every supporter button is a link to
      `/supporter`. The click router needs `/supporter` and `/track_record` in the sidebar
      navigation (A.1, A.2); check that both navigate without a full reload.
- [ ] **The homepage must never fail or slow down because of the store.** Today it reads nothing.
      Read the sample card, the headlines and the "Tonight" numbers through one cached function
      (`st.cache_data`, 10 minutes; the supporter's own positions keyed by email). On any error,
      log it and render the page without that panel.
- [ ] Copy as in the mockup. No returns promises, no "buy", "best" or "recommended", no
      countdown timers, no popups. Keep the line "Research and record keeping, not financial
      advice. Past results do not predict future ones." under the track record strip.
- [ ] AppTests for the four states, plus a visitor run where the store raises. Preview at
      desktop and mobile width with `harvest` (gate off) and `harvest-premium`.
- The redesign uses the existing screenshots in `static/home/`; none need regenerating.

Done when: the four states match the mockup with live data, a logged-out visitor reaches
`/supporter` and `/track_record` from the homepage without a reload, and the page still renders
when Supabase is unreachable.

Done when (Phase A): with `PREMIUM_ENFORCED=true`, a logged-out visitor can open Track Record and
the supporter page, a new Google account can start a trial and sees its end date, an expired
account sees the renewal notice, and the homepage shows the supporter offer to visitors and
"Tonight for you" to supporters.

## Phase B: Alerts and the nightly digest

Goal: a supporter hears from the site every trading night without opening it.

### B.1 Switch Telegram on

**[ASK USER]** D5 (create the bot, set the three secrets, run `pipeline/telegram_setup.py`), D4.

Read first: `harvest/alerts.py`, `harvest/webhooks.py` (`telegram_webhook`),
`tests/test_donations_webhooks.py`, `apps/trading/my_trades.py` (`render_telegram`).

- [ ] Verify the existing flow end to end with a test chat: link, `/stop`, a Dividend Play signal
      alert, an exit reminder. Fix what the dry run finds. Tick the Telegram item in the first plan.
- [ ] Move the Telegram card out of the journal in My Trades into a shared
      `ui`-level function so the portfolio page (Phase C) can show the same card.
- [ ] Bot replies to `/start` without a token and to unknown text with a short help message.

### B.2 Nightly digest

- [ ] `harvest/alerts.py`: `digest_message(as_of, tools: dict, positions, expiry) -> str | None`.
      One message per supporter: per tool, the number of new cards and up to three tickers with the
      one number that matters (entry limit, or "buy at open"); the supporter's own positions to
      sell within two sessions; one closing line with the link to `/my_trades`. Returns None when
      there is nothing to say, so no empty messages go out.
- [ ] `send_digests(client, as_of)`: for subscribers with `alert_prefs.digest` (default on for
      supporters), dedupe with `alert_log` kind `digest`, ref `as_of`. Non-supporters do not get the
      digest (they keep exit reminders, as today).
- [ ] New CLI `python -m pipeline.trading_digest`, run by `pipeline/trading_nightly.sh` after
      `trading_track`. It records a `trading_runs` row (tool `account`, kind `digest`; see
      "Job runs" under the data model).
- [ ] The per-signal Dividend Play alerts stay for watched stocks. When a user has the digest on,
      the "every new entry window" alert is folded into the digest instead of sent separately.
- [ ] Settings toggles in My Trades for each `alert_prefs` key that is live.
- [ ] Tests with a fake `send`: content, nothing-to-say, dedupe, supporter filter, failure isolation.

### B.3 Swing alerts and one watchlist for every tool

- [ ] `alerts.send_swing_alerts` called from `trading_scan.swing`: tomorrow's plans to users with
      `alert_prefs.swing`, one message listing all plans (they are valid for one session only).
      Ref `swing:<as_of>`.
- [ ] "Alert me" works on any stock from any tool page: the `watch` list becomes tool-agnostic.
      Long Term: alert when a watched company enters the screen.
- [ ] Tests.

### B.4 Expiry reminders

- [ ] In `pipeline/trading_digest`: a Telegram reminder 7 days and 1 day before `expires_at`
      (kind `expiry`, ref `<expires_at date>:<days>`), with the link to `/supporter`. Never for
      lifetime rows. Trials get one reminder, 2 days before the end.

Done when (Phase B): a test supporter receives the digest after a nightly run, a swing alert on a
night with plans, and an expiry reminder; a rerun of the same night sends nothing twice.

## Phase C: Portfolio autopilot (dividend pages)

Goal: the supporter's own holdings drive a forecast, reminders and a monthly report. This is the
main reason for a dividend investor to donate.

### C.1 Income forecast engine

Read first: `harvest/portfolio.py`, `harvest/dividend_play.py` (`prepare_dividends`,
`announced_dividends`, `estimate_next_ex_dates`), `tests/test_portfolio.py`.

- [ ] New `harvest/income.py` (no Streamlit, no network):
  - `forecast_income(holdings, events, as_of, months=12, tax_pct=10.0) -> DataFrame` with one row
    per expected dividend: `symbol`, `ex_date`, `cum_date`, `dividend` (per share), `lots`,
    `gross`, `net`, `status` (`announced` | `estimated`). Announced rows come from future ex-dates
    already in the history; estimated rows from `estimate_next_ex_dates` using last year's amount
    for the same period. An estimate within 60 days of an announced one is dropped (same rule as
    the Dividend Play scan).
  - `monthly_income(forecast) -> DataFrame` (month, announced, estimated, net).
  - `received_income(holdings, events, start, end)` for past months (assumes today's lots, and
    says so in the page copy, because the portfolio has no purchase dates).
  - `income_goal(forecast_net_year, goal_idr, goal_year, growth_pct, monthly_topup)` returning
    progress, the gap and the monthly top-up that closes it at the portfolio's current yield.
- [ ] The history has no payment dates. Show ex-dates and cum dates, and label the month as
      "ex-date month". Do not invent payment dates. **[ASK USER]** whether to add them later from
      another source (IDX announcements).
- [ ] Unit tests with fixed holdings and dividend frames: announced beats estimated, a stock whose
      last dividend is over `dp.MAX_SCHEDULE_AGE_DAYS` (550) days old gets no estimate, tax, month buckets, symbol suffix handling.

### C.2 "Dividend paycheck" section on Portfolio Analytics

Read first: `apps/porto/porto_overview.py` (the "Dividend timeline" and "Compounding projection"
expanders), `tests/apps/test_porto_overview.py`.

- [ ] New expander "Dividend paycheck, next 12 months" behind `ui.supporter_section` (A.0). The
      existing timeline and every other section stay free and unchanged.
- [ ] Supporters see: next payout card (stock, cum date, amount after tax, sessions left); a
      12-month bar chart split into announced and estimated; the table of upcoming dividends; a
      "Recently passed ex-dates" list for the last 60 days.
- [ ] Free preview: the next single upcoming dividend in full, the rest as a count
      ("7 more dividends expected in the next 12 months").
- [ ] Charts follow `harvest/chart_style.py`; load the `dataviz` skill before writing chart code.
- [ ] AppTests: supporter, free, logged out, portfolio with no dividend payers.

### C.3 Alerts on holdings

**[ASK USER]** D6 first.

Read first: `harvest/alerts.py`, `pipeline/trading_data.py`, `pipeline/trading_nightly.sh`.

- [ ] `harvest/trading_store.py`: `portfolios_all(client) -> dict[email, DataFrame]` reading
      `users.portfolio` through `hpf.normalize_portfolio`; a broken portfolio is skipped and counted,
      never raised.
- [ ] `harvest/alerts.py`: `send_portfolio_alerts(client, as_of, events)` for supporters with a
      linked chat and `alert_prefs.portfolio` (default on):
  - **Announcement**: a holding has a newly declared dividend. Message: amount per share, cum
    date, what the user receives after tax for their lots, change against the same period last
    year. Ref `announce:<symbol>:<ex_date>`.
  - **Cum-date reminder**: one message 3 sessions before the cum date and one on the last session
    ("last day to buy more and still qualify"). Ref `cum:<symbol>:<ex_date>:<sessions>`.
  - **Lower dividend**: the announced amount is more than 10% below the same period last year.
    Folded into the announcement message, not a second message.
  - Several events for one user on one night go out as one message.
- [ ] New CLI `python -m pipeline.portfolio_alerts`, in the nightly script after `daily_data`
      (it does not need the scans). Records a `trading_runs` row (tool `account`, kind
      `portfolio`).
- [ ] The Telegram card and an alerts toggle on Portfolio Analytics (shared function from B.1).
- [ ] Tests with a fake client and fake `send`.

### C.4 Income goal and buying plan

When the portfolio earns less than the goal, this section shows how to close the gap: how much to
invest, which stocks add the most income within the user's own limits, how many lots of each, and
how much of the cost comes from reinvested dividends and how much from new capital. It replaces
the former F.5.

**[ASK USER]** D9 and D10 first.

Read first: `apps/porto/porto_overview.py` (the "Income goal" popover and the "Compounding
projection" expander), `harvest/portfolio.py` (`enrich_holdings`, `summarize_portfolio`,
`project_income`), `harvest/income.py` from C.1, `harvest/trading_store.py` (`get_settings`,
`save_settings`, `SETTINGS_DEFAULTS`).

Why it is a plan for one year and not for the whole gap: the gap is usually far larger than one
purchase (Rp 80 million of missing income at a 6% yield is about Rp 1.3 billion of stock). So the
section answers "what do I buy with the next 12 months of cash, and when does that get me there".

- [ ] Save the yearly target: the `porto_target` widget reads and writes
      `user_settings.income_goal_idr` for logged-in users (free feature, as today), with
      `income_goal_year` as a new optional input. One tax basis everywhere, as decided in D10.
- [ ] Save the plan inputs for logged-in supporters in `user_settings.goal_plan` (jsonb):
      `monthly_topup_idr`, `max_stock_pct` (default 20), `max_sector_pct` (default 40),
      `include_new` (default false). Add the key to `SETTINGS_DEFAULTS`; rows without it must
      keep working.
- [ ] `harvest/income.py`: `plan_purchases(holdings, candidates, reinvest_idr, new_capital_idr,
      max_stock_pct, max_sector_pct, tax_pct=10.0) -> tuple[DataFrame, dict]`. No Streamlit, no
      network.
  - `holdings` is `enrich_holdings` output. `candidates` has one row per stock that may be
    bought (`symbol`, `price`, dividend per share, `sector`, `score`): the current holdings, plus
    new names when D9 allows them.
  - Budget = `reinvest_idr + new_capital_idr`. Buy one whole lot (100 shares) at a time of the
    stock with the highest dividend per rupiah that still fits the remaining budget and keeps its
    stock and its sector at or under the limits. The limits are measured against the market
    value after the whole budget is spent. Ties go to the higher `score`. Stop when no lot fits.
  - A stock with no trailing dividend, no price, or a last dividend older than
    `dp.MAX_SCHEDULE_AGE_DAYS` is never bought. A holding already over a limit gets no lots. The
    function never proposes a sale.
  - Rows: `symbol`, `lots`, `price`, `cost`, `from_dividends`, `from_new_capital`,
    `added_income_net` (per year), `weight_after_pct`, `is_new`. Dividends are spent first, in
    row order, then new capital; the two columns always add up to `cost`.
  - Summary dict: cash left over, income after the purchases, progress and gap after them, and
    the stocks skipped because of a limit (so the page can say why a high yielder is missing).
- [ ] Supporter section "Path to your goal" on Portfolio Analytics, behind
      `ui.supporter_section` (A.0):
  - Below the goal: progress, the gap, and the capital that closes it at the portfolio's current
    yield.
  - Inputs: monthly top-up, the two limits, and the D9 toggle if it exists.
  - "The next 12 months": reinvestment budget = net dividends expected in the next 12 months
    (`income.forecast_income`, announced and estimated shown separately); new capital = 12 × the
    monthly top-up. Then the `plan_purchases` table and one line with the income and progress
    after it.
  - The path: a line chart of projected income against the goal for "reinvest only" and
    "reinvest plus top-up", with the year each one reaches the goal, and the monthly top-up
    needed to reach it by `income_goal_year` when that is set (`income.income_goal`). Reuse
    `hpf.project_income`; do not write a second projection.
  - At or above the goal: one line saying so. The plan table stays available as "Add more
    income".
- [ ] Free preview: the gap and the capital needed, in full. The table is locked, with a count
      ("5 stocks in your plan"). Logged-out visitors with a pasted portfolio see the same preview.
- [ ] Copy: this is arithmetic on trailing dividends and the user's own limits. No "buy",
      "recommended" or "best" in headings, buttons or column names (use "Lots", "Cost", "Added
      income"). A caption says trailing dividends can be cut and that this is not advice.
- [ ] Charts follow `harvest/chart_style.py`; load the `dataviz` skill before writing chart code.
- [ ] Unit tests for `plan_purchases`: the budget is never exceeded, whole lots only, both limits
      hold, dividends are spent before new capital, a holding over its limit gets nothing, zero
      budget, no dividend payers, new names appear only when passed in, symbol suffix handling.
- [ ] AppTests: supporter below the goal, supporter at the goal, free, logged out.
- [ ] After D.1: add "minimum safety grade" as a third limit, default `B`, with stocks graded
      `None` excluded and counted in the skipped list.

### C.5 Monthly income report

- [ ] `alerts.monthly_report_message(...)`: dividends with an ex-date last month on the user's
      holdings (after tax), the 12-month forecast, yield on cost, what is expected next month, and
      the days left of supporter access when under 30.
- [ ] Sent by `pipeline.portfolio_alerts` on the first IDX session of a month, kind `report`,
      ref `<yyyy-mm>`.
- [ ] Tests.

### C.6 Portfolio history

- [ ] `pipeline.portfolio_alerts` also writes one `portfolio_snapshots` row per saved portfolio
      per session (for **every** user with a saved portfolio, so the history is already there on
      the day someone becomes a supporter). Uses the same `enrich_holdings` / `summarize_portfolio`
      maths as the page, with prices from `div_score_jkse`.
- [ ] Supporter section "Your portfolio over time" on Portfolio Analytics: market value against
      invested, and trailing annual dividend, with a range selector. Free users see the locked
      card with "History recorded since <first snapshot date>".
- [ ] Tests for the job (one row per session, rerun replaces the day) and the section.

Done when (Phase C): with a saved test portfolio, the paycheck section matches a hand-checked
forecast, the buying plan matches a hand-checked allocation and stays inside both limits, an
announced dividend on a holding produces one Telegram message, and two nightly runs leave two
snapshot rows.

## Phase D: Dividend safety and yield signals

**[ASK USER]** D7 before D.1. If FMP limits rule out the whole universe, restrict to the
companies the Long Term scan already analyzes plus every stock held in a saved portfolio.

### D.1 Safety grade

Read first: `harvest/position.py` (`quality_metrics`, `prepare_statements`,
`statements_in_idr`), `pipeline/trading_data.py::load_long_term_fundamentals`,
`harvest/data.py::calc_div_stats`.

- [ ] New `harvest/safety.py`: `safety_grade(divs, statements, financials, bank) -> dict` with a
      grade (`A` to `D`, or `None` for "not enough data"), a score and the reasons as short
      sentences. Inputs: payout ratio (TTM dividends over TTM earnings), free cash flow cover,
      debt to equity and interest cover (not for banks), years paid without a gap, years without a
      cut, and a one-off flag (a special dividend more than twice the usual amount).
- [ ] Missing or inconsistent statements give `None`, never a guessed grade. Use the Daguerreo
      net income as Phase 4 did; FMP cash flow net income has unit errors.
- [ ] Weekly job (`pipeline/trading_weekly.sh`) publishes Redis `div_safety_jkse` (symbol, grade,
      score, reasons, as_of). Throttled like the Long Term scan.
- [ ] Unit tests per rule, including a bank and a stock with a special dividend.

### D.2 Screener and stock page

Read first: `apps/screener/stock_picker.py`, and the memory note that Syariah/BUMN filters rerank
while other filters keep the full-list rank (do not change that).

- [ ] Ranking table: a "Safety" column with the grade for everyone. Ranks and existing columns do
      not change.
- [ ] Stock detail: the grade is free; the reasons and the underlying numbers are a
      `ui.supporter_section`.
- [ ] "Yield trap" chip on a stock whose yield is inflated by a one-off dividend or a price that
      fell more than 30% in a year. The chip is free, its explanation is for supporters.

### D.3 Safety changes on holdings

- [ ] In `pipeline.portfolio_alerts`: when a holding's grade drops by a letter between two weekly
      runs, one message with the reason that changed. Ref `safety:<symbol>:<as_of>`.

### D.4 Yield zones

- [ ] `harvest/income.py`: `yield_zone(prices, divs, years=5)`: today's trailing yield, its
      percentile in the stock's own range and the price at which the yield reaches a target.
- [ ] Stock page: a yield-band chart (free) with "Alert me at X%" (supporters), stored in
      `alert_prefs.yield_targets`.
- [ ] Nightly alert when the close puts the yield at or above the target, once per crossing
      (ref `yield:<symbol>:<target>:<yyyy-mm>`).

## Phase E: Trading journal and nightly plan

Read first: `apps/trading/my_trades.py`, `harvest/trading_store.py` (positions section),
`harvest/portfolio_sim.py`, `apptests/test_my_trades.py`.

### E.1 Journal

- [ ] Open positions show the latest close, unrealised result after the user's fees, and distance
      to stop and target. Prices from Redis `div_score_jkse` (already loaded elsewhere), with the
      price date in a caption.
- [ ] "Edit" dialog (lots, entry price, entry date, stop, target) using `store.update_position`.
- [ ] "Add a trade" for trades that did not come from a card: `tool='manual'`, optional stop,
      target and sell-by date.
- [ ] Partial close: closing fewer lots than held splits the position (closed part as its own row).
- [ ] Closed trades: CSV download, result by tool, average win and loss, a cumulative result chart.
- [ ] Tests in the store for the split and the `manual` tool; AppTests for the new controls.

### E.2 Tonight's plan for my capital

- [ ] `harvest/portfolio_sim.py`: `plan_tonight(signals, open_positions, capital, max_positions,
      liquidity) -> DataFrame`: which of tonight's cards fit the free slots and cash, in what size
      (lots, IDR), and why the others are skipped (slots full, already held, too small, liquidity
      cap). Same rules as `simulate_portfolio`, so the plan matches the published simulation.
- [ ] Section in My Trades above "Act now" for supporters who saved a capital figure; otherwise a
      prompt to set it in Settings. `max_positions` becomes a user setting (default 5).
- [ ] Tests with fixed signals.

### E.3 You against the cards

- [ ] For closed positions that came from a card (`exit_rule.signal_as_of`), join to
      `trading_outcomes` and show the user's result next to the replayed one, plus two discipline
      numbers: trades sold by the sell-by date, and average result of those sold late.

## Phase F: Convenience and payments

- [ ] **F.1 Watchlists and saved screens.** `watchlists` table, store functions, a picker on the
      screener ("Save this screen"), and a nightly message when a new stock passes a saved screen.
      Free: one list of up to 10 stocks. Supporters: unlimited, with alerts.
- [ ] **F.2 Exports.** Excel export of holdings, forecast and a yearly dividend summary with the
      10% final tax withheld per stock (for SPT reporting). Supporter section on Portfolio
      Analytics. Label it as a summary from the site's data, to be checked against broker statements.
- [ ] **F.3 Calendar feed.** `GET /calendar/<token>.ics` in `server.py` (Starlette route, like the
      webhooks): the user's cum dates, ex-dates and sell-by dates. `calendar_token` is created on
      demand in settings and can be regenerated. The route returns 404 for unknown tokens and for
      expired supporters, and never includes the email.
- [ ] **F.4 Direct payment.** **[ASK USER]** D8. Only after the user has a merchant account:
      a checkout link created for the logged-in email, and `POST /webhooks/<gateway>` that verifies
      the signature and calls `record_donation`, exactly like the BMC route. The agent never
      handles live keys or enters payment details; the user sets the secrets.
- **F.5 "Where should my next Rp X go?"** Merged into C.4 on 2026-11-15. Nothing to do here.

## Suggested order

A (all) → B.1, B.2 → C.1, C.2, C.3 → B.3, B.4 → C.4 to C.6 → E.1 → D → E.2, E.3 → F.

Within A, the homepage (A.6) comes last. A and B need no new data. C.1 to C.3 are the first dividend features and share one piece of work
(holdings joined to the dividend history). D depends on FMP limits. F.4 depends on the user.

## Testing checklist (every phase)

- `pytest` passes, including `tests/test_apptests.py`.
- New engine code has unit tests; new jobs have tests with `tests/fake_supabase.py` and a fake
  `send`.
- Pages checked in the local preview at desktop and mobile width, in the logged-out, free and
  supporter states, with `harvest-premium` for the gated states.
- Every free page still renders completely for a logged-out visitor.

## Risks

- **Regulatory**: alerts that say "buy" read as advice. Keep messages factual (dates, amounts,
  published rules) and keep the disclaimers.
- **Portfolio accuracy**: portfolios have no purchase dates, so "received" figures assume today's
  lots. Say so wherever they appear.
- **No payment dates** in the dividend data. Do not show a payment date until a source exists.
- **FMP statement quality** for IDX companies limits Phase D; prefer "not enough data" over a
  wrong grade.
- **Supabase free tier**: `portfolio_snapshots` grows by one row per saved portfolio per session.
  Keep the row small, and add a yearly roll-up if it nears the limit.
- **Telegram only**: users without Telegram get none of Phase B or C alerts until email exists (D4).
- **Trial abuse**: one trial per Google account is easy to repeat with new accounts. Acceptable at
  this size; revisit if trial starts far exceed donations.

## Progress log

- 2026-11-12: plan written from a code review. Nothing implemented.
- 2026-11-15: C.4 rewritten as "Income goal and buying plan" and F.5 merged into it. Decisions D9
  and D10 added. Nothing implemented.
- 2026-11-15: A.6 "Homepage: show the supporter offer" added, with the static mockup
  `docs/home-redesign/home-mockup.html` and the preview config `home-mockup`. The homepage line
  of A.2 now points to A.6. Nothing implemented.
