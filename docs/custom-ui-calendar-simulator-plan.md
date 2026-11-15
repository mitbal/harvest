# Custom UI for the Monthly Dividend Calendar and the Compounding Simulator — Implementation Plan

Status: approved direction, ready for implementation. Last updated 12 Nov 2026.

This document is written to be executed by a coding agent in a fresh session with no prior
conversation context. Everything needed to start is here. Work **one phase per session**.

## How to use this document (for the agent)

Kickoff prompt the user will give you, for example:

> Read `docs/custom-ui-calendar-simulator-plan.md` and implement Phase 1. Follow the ground rules.

1. Read this whole file first, then the files listed under the phase you are given.
2. Do only the requested phase. Do not start the next phase.
3. Tick the checkboxes in this file as tasks are completed, and add a short note under
   "Progress log" at the bottom (date, what was done, anything left open).
4. Stop and ask the user at every point marked **[ASK USER]**.

## Goal and scope

Move two pages from the native Streamlit look to the custom look the user approved:

- **Payout Calendar, Single Month view** (`apps/calendar/calendar.py`, URL `/calendar?view=monthly`)
- **Compounding Simulator** (`apps/simulator/simulator.py`, URL `/simulator`)

Out of scope: the app shell (sidebar, top bar, footer), every other page, the calendar's
Full Year view (it only inherits the shared styles), and any change to data loading,
pipelines or Redis keys.

## Design reference

The approved mock-ups are saved in the repo as plain markup. They are a visual spec, not code
to copy: they use a design-tool runtime and sample numbers.

| File | What it shows |
| --- | --- |
| `docs/custom-ui/reference/Calendar.dc.html` | Single Month calendar page |
| `docs/custom-ui/reference/Simulator.dc.html` | Compounding Simulator page |
| `docs/custom-ui/reference/ui.css` | Tokens and component styles both mock-ups use |

Ignore the `<nav class="rail">` block and the `<header class="top">` tools in the mock-ups;
they belong to the shell, which is out of scope. Everything inside `<main class="main">`
below the header is in scope.

The mock-ups differ from what this plan asks for in a few places. Where they disagree,
**this plan wins**:

- Calendar yield colours: the mock-up shows four buckets. Keep the existing seven-step scale
  (`hp.DIVIDEND_CALENDAR_COLORS` with `hp.dividend_calendar_domain(sl)`).
- Calendar note "Announced dates are outlined": the calendar has no announced-date data. Do
  not build the outline or the note.
- Simulator "Monthly top-up" input: not an existing feature. It is Phase 4, optional.
- Simulator "Run simulation" button on Fixed yield: not needed, that simulation is instant.

## Ground rules

- Start a new branch from `master` unless the user says otherwise. Commit only when asked.
- Do not edit `harvest/trading_ui.py` or anything under `apps/trading/`. They have separate
  in-flight work. The new shared styles go in a new module (Phase 0).
- Keep every pure helper in `apps/calendar/calendar.py` with its current name and signature
  (`prep_div_table`, `project_to_next_year`, `add_yield`, `parse_query_params`,
  `yield_summary`, `payout_frequency`, and the rest). `tests/apps/test_calendar.py` imports
  the page module and calls them.
- Keep the page headings as real `h1` text containing "Dividend Calendar" and
  "Compounding Simulator". `tests/live/smoke.spec.js` looks for them.
- Keep the calendar URL parameters (`view`, `year`, `month`) working exactly as now.
- New HTML goes through `st.html`. Escape every value that comes from data (`html.escape`).
- No emoji as icons. Use Material icons (`:material/...:`) on Streamlit widgets and plain
  text elsewhere.
- Custom CSS may target your own keyed containers (`st.container(key='pdx-...')` gives the
  class `st-key-pdx-...`). Avoid Streamlit's internal `data-testid` selectors except
  `stMetric*` and `stMainBlockContainer`, which the app already relies on.
- Text contrast at least 4.5:1. On calendar chips, pick the text colour with
  `hp.readable_text_color(background)`.
- Every layout must work at phone width (about 380px): rows wrap, nothing scrolls sideways
  except tables.
- Run before finishing each phase: `python -m pytest tests apptests -q`. Report failures
  with their output; do not weaken existing tests to make them pass.
- Verify in the browser with the dev server (`make run`), at desktop and phone width, for
  both JKSE and S&P500 on the calendar.

## Phase 0 — Shared page styles

Files: new `harvest/page_ui.py`, new `tests/test_page_ui.py`.

- [x] Create `harvest/page_ui.py` with no imports from other `harvest` modules except
      `chart_style`. It holds:
  - `TOKENS`: the colours from `docs/custom-ui/reference/ui.css` (canvas `#eff1ef`, ink
    `#10251d`, muted `#5c6d66`, line `#dbe3df`, forest `#073f2e`, deep `#052d22`, lime
    `#d7f35a`, mint `#e3f1eb`, up `#15803d`, down `#c2410c`, amber `#b45309`).
  - `STYLES`: one `<style>` block using a `pdx-` class prefix, covering only what these two
    pages need: page font (Manrope on the main block, loaded the same way `home.py` does),
    card, metric tile, chip (plain, up, down, warn, brand, lime), note text, and the calendar
    grid classes from the reference CSS (`.cal`, `.dow`, `.day`, `.ev`).
  - `inject_styles()`, `esc(value)`, `render(markup)`, `chip(text, tone='')`,
    `kpi(label, value, note='', tone='')` returning markup, and `kpi_row(items)` rendering a
    wrapping grid of tiles.
  - `card(key)`: returns `st.container(key=f'pdx-card-{key}')`, styled as a white card with
    a 16px radius and a 1px `#dbe3df` border, so native widgets and charts can sit inside it.
- [x] Unit tests for the pure functions: `esc` escapes `<`, `&` and quotes; `chip` and `kpi`
      escape their text and emit the tone class.
- [x] Do not inject these styles from `main.py`. Each of the two pages calls
      `inject_styles()` itself so no other page changes.

Done when: both pages can import the module, the tests pass, and no other page looks
different.

## Phase 1 — Calendar, Single Month view

Files: `apps/calendar/calendar.py`, new `harvest/calendar_grid.py`,
new `tests/test_calendar_grid.py`, `tests/apps/test_calendar.py` (add only),
`tests/apps/conftest.py` (only if the Streamlit stub lacks a call you add).

Read first: `apps/calendar/calendar.py` in full, `harvest/plot.py` lines 935 to 1110
(`month_plot`, the colour helpers, `plot_dividend_month`), and the reference mock-up.

### 1.1 Month grid as HTML

- [ ] New pure module `harvest/calendar_grid.py` (pandas and the standard library only, plus
      colour helpers passed in or imported from `harvest.plot`):
  - `month_cells(df, year, month, closed_dates)` returns a list of weeks, each a list of
    seven day dicts: `day` (int or `None` for padding), `closed` (weekend or in
    `closed_dates`), `holiday` (name or `''`), and `events` (that day's rows sorted by yield,
    highest first, each with `symbol`, `yield`, `dividend`).
  - `month_grid_html(cells, sl, market, site_url, max_events=3)` returns the grid markup:
    a weekday header row, one cell per day, one chip per event.
- [ ] Week start: use the same first weekday the current chart shows, so the page does not
      flip between views. Check `hp.month_plot` output before choosing.
- [ ] Chip: ticker without the `.JK` suffix, background from the existing seven-step scale
      (clamp the yield to the domain first, as `hp._color_values` does), text colour from
      `hp.readable_text_color`. Each chip is a link to
      `{SITE_URL}/stock_picker?market={market}&stock={symbol}` with a `title` of
      `"12 May: PTBA (11.20%), dividend 296"`.
- [ ] More events than `max_events` on one day: show the top ones by yield, then a
      "+N more" chip whose `title` lists the rest. S&P 500 months have many payers per day,
      so this path is the common case there.
- [ ] Closed days: date number in the down colour; holiday name in the cell's `title`.
- [ ] Phone width: the grid keeps seven columns; chips shrink to the ticker only and cells
      get a smaller minimum height. If it still does not fit at 380px, let the grid scroll
      sideways inside its card rather than the page.
- [ ] Legend: an HTML row of the seven swatches with their breakpoints (last one with "+"),
      replacing `hp.plot_dividend_calendar_legend` in this view only.
- [ ] Tests in `tests/test_calendar_grid.py`: padding days before the 1st and after the
      last day; a weekend and a named holiday are `closed`; events are sorted by yield;
      the overflow chip appears with the right count; symbols and holiday names are
      HTML-escaped; a month with no events still renders all its days.

### 1.2 Page layout

Replace the body of the `else:` branch (Single Month) and the statistics block below it for
that view. Target layout, top to bottom:

- [ ] Header: keep `st.title(f'Dividend Calendar {projected_year}')` and the
      "year is not over yet" warning. Add the one-line subtitle under the title:
      "Projected ex-dates, based on what each company paid in {selected_year}".
- [ ] Controls row inside a card: stock list, view, year, month, Syariah toggle (same
      widgets and behaviour as now), with the HTML legend at the right end.
- [ ] Two columns that wrap on narrow screens (`st.container(horizontal=True)` with
      `card()` children, or `st.columns([3, 2])`):
  - Left card: month name as heading, Previous and Next buttons, the HTML grid, and the
    projection note as small text.
  - Right column, three stacked blocks:
    1. Three metric tiles: Dividends, Average yield, Median yield (same numbers as today's
       `st.metric` row, month-scoped).
    2. Card "Ex-dates in {month}": the existing `st.dataframe(month_table, ...)` with the
       existing `column_config`. Keep it native so sorting and the stock links still work.
       Keep the "N dividends this month" caption when there are more than ten.
    3. Card "Dividends per month": a new Altair bar chart of ex-date counts for all twelve
       months from `yield_summary(df, 'month')`, with the selected month in forest
       `#073f2e` and the others in a light tint. Tooltip: month, count, average yield.
- [ ] Previous / Next: give the month selectbox a `key` and move it with `on_click`
      callbacks that change `st.session_state`. Disable Previous in January and Next in
      December (the year is a separate control). The URL `month` parameter must keep
      following the widget, and opening `?view=monthly&month=6` must still land on June.
- [ ] Below the two columns, keep in cards: the highest-yield caption, "Dividends and
      average yield per day", "Yield distribution", and "Payout frequency per year".
- [ ] Empty month: grid still renders (all days, no chips), the right column shows the
      existing "No ex-dividend dates in {month}" message instead of tiles and table, and the
      twelve-month bar chart still shows.
- [ ] Full Year view: unchanged apart from inheriting the page font and card styles. Check
      that the shareable 4x3 image still fits and looks the same.
- [ ] Remove `SINGLE_MONTH_WIDTH`, `SINGLE_MONTH_HEIGHT` and `SINGLE_MONTH_COLUMN_WIDTH` if
      nothing uses them any more. Leave `hp.plot_dividend_month` in `harvest/plot.py`; its
      tests stay.

Done when: Single Month matches the reference layout for JKSE and S&P500, Syariah filter and
URL parameters behave as before, the page works at 380px, and all tests pass.

## Phase 2 — Simulator, structure and Fixed yield

Files: `apps/simulator/simulator.py`, new `tests/apps/test_simulator_page.py` (if helpers
are extracted), `tests/apps/conftest.py` (only if needed).

Read first: `apps/simulator/simulator.py` in full, `harvest/data.py`
`simulate_simple_compounding` (around line 1185), and the reference mock-up.

- [x] Replace the three stacked bordered sections with one segmented control under the
      title: **Fixed yield**, **Single stock**, **Portfolio**. Only the selected simulation
      runs. Today all three run on every rerun, including the network fetches for
      simulations 2 and 3.
- [x] Sync the choice to the URL as `?sim=fixed|stock|portfolio`, default `fixed`, falling
      back to `fixed` on a bad value. Follow the pattern of `parse_query_params` in the
      calendar page and unit-test the parser.
- [x] Remove the page-level `<style>` block that restyles `stMetric`; use the Phase 0 tiles.
- [x] Page layout for every mode: a left "Assumptions" card with the inputs stacked, and a
      right results column. On narrow screens the assumptions card goes on top.
- [x] Fixed yield mode:
  - Inputs unchanged: initial investment (million rupiah), number of years, yield (percent),
    with the same limits and defaults. Keep the caption about what the projection ignores,
    as small text in the assumptions card.
  - Tiles: **Final asset value** with the note "{x.xx}x your capital"; **Total passive
    income** with "Over {n} years"; **Income in year {n}** with "{y.yy}% yield on cost".
    Same formulas as the current three metrics.
  - Chart "Asset value by year": stacked bars per year, bottom segment the initial capital
    (forest `#073f2e`), top segment the accumulated reinvested income, which is
    `investment - initial_value` (`#7cc4a4`). Tooltip: year, capital, reinvested income,
    total, and that year's income. Legend above the chart. Label the last bar with its total.
    This replaces the current bar-plus-line chart.
  - Table "Year by year" in a card: year, start value, income received, end value, yield on
    cost. Native `st.dataframe`.
  - Keep the CSV download with the same file name and the same columns as today.

Done when: switching modes changes the URL and only runs that mode, Fixed yield matches the
reference layout, the numbers equal the current page's for the default inputs (final value,
total income, final yield on cost), and all tests pass.

## Phase 3 — Simulator, Single stock and Portfolio

Files: `apps/simulator/simulator.py`. Do not change `harvest/simulator.py`.

- [ ] Put each mode's inputs in an `st.form` with a **Run simulation** submit button, so
      typing a ticker or editing the allocation table does not fetch on every keystroke.
      Store the submitted inputs in `st.session_state` and run once with the defaults on
      first load, so the page is never empty.
- [ ] Single stock mode:
  - Inputs unchanged: ticker, initial investment, start year, end year, DRIP toggle.
  - Tiles: Final portfolio value, Total dividend income, Yield on cost (final), Residual
    cash, plus one new tile **DRIP advantage**: final value with DRIP minus without, as an
    amount and a percentage. Both series are already returned by
    `simulate_single_stock_compounding`.
  - Chart: keep the grouped DRIP comparison bars with the dividend line, in a card, with the
    title moved out of the chart into the card heading.
  - Table, activity log (collapsed) and CSV download: unchanged content, inside cards.
- [ ] Portfolio mode: same treatment. Inputs are the allocation editor, start year, end
      year and the DRIP toggle. Tiles are the existing four plus DRIP advantage. Keep the
      per-stock stacked chart, the table, the transaction log and the CSV download.
- [ ] Errors keep their current wording and appear inside the results column:
      `SimulatorValidationError` messages as they are, and "The market data service could
      not complete this simulation." for anything else.

Done when: both modes match the reference layout, results for the default inputs equal the
current page's, a bad ticker shows the validation message without a traceback, and all
tests pass.

## Phase 4 — Optional: monthly top-up for Fixed yield

**[ASK USER]** before starting: confirm this feature is wanted, and whether top-ups are added
at the start or the end of each month.

- [ ] Add a `monthly_topup=0` parameter to `hd.simulate_simple_compounding` in
      `harvest/data.py`, with the default keeping today's results identical. Add the total
      contributed capital to each row.
- [ ] Tests in `tests/test_data.py`: zero top-up equals the current output; a known small
      case with a top-up matches a hand-computed value.
- [ ] Page: add the "Monthly top-up (million rupiah)" input, show capital in the chart as
      initial plus top-ups, base "x your capital" and yield on cost on total contributed
      capital, and update the caption that says additional deposits are not modelled.

## Phase 5 — Follow-ups

- [ ] **[ASK USER]** The home page shows screenshots of both pages
      (`static/home/dividend_monthly_calendar.webp`, `static/home/compounding_simulator.webp`).
      They will be out of date. Ask whether to regenerate them (WebP, 1600px wide, as noted
      at the top of `home.py`).
- [ ] If the user later wants the same look on more pages, move the shared styles into one
      place together with `harvest/trading_ui.py` and `home.py`. Not part of this work.

## Progress log

(Add entries here: date, phase, what was done, what is left open.)

- **15 Nov 2026, Phases 0 and 2.** Added `harvest/page_ui.py` (TOKENS, STYLES with the `pdx-`
  prefix incl. calendar grid classes, `inject_styles`, `esc`, `render`, `chip`, `kpi`,
  `kpi_row_html`, `kpi_row`, `split`, `card`) and `tests/test_page_ui.py`. Rebuilt
  `apps/simulator/simulator.py`: subtitle, one segmented control synced to
  `?sim=fixed|stock|portfolio` (`parse_sim_param`), one `render_*` function per mode so only
  the selected mode runs, Assumptions card left / results right (`page_ui.split`, a wrapping
  horizontal container with 300px / 560px flex bases, not `st.columns`), the
  `stMetric` style block removed in favour of `kpi_row` tiles. Fixed yield has the new tiles,
  the stacked "Asset value by year" chart with HTML legend and last-bar label, and the "Year by
  year" table card with the CSV download (same file name and columns). Single stock and
  Portfolio only moved into the new layout (charts and tables in cards, numbers unchanged).
  Pure helpers tested in `tests/apps/test_simulator_page.py`. Default numbers checked equal to
  the old page (IDR 222,103,896 / IDR 102,103,896 / 11.05%). After the first browser review: split layout replaced
  `st.columns` (it squeezed the Assumptions card at 800 to 900px viewports), tiles have a 175px
  minimum with 20px values and no mid-digit breaks, the allocation editor headers are
  shortened ("Ticker", "Million IDR", long wording in `help`), and the chart has padding so the
  last-bar label is not clipped. Left open: re-check those four in the browser; Phase 1 not started; Phase 3
  (forms, DRIP advantage tile, chart titles into card headings) still to do.
