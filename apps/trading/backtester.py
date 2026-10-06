import datetime
import logging

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import harvest.plot as hp
import harvest.position as pos
import harvest.trading_store as store
import harvest.trading_ui as ui
from harvest import chart_style as cs
from harvest.premium import current_email, get_connection, has_access, require_premium


VIEW_TODAY, VIEW_EVIDENCE, VIEW_LAB = ui.VIEW_TODAY, ui.VIEW_EVIDENCE, ui.VIEW_LAB
# The one stock whose evidence is open to everyone.
FREE_EVIDENCE_TICKER = "BBRI.JK"
CACHE_TTL = 6 * 60 * 60
SCAN_SCHEMA_VERSION = pos.SCAN_SCHEMA_VERSION
_finite, _format_idr = pos.finite, pos.format_idr
logger = logging.getLogger(__name__)


st.set_page_config(page_title='Long Term - Panen Dividen')
hp.enable_chart_theme()
ui.inject_styles()
ui.hero(
    "Trading · Long term position", "Growth at a Discount",
    "Profitable IDX businesses whose earnings and revenue multiples are below "
    "their own history while revenue and earnings continue to grow.", "trending_up",
)
# Initial values go through session state rather than `default`, because buttons below
# also set them; a ?view= or ?stock= in the URL still wins.
if "view" not in st.session_state:
    st.session_state["view"] = VIEW_EVIDENCE if "stock" in st.query_params else VIEW_TODAY
view = st.segmented_control(
    "View", [VIEW_TODAY, VIEW_EVIDENCE, VIEW_LAB],
    required=True, key="view", bind="query-params", label_visibility="collapsed",
)

st.sidebar.markdown("### Research method")
st.sidebar.markdown(
    "The screen compares each company with itself. P/E measures the price paid for "
    "earnings, P/S the price paid for revenue (P/B, book value, for banks), and growth "
    "confirms that a low multiple is not simply the result of a shrinking business. "
    "Quality checks on debt, cash flow, returns and dilution remove companies that are "
    "cheap for a reason."
)

st.html(
    """
    <style>
        .method-note {
            color: #36534a;
            font-size: 0.9rem;
            line-height: 1.55;
            max-width: 72ch;
        }
    </style>
    """
)


@st.cache_data(ttl=CACHE_TTL, max_entries=8, show_spinner=False)
def load_idx_universe() -> pd.DataFrame:
    return pos.load_idx_universe()


@st.cache_data(ttl=CACHE_TTL, max_entries=8, show_spinner=False)
def run_fundamental_scan(
    profile_version: str,
    _profile_records: tuple[tuple[str, tuple[tuple[str, object], ...]], ...],
    history_years: int,
    schema_version: int,
) -> pd.DataFrame:
    del profile_version, schema_version
    profiles = {symbol: dict(values) for symbol, values in _profile_records}
    return pos.scan_profiles(profiles, history_years)


@st.cache_data(ttl=CACHE_TTL, max_entries=16, show_spinner=False)
def load_stock_detail(
    symbol: str,
    profile_values: tuple,
    history_years: int,
    schema_version: int,
):
    del schema_version  # Included in the cache key to invalidate older detail schemas.
    return pos.analyze_stock(symbol, dict(profile_values), history_years, include_history=True)


@st.cache_data(ttl=10 * 60, show_spinner=False)
def _load_published() -> dict:
    client = get_connection()
    return {
        "signals": store.read_signals(client, "long_term"),
        "run": store.latest_run(client, "long_term", "scan"),
        "setups": store.read_setups(client, "long_term"),
        "stale": store.is_stale(client, "long_term"),
    }


def load_published() -> dict | None:
    """Latest published candidates, scan run and screen; None when the store is unreachable."""
    try:
        return _load_published()
    except Exception:
        logger.exception("Could not load published Long Term candidates")
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_open_positions(email: str) -> pd.DataFrame:
    positions = store.list_positions(get_connection(), email, open_only=True)
    return positions[positions["tool"] == "long_term"]


def _ratio_chart(history: pd.DataFrame, label: str, years: int):
    chart_data = history.copy()
    cutoff = chart_data["date"].max() - pd.DateOffset(years=years)
    chart_data = chart_data[chart_data["date"] >= cutoff]
    mean = chart_data["ratio"].mean()
    lower_bound = chart_data["ratio"].quantile(0.05)
    upper_bound = chart_data["ratio"].quantile(0.95)
    chart_data["lower_bound"] = lower_bound
    chart_data["upper_bound"] = upper_bound
    interval = (
        alt.Chart(chart_data)
        .mark_area(color=cs.CHART_AMBER, opacity=0.12)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("lower_bound:Q", title=label, scale=alt.Scale(zero=False)),
            y2="upper_bound:Q",
        )
    )
    line = (
        alt.Chart(chart_data)
        .mark_line(color=cs.CHART_PRIMARY, strokeWidth=2.25)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("ratio:Q", title=label, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("ratio:Q", title=label, format=".2f")],
        )
    )
    mean_rule = (
        alt.Chart(pd.DataFrame({"mean": [mean]}))
        .mark_rule(color=cs.CHART_AMBER, strokeDash=[6, 5])
        .encode(y="mean:Q")
    )
    return (interval + line + mean_rule).properties(height=290)


def _vs_median(discount: float) -> str:
    gap = pos.gap_to_median(discount)
    return f"{abs(gap):.0f}% {'below' if gap >= 0 else 'above'}"


def _pct(value, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# COMPANY DETAIL (Evidence and Lab)
# ═══════════════════════════════════════════════════════════════════════════════

def render_detail(symbol: str, profile_row: pd.Series, history_years: int, score: float | None = None) -> None:
    """Valuation history, growth, projections and the checklist for one company."""
    with st.spinner(f"Loading the research view for {symbol}..."):
        detail = load_stock_detail(symbol, pos.profile_values(profile_row), history_years, SCAN_SCHEMA_VERSION)
    if not detail:
        st.warning("The detailed history is unavailable for this company: it needs at least two years of "
                   "quarterly statements and a positive valuation history.")
        return

    detail_row = detail["row"]
    if not pos.detail_is_current(detail_row):
        load_stock_detail.clear()
        st.warning("The cached company detail is outdated. Select the company again to refresh it.")
        return

    bank = detail_row.get("Value Basis") == "P/B"
    second, second_key = ("P/B", "PB") if bank else ("P/S", "PS")
    leg = "Book" if bank else "Revenue"
    current_price = _finite(detail_row.get("Current Price"), _finite(profile_row.get("price")))
    detail_cols = st.columns(5)
    detail_cols[0].metric("Research score", f"{score:.0f} / 100" if score is not None else "n/a",
                          help="Scores are relative to the companies analyzed together.")
    detail_cols[1].metric(
        "P/E vs median",
        f"{detail_row['Current PE']:.2f}",
        delta=f"{detail_row['PE Discount']:+.1f}% vs median",
    )
    detail_cols[2].metric(
        f"{second} vs median",
        f"{detail_row[f'Current {second_key}']:.2f}",
        delta=f"{detail_row[f'{second_key} Discount']:+.1f}% vs median",
    )
    detail_cols[3].metric("Revenue growth", f"{detail_row['Revenue Growth 5Y']:.1f}%", help="Median YoY TTM growth")
    detail_cols[4].metric("Earnings growth", f"{detail_row['Earnings Growth 5Y']:.1f}%", help="Median YoY TTM growth")

    st.caption(
        f"{detail_row['Company']} | {detail_row['Sector']} | {_format_idr(detail_row['Market Cap'])} market cap | "
        f"latest statement {pd.Timestamp(detail_row['Statement Date']).date().isoformat()}"
        + (" | valued on book value as a bank" if bank else "")
    )

    projection_cols = st.columns(4)
    projection_cols[0].metric(
        "Current price",
        f"IDR {current_price:,.0f}" if np.isfinite(current_price) else "N/A",
    )
    projection_cols[1].metric(
        "Upside to mean valuation",
        f"{detail_row['Upside to Mean']:.1f}%",
        help=f"No-growth estimate using the historical mean P/E and {second}.",
    )
    projection_cols[2].metric(
        "Implied price in 1Y",
        f"IDR {detail_row['Implied Price 1Y']:,.0f}",
        delta=f"{detail_row['Potential Upside 1Y']:+.1f}% potential",
        help="Uses the latest year-over-year growth, capped between -20% and 30%.",
    )
    projection_cols[3].metric(
        "Implied price in 5Y",
        f"IDR {detail_row['Implied Price 5Y']:,.0f}",
        delta=f"{detail_row['Potential Upside 5Y']:+.1f}% potential",
        help="Compounds median five-year growth, capped between -20% and 30% annually.",
    )

    with st.expander("How potential upside is estimated"):
        second_line = (
            "historical mean P/B / current P/B x projected growth of book value per share"
            if bank else "historical mean P/S / current P/S x projected revenue growth"
        )
        st.markdown(
            f"""
            The model creates two estimates and gives them equal weight:

            - **Earnings value:** historical mean P/E / current P/E x projected earnings growth
            - **{leg} value:** {second_line}
            - **One year:** earnings growth **{detail_row['Earnings Growth Assumption 1Y']:.1f}%**, {leg.lower()} growth **{detail_row[f'{leg} Growth Assumption 1Y']:.1f}%**
            - **Five years:** earnings growth **{detail_row['Earnings Growth Assumption 5Y']:.1f}%**, {leg.lower()} growth **{detail_row[f'{leg} Growth Assumption 5Y']:.1f}%** per year

            Growth assumptions are capped at {pos.MIN_PROJECTION_GROWTH:.0f}% to {pos.MAX_PROJECTION_GROWTH:.0f}% annually.
            The result is a scenario, not a price target or discounted cash-flow valuation.
            """
        )

    valuation_tab, growth_tab, checklist_tab = st.tabs(
        ["Valuation history", "Business growth", "Investment checklist"]
    )
    with valuation_tab:
        chart_cols = st.columns(2)
        chart_cols[0].altair_chart(
            _ratio_chart(detail["pe_history"], "P/E", history_years), width="stretch", theme=None
        )
        second_history = detail["pb_history"] if bank else detail["ps_history"]
        if not second_history.empty:
            chart_cols[1].altair_chart(
                _ratio_chart(second_history, second, history_years), width="stretch", theme=None
            )
        st.caption(
            "The shaded band covers the historical 5th-to-95th percentile range. Dashed lines show the "
            "mean multiple used by the potential-upside scenarios."
        )

    with growth_tab:
        financial_history = detail["financial_history"].dropna(subset=["Revenue TTM", "Earnings TTM"]).copy()
        normalized = financial_history[["date", "Revenue TTM", "Earnings TTM"]].copy()
        for column in ("Revenue TTM", "Earnings TTM"):
            first_valid = normalized[column].replace(0, np.nan).dropna()
            normalized[column] = normalized[column] / first_valid.iloc[0] * 100 if not first_valid.empty else np.nan
        normalized = normalized.melt("date", var_name="Series", value_name="Index")
        growth_chart = (
            alt.Chart(normalized.dropna())
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("Index:Q", title="TTM index (first period = 100)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["Revenue TTM", "Earnings TTM"],
                        range=[cs.CHART_PRIMARY, cs.CHART_AMBER],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["date:T", "Series:N", alt.Tooltip("Index:Q", format=".1f")],
            )
            .properties(height=330)
        )
        st.altair_chart(growth_chart, width="stretch", theme=None)
        growth_metrics = st.columns(4)
        growth_metrics[0].metric("Revenue growth TTM", f"{detail_row['Revenue Growth TTM']:.1f}%")
        growth_metrics[1].metric("Earnings growth TTM", f"{detail_row['Earnings Growth TTM']:.1f}%")
        growth_metrics[2].metric("Net margin TTM", f"{detail_row['Margin TTM']:.1f}%")
        growth_metrics[3].metric("Historical margin", f"{detail_row['Margin Median']:.1f}%")

    with checklist_tab:
        checks = pos.checklist(detail_row)
        st.dataframe(checks[["Status", "Test", "Evidence"]], hide_index=True, width="stretch")
        st.warning(
            "This screen identifies candidates, not intrinsic value. Review one-off earnings, cyclicality, "
            "governance, and the latest filing before making an investment decision."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TODAY
# ═══════════════════════════════════════════════════════════════════════════════

def candidates_table(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per published candidate with the fields Today filters and shows."""
    rows = []
    for _, signal in signals.iterrows():
        d = signal["detail"]
        quality = d.get("quality") or {}
        rows.append({
            "Rank": d.get("rank"),
            "Symbol": signal["symbol"],
            "Company": d.get("company"),
            "Sector": d.get("sector") or "Unknown",
            "Score": d["scores"]["score"],
            "P/E": d["pe"]["current"],
            "P/E discount": pos.gap_to_median(d["pe"]["discount"]),
            "Basis": d.get("basis"),
            "Second discount": pos.gap_to_median(d["second"]["discount"]),
            "Upside to mean": d["upside"]["to_mean"],
            "Potential 1Y": d["upside"]["potential_1y"],
            "Revenue 5Y": d["growth"]["revenue_5y"],
            "Earnings 5Y": d["growth"]["earnings_5y"],
            "ROE": quality.get("roe"),
            "Debt to equity": quality.get("debt_to_equity"),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    text = ["Symbol", "Company", "Sector", "Basis"]
    numeric = table.columns.difference(text)
    table[numeric] = table[numeric].apply(pd.to_numeric, errors="coerce")
    return table.sort_values("Rank").reset_index(drop=True)


@st.dialog("Track this position")
def track_dialog(signal: pd.Series) -> None:
    detail = signal["detail"]
    st.markdown(f"**{ui.short(signal['symbol'])}** · {detail.get('company') or ''}")
    reference = float(signal["entry_price"])
    entry_price = st.number_input("Your buy price (Rp)", min_value=1.0, value=reference, step=5.0, format="%.0f")
    lot_count = st.number_input("Lots", min_value=1, value=10, step=1, help="1 lot is 100 shares")
    bought = st.date_input("Bought on", value=datetime.date.today(), max_value=datetime.date.today())
    pe = detail["pe"]
    st.caption(
        f"You get a Telegram alert when P/E reaches its mean of {pe['mean']:.1f} (now {pe['current']:.1f}) or a "
        "new quarter turns trailing growth negative."
    )
    if st.button("Track", type="primary", icon=":material/bookmark_add:"):
        store.create_position(get_connection(), current_email(), {
            "tool": "long_term", "symbol": signal["symbol"], "entry_date": bought,
            "entry_price": float(entry_price), "lots": int(lot_count), "exit_by": None,
            "exit_rule": {
                "kind": "valuation", "pe_at_entry": pe["current"], "pe_mean": pe["mean"],
                "target_price": signal["target_price"], "statement_date": signal["key_date"],
                "signal_as_of": signal["as_of"],
            },
        })
        load_open_positions.clear()
        st.session_state["long_term_toast"] = f"Tracking {ui.short(signal['symbol'])}. See My Trades."
        st.rerun()


def _meter(label: str, current: float, median: float, discount: float, digits: int) -> str:
    """Current multiple against its own median; the median sits at three quarters of the bar."""
    scale = median / 0.75 if median and median > 0 else max(current, 1.0)
    fill = min(max(current / scale, 0.0), 1.0) * 100
    return (
        f'<div class="hv-meter"><div class="row"><span>{ui.esc(label)} <b>{current:.{digits}f}</b></span>'
        f'<span>{ui.esc(_vs_median(discount))} its median of {median:.{digits}f}</span></div>'
        f'<div class="bar"><div class="fill" style="width:{fill:.1f}%"></div>'
        f'<div class="median" style="left:75%" title="Median"></div></div></div>'
    )


def render_candidate_card(signal: pd.Series, can_track: bool) -> None:
    d = signal["detail"]
    pe, second, growth, upside = d["pe"], d["second"], d["growth"], d["upside"]
    quality = d.get("quality") or {}
    score = d["scores"]["score"]
    ring = f'<div class="hv-ring" style="--p:{score:.0f}" title="Score out of 100"><span>{score:.0f}</span></div>'
    chips = [ui.chip(d.get("sector") or "Unknown sector", "brand", "domain"),
             ui.chip(f"ROE {_pct(quality.get('roe'), signed=False)}", "", "verified")]
    if d.get("basis") != "P/B" and quality.get("debt_to_equity") is not None:
        chips.append(ui.chip(f"Debt {quality['debt_to_equity']:.2f}x equity", "", "account_balance"))
    if d.get("basis") != "P/B" and quality.get("cash_conversion") is not None:
        chips.append(ui.chip(f"Cash flow {quality['cash_conversion']:.0f}% of profit", "", "water_drop"))
    tone = lambda v: "" if v is None or pd.isna(v) else ("up" if v > 0 else "down")
    parts = [
        ui.card_head(signal["symbol"], d.get("company"), ring, rank=d.get("rank"), icon_name="trending_up"),
        f'<div class="hv-chips">{"".join(chips)}</div>',
        '<div class="hv-meters">' + _meter("P/E", pe["current"], pe["median"], pe["discount"], 1)
        + _meter(second["label"], second["current"], second["median"], second["discount"], 2) + "</div>",
        ui.cells([
            ("Revenue a year", _pct(growth["revenue_5y"]), "5-year growth", tone(growth["revenue_5y"])),
            ("Earnings a year", _pct(growth["earnings_5y"]), "5-year growth", tone(growth["earnings_5y"])),
            ("Latest earnings", _pct(growth["earnings_ttm"]), "trailing 12 months", tone(growth["earnings_ttm"])),
        ]),
        ui.strip(
            f"Back at its mean valuation: about <b>{ui.esc(ui.idr(signal['target_price']))}</b> "
            f"({_pct(upside['to_mean'])} from {ui.esc(ui.idr(signal['entry_price']))})", icon_name="flag",
        ),
        ui.foot(f"Latest statement {ui.esc(ui.day(signal['key_date']))} {pd.Timestamp(signal['key_date']).year}. "
                "Exit when the discount closes or growth turns negative."),
    ]
    with ui.card(signal["symbol"]):
        ui.render("".join(parts))
        with st.container(horizontal=True):
            if can_track and st.button("Track this position", key=f"track_{signal['symbol']}", type="primary",
                                       icon=":material/bookmark_add:"):
                if current_email() is None:
                    st.login("google")
                else:
                    track_dialog(signal)
            st.button("See evidence", key=f"evidence_{signal['symbol']}", icon=":material/query_stats:",
                      on_click=ui.open_view, args=(VIEW_EVIDENCE, signal["symbol"]))


def backtest_line(record: dict) -> str | None:
    if not record or not record.get("years"):
        return None
    return (
        f"Backtest since {ui.day(record['start'])} {pd.Timestamp(record['start']).year}: the top "
        f"{record['top_n']} each April returned **{record['cagr']:+.1f}% a year** against "
        f"{record['benchmark_cagr']:+.1f}% for the JKSE, ahead in {record['beat_years']} of {record['years']} years."
    )


def render_backtest(record: dict) -> None:
    """Year-by-year record of the published screen: growth of 100 and a table."""
    years = pd.DataFrame(record.get("per_year") or [])
    if years.empty:
        return
    years["start"] = pd.to_datetime(years["start"])
    years["end"] = pd.to_datetime(years["end"])
    growth = pd.DataFrame({
        "date": [years["start"].iloc[0], *years["end"]],
        "Screen": 100 * np.cumprod([1.0, *(1 + years["portfolio"] / 100)]),
        "JKSE": 100 * np.cumprod([1.0, *(1 + years["benchmark"] / 100)]),
    }).melt("date", var_name="Series", value_name="Value")
    chart = (
        alt.Chart(growth)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Value:Q", title="Growth of 100", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Screen", "JKSE"],
                                                        range=[cs.CHART_PRIMARY, cs.CHART_AMBER]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["date:T", "Series:N", alt.Tooltip("Value:Q", format=".1f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch", theme=None)
    table = pd.DataFrame({
        "Held from": years["start"].dt.date,
        "To": years["end"].dt.date,
        "Screen": years["portfolio"],
        "JKSE": years["benchmark"],
        "Picks": years["holdings"].map(lambda h: [ui.short(s) for s in h] if h else ["cash"]),
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Picks": st.column_config.ListColumn("Picks"),
            "Screen": st.column_config.NumberColumn("Screen", format="%+.1f%%"),
            "JKSE": st.column_config.NumberColumn("JKSE", format="%+.1f%%"),
        },
    )
    open_year = "" if years["complete"].iloc[-1] else " The last row is the current year, still open."
    st.caption(
        f"Equal weight, rebalanced on the first session in April, price returns after "
        f"{record.get('costs_pct', pos.BACKTEST_ROUND_TRIP_COST):g}% costs a year; dividends are excluded on both "
        f"sides. Each year uses only statements already public: {pos.INTERIM_LAG_DAYS} days after a quarter "
        f"ends, {pos.ANNUAL_LAG_DAYS} after the year ends. Companies that have since delisted are missing, which "
        f"flatters the result.{open_year}"
    )


def render_today(access: bool) -> None:
    if "long_term_toast" in st.session_state:
        st.toast(st.session_state.pop("long_term_toast"), icon=":material/bookmark_added:")
    published = load_published()
    if published is None:
        st.warning("Candidates are unavailable right now. Please try again in a few minutes.")
        return
    run, signals = published["run"], published["signals"]
    if run is None:
        st.info("Candidates have not been published yet. The first update arrives after the next market close.")
        return
    ui.status_line(run, published["stale"])

    stats = run.get("stats") or {}
    record = ((published["setups"].get("global") or {}).get("holdout")) or {}
    email = current_email()
    metrics = st.container(horizontal=True)
    metrics.metric("Candidates", len(signals))
    if stats.get("requested"):
        metrics.metric("Companies analyzed", f"{stats.get('analyzed', 0)} / {stats['requested']}",
                       help="The largest IDX companies by market cap. The rest lacked enough statements or "
                            "price history, or could not be fetched.")
    if email:
        try:
            metrics.metric("Your long term positions", len(load_open_positions(email)))
        except Exception:
            metrics.metric("Your long term positions", "–")
    line = backtest_line(record)
    if line:
        ui.render(ui.strip(ui.bold(line), icon_name="insights"))

    if signals.empty:
        ui.empty_state("No company passes every rule of the screen today",
                       "The screen runs again after the next close.", "filter_alt_off")
    elif not access:
        first = signals.sort_values("detail", key=lambda d: d.map(lambda x: x.get("rank") or 0)).iloc[0]
        ui.section("Top candidate")
        render_candidate_card(first, can_track=False)
        others = [s for s in candidates_table(signals)["Symbol"] if s != first["symbol"]]
        if others:
            ui.unlock_prompt(others, noun="candidate")
    else:
        table = candidates_table(signals)
        ui.section("Candidates", len(table), "Filter the table, then select a row to open its card")
        filters = st.container(horizontal=True)
        sectors = filters.multiselect("Sector", sorted(table["Sector"].unique()), placeholder="All sectors")
        max_pe = filters.number_input("Maximum P/E", min_value=1.0, value=float(np.ceil(table["P/E"].max())),
                                      step=1.0)
        min_upside = filters.number_input("Minimum upside to mean (%)", value=0.0, step=5.0)
        min_roe = filters.number_input("Minimum ROE (%)", value=0.0, step=1.0)
        shown = table[
            (table["Sector"].isin(sectors) if sectors else True)
            & (table["P/E"] <= max_pe)
            & (table["Upside to mean"] >= min_upside)
            & (table["ROE"].fillna(0) >= min_roe)
        ]
        st.caption(f"{len(shown)} of {len(table)} candidates. Select a row for its card.")
        selection = st.dataframe(
            shown, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            key="candidate_table",
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
                "P/E discount": st.column_config.NumberColumn("P/E below median", format="%.0f%%",
                                                              help="Negative means above the median"),
                "Basis": st.column_config.TextColumn("Second multiple", help="P/S, or P/B for banks"),
                "Second discount": st.column_config.NumberColumn("Below median", format="%.0f%%"),
                "Upside to mean": st.column_config.NumberColumn("Upside to mean", format="%+.0f%%"),
                "Potential 1Y": st.column_config.NumberColumn("Potential 1Y", format="%+.0f%%"),
                "Revenue 5Y": st.column_config.NumberColumn("Revenue 5Y", format="%+.1f%%"),
                "Earnings 5Y": st.column_config.NumberColumn("Earnings 5Y", format="%+.1f%%"),
                "ROE": st.column_config.NumberColumn("ROE", format="%.0f%%"),
                "Debt to equity": st.column_config.NumberColumn("Debt/equity", format="%.2f"),
            },
        )
        rows = selection.selection.rows if selection else []
        if not shown.empty:
            symbol = shown.iloc[rows[0] if rows else 0]["Symbol"]
            render_candidate_card(signals[signals["symbol"] == symbol].iloc[0], can_track=True)

    if record.get("per_year"):
        with st.expander("How this screen did, year by year"):
            render_backtest(record)
    st.caption("Fundamental screen of past statements, not a price target. Not financial advice.")


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════════

def render_evidence(access: bool) -> None:
    try:
        universe = load_idx_universe()
    except Exception:
        logger.exception("Could not load the IDX universe")
        universe = pd.DataFrame()
    if universe.empty:
        st.warning("The company list is unavailable right now. Please try again in a few minutes.")
        return
    options = sorted(universe.index)
    default = FREE_EVIDENCE_TICKER if FREE_EVIDENCE_TICKER in options else options[0]
    if st.session_state.get("stock") not in options:
        st.session_state["stock"] = default
    # No format_func: the bound URL parameter is the displayed value, and links use the full symbol.
    stock = st.selectbox("Company", options, key="stock", bind="query-params")
    if not access and stock != FREE_EVIDENCE_TICKER:
        require_premium(teaser=lambda: st.info(
            f"The evidence for {ui.short(FREE_EVIDENCE_TICKER)} is free to explore. Supporters can open any company.",
            icon=":material/info:",
        ))

    published = load_published()
    signals = published["signals"] if published else pd.DataFrame()
    match = signals[signals["symbol"] == stock] if not signals.empty else signals
    score = None
    if not match.empty:
        detail = match.iloc[0]["detail"]
        score = detail["scores"]["score"]
        ui.render(ui.strip(f"Rank <b>{detail.get('rank')}</b> of {len(signals)} in the latest screen.",
                           icon_name="military_tech"))
    elif published and published["run"]:
        ui.render(ui.strip("Not among the latest screen's candidates. The Lab shows which rules it fails.",
                           "plain", "info"))
    history_years = int(((published or {}).get("setups", {}).get("global") or {}).get("params", {}).get(
        "history_years", pos.DEFAULT_FILTERS["history_years"]))
    render_detail(stock, universe.loc[stock], history_years, score)


access = has_access()
if view == VIEW_TODAY:
    render_today(access)
    st.stop()
if view == VIEW_EVIDENCE:
    render_evidence(access)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# LAB
# ═══════════════════════════════════════════════════════════════════════════════

require_premium()
st.caption("Research tools. Results here do not change the published candidates on the Today view.")

try:
    with st.spinner("Loading the IDX company universe..."):
        universe = load_idx_universe()
except Exception as exc:
    universe = pd.DataFrame()
    st.error(f"The IDX universe could not be loaded: {exc}")


st.subheader("Build the research universe", divider="green")
with st.form("fundamental_screen"):
    filter_cols = st.columns(4)
    min_market_cap = filter_cols[0].number_input(
        "Minimum market cap (IDR T)", min_value=0.1, max_value=500.0, value=2.0, step=0.5
    )
    max_companies = filter_cols[1].number_input(
        "Companies to analyze", min_value=10, max_value=300, value=100, step=10
    )
    history_years = filter_cols[2].selectbox(
        "Valuation history", options=[3, 5, 10], index=1, format_func=lambda value: f"{value} years"
    )
    max_pe = filter_cols[3].number_input(
        "Maximum current P/E", min_value=5.0, max_value=100.0, value=20.0, step=5.0
    )

    growth_cols = st.columns(4)
    min_revenue_growth = growth_cols[0].number_input(
        "Minimum 5Y revenue growth (%)", min_value=-50.0, max_value=100.0, value=3.0, step=1.0
    )
    min_earnings_growth = growth_cols[1].number_input(
        "Minimum 5Y earnings growth (%)", min_value=-100.0, max_value=200.0, value=0.0, step=1.0
    )
    require_recent_growth = growth_cols[2].checkbox(
        "Require positive TTM growth", value=True,
        help="Both latest revenue and latest earnings must be above the prior trailing twelve months.",
    )
    require_both_discounts = growth_cols[3].checkbox(
        "Require P/E and P/S discounts", value=True,
        help="Both multiples must be below their historical medians (P/E and P/B for banks).",
    )
    require_quality = st.checkbox(
        "Require quality checks", value=True,
        help=(
            f"Debt at most {pos.MAX_DEBT_TO_EQUITY:g}x equity, interest covered {pos.MIN_INTEREST_COVERAGE:g}x, "
            f"operating cash flow at least {pos.MIN_CASH_CONVERSION:g}% of profit over three years, "
            f"ROE at least {pos.MIN_ROE:g}% and shares up at most {pos.MAX_SHARE_CHANGE:g}% in a year. "
            "Banks only need the ROE and share count checks."
        ),
    )
    submitted = st.form_submit_button(
        "Run fundamental screen", type="primary", disabled=universe.empty, width="stretch"
    )


if submitted:
    selected_profiles = pos.select_universe(universe, min_market_cap, max_companies)
    if selected_profiles.empty:
        st.warning("No active companies meet the selected market-cap floor.")
    else:
        with st.spinner(
            f"Analyzing earnings, revenue, and valuation history for {len(selected_profiles)} companies..."
        ):
            profile_version, profile_records = pos.profile_snapshot(selected_profiles)
            raw_results = run_fundamental_scan(
                profile_version,
                profile_records,
                int(history_years),
                SCAN_SCHEMA_VERSION,
            )
        st.session_state["fundamental_scan"] = {
            "schema_version": SCAN_SCHEMA_VERSION,
            "results": raw_results,
            "profiles": selected_profiles,
            "history_years": int(history_years),
            "filters": {
                "max_pe": float(max_pe),
                "min_revenue_growth": float(min_revenue_growth),
                "min_earnings_growth": float(min_earnings_growth),
                "require_recent_growth": bool(require_recent_growth),
                "require_both_discounts": bool(require_both_discounts),
                "require_quality": bool(require_quality),
            },
            "requested": len(selected_profiles),
        }


scan_state = st.session_state.get("fundamental_scan")
if scan_state and (
    scan_state.get("schema_version") != SCAN_SCHEMA_VERSION
    or not pos.detail_is_current(scan_state.get("results", pd.DataFrame()).columns)
):
    st.session_state.pop("fundamental_scan", None)
    scan_state = None
    st.info("The valuation model was updated. Run the fundamental screen again to calculate the new projections.")

if not scan_state:
    st.info(
        "Choose the universe and run the screen. The first scan fetches historical prices and quarterly "
        "statements; results are cached for six hours."
    )
    st.stop()


raw_results = scan_state["results"].copy()
filters = scan_state["filters"]
history_years = scan_state["history_years"]
if raw_results.empty:
    st.warning("No companies had enough valid price and quarterly financial history for comparison.")
    st.stop()

raw_results = pos.apply_screen(raw_results, filters)
results = pos.ranked(raw_results[raw_results["Passes All"]])

st.subheader("Candidates", divider="green")
summary_cols = st.columns(4)
summary_cols[0].metric("Qualified", f"{len(results)}", help="Companies passing every selected rule")
summary_cols[1].metric("Analyzed", f"{len(raw_results)} / {scan_state['requested']}")
summary_cols[2].metric(
    "Median 1Y potential",
    f"{results['Potential Upside 1Y'].median():.1f}%" if not results.empty else "N/A",
)
summary_cols[3].metric(
    "Median revenue growth", f"{results['Revenue Growth 5Y'].median():.1f}%" if not results.empty else "N/A"
)

st.markdown(
    '<p class="method-note"><strong>Score:</strong> 55% historical valuation, 35% growth, and 10% '
    'quality (net margin and ROE). A valuation score of 80 means the combined P/E and P/S are near the cheapest 20% '
    'of their own selected history. Growth and quality scores are relative to the analyzed universe. '
    '<strong>Potential upside:</strong> the average of earnings- and revenue-based implied values if '
    'multiples return to their historical means and growth persists.</p>',
    unsafe_allow_html=True,
)

show_all = st.toggle(
    "Show all analyzed companies",
    key="fundamental_show_all",
    help="List every analyzed company. Companies that pass all selected rules are marked in the Passes column.",
)

if results.empty:
    st.warning(
        "No company passes every rule. Relax one constraint at a time, starting with positive TTM "
        "earnings growth, which can be volatile for cyclical businesses."
    )
    if not show_all:
        st.stop()

if show_all:
    table = pos.ranked(raw_results)
    st.caption(f"{int(table['Passes All'].sum())} of {len(table)} analyzed companies pass every selected rule.")
else:
    table = results

display_columns = [
    "Passes All",
    "Symbol",
    *(["Failed Rules"] if show_all else []),
    "Company",
    "Score",
    "Upside to Mean",
    "Potential Upside 1Y",
    "Potential Upside 5Y",
    "Current PE",
    "PE Discount",
    "PE Percentile",
    "Current PS",
    "PS Discount",
    "PS Percentile",
    "Current PB",
    "PB Discount",
    "Revenue Growth 5Y",
    "Revenue Growth TTM",
    "Earnings Growth 5Y",
    "Earnings Growth TTM",
    "Margin TTM",
    "ROE",
    "Debt to Equity",
    "Cash Conversion",
    "Market Cap",
]
st.dataframe(
    table[display_columns].assign(
        **{"Passes All": np.where(table["Passes All"], "✅", "❌" if show_all else "")},
        **({"Failed Rules": table["Failed Rules"].map(lambda text: text.split(", ") if text else [])}
           if show_all else {}),
    ),
    column_config={
        "Passes All": st.column_config.TextColumn(
            "Passes", help="Passes every selected screening rule", width="small"
        ),
        "Failed Rules": st.column_config.ListColumn(
            "Fails on", help="Screening rules this company does not meet", width="medium"
        ),
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Upside to Mean": st.column_config.NumberColumn("Mean Reversion", format="%.1f%%"),
        "Potential Upside 1Y": st.column_config.NumberColumn("Potential 1Y", format="%.1f%%"),
        "Potential Upside 5Y": st.column_config.NumberColumn("Potential 5Y", format="%.1f%%"),
        "Current PE": st.column_config.NumberColumn("P/E", format="%.2f"),
        "PE Discount": st.column_config.NumberColumn("P/E Discount", format="%.1f%%"),
        "PE Percentile": st.column_config.NumberColumn("P/E Percentile", format="%.0f%%"),
        "Current PS": st.column_config.NumberColumn("P/S", format="%.2f"),
        "PS Discount": st.column_config.NumberColumn("P/S Discount", format="%.1f%%"),
        "PS Percentile": st.column_config.NumberColumn("P/S Percentile", format="%.0f%%"),
        "Revenue Growth 5Y": st.column_config.NumberColumn("Revenue 5Y", format="%.1f%%"),
        "Revenue Growth TTM": st.column_config.NumberColumn("Revenue TTM", format="%.1f%%"),
        "Earnings Growth 5Y": st.column_config.NumberColumn("Earnings 5Y", format="%.1f%%"),
        "Earnings Growth TTM": st.column_config.NumberColumn("Earnings TTM", format="%.1f%%"),
        "Margin TTM": st.column_config.NumberColumn("Net Margin", format="%.1f%%"),
        "Current PB": st.column_config.NumberColumn("P/B", format="%.2f"),
        "PB Discount": st.column_config.NumberColumn("P/B Discount", format="%.1f%%"),
        "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
        "Debt to Equity": st.column_config.NumberColumn("Debt/Equity", format="%.2f"),
        "Cash Conversion": st.column_config.NumberColumn("Cash Conversion", format="%.0f%%",
                                                         help="Operating cash flow over net income, 3 years"),
        "Market Cap": st.column_config.NumberColumn("Market Cap", format="compact"),
    },
    hide_index=False,
    width="stretch",
    height=min(720, 38 * len(table) + 38),
)

st.download_button(
    "Download all analyzed" if show_all else "Download candidates",
    data=table.reset_index().to_csv(index=False).encode("utf-8"),
    file_name=f"idx_growth_at_a_discount_{datetime.date.today().isoformat()}.csv",
    mime="text/csv",
)


st.subheader("Inspect a candidate", divider="green")
symbols = table["Symbol"].tolist()
selected_symbol = st.selectbox(
    "Company",
    symbols,
    format_func=lambda symbol: (
        f"{symbol} - {table.loc[table['Symbol'] == symbol, 'Company'].iloc[0]}"
        + ("" if table.loc[table["Symbol"] == symbol, "Passes All"].iloc[0] else " (does not pass all rules)")
    ),
)
selected_row = table.loc[table["Symbol"] == selected_symbol].iloc[0]
render_detail(selected_symbol, scan_state["profiles"].loc[selected_symbol], history_years, selected_row["Score"])
