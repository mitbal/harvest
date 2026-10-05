import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import harvest.plot as hp
import harvest.position as pos
from harvest import chart_style as cs
from harvest.premium import require_premium


hp.enable_chart_theme()


CACHE_TTL = 6 * 60 * 60
SCAN_SCHEMA_VERSION = pos.SCAN_SCHEMA_VERSION
_finite, _format_idr = pos.finite, pos.format_idr


st.set_page_config(page_title='Position Trading - Panen Dividen')
require_premium()
st.title("Growth at a Discount")
st.caption(
    "Find profitable IDX businesses whose earnings and revenue multiples are below "
    "their own history while revenue and earnings continue to grow."
)

st.sidebar.markdown("### Research method")
st.sidebar.markdown(
    "The screen compares each company with itself. P/E measures the price paid for "
    "earnings, P/S measures the price paid for revenue, and growth confirms that a "
    "low multiple is not simply the result of a shrinking business."
)

st.html(
    """
    <style>
        [data-testid="stMetric"] {
            background: color-mix(in srgb, #064e3b 4%, transparent);
            border-top: 2px solid #0f766e;
            padding: 0.8rem 0.9rem;
        }
        [data-testid="stMetricLabel"] { color: #36534a; }
        [data-testid="stMetricValue"] { color: #123c31; }
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
        help="Both multiples must be below their historical medians.",
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
    'profit margin. A valuation score of 80 means the combined P/E and P/S are near the cheapest 20% '
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
    "Revenue Growth 5Y",
    "Revenue Growth TTM",
    "Earnings Growth 5Y",
    "Earnings Growth TTM",
    "Margin TTM",
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
profile_row = scan_state["profiles"].loc[selected_symbol]
profile_values = pos.profile_values(profile_row)

with st.spinner(f"Loading the research view for {selected_symbol}..."):
    detail = load_stock_detail(
        selected_symbol,
        profile_values,
        history_years,
        SCAN_SCHEMA_VERSION,
    )

if not detail:
    st.warning("The detailed history is temporarily unavailable for this company.")
    st.stop()

detail_row = detail["row"]
if not pos.detail_is_current(detail_row):
    load_stock_detail.clear()
    st.warning("The cached company detail is outdated. Select the company again to refresh it.")
    st.stop()

current_price = _finite(detail_row.get("Current Price"), _finite(profile_row.get("price")))
detail_cols = st.columns(5)
detail_cols[0].metric("Research score", f"{selected_row['Score']:.0f} / 100")
detail_cols[1].metric(
    "P/E vs median",
    f"{detail_row['Current PE']:.2f}",
    delta=f"{detail_row['PE Discount']:+.1f}% vs median",
)
detail_cols[2].metric(
    "P/S vs median",
    f"{detail_row['Current PS']:.2f}",
    delta=f"{detail_row['PS Discount']:+.1f}% vs median",
)
detail_cols[3].metric("Revenue growth", f"{detail_row['Revenue Growth 5Y']:.1f}%", help="Median YoY TTM growth")
detail_cols[4].metric("Earnings growth", f"{detail_row['Earnings Growth 5Y']:.1f}%", help="Median YoY TTM growth")

st.caption(
    f"{detail_row['Company']} | {detail_row['Sector']} | {_format_idr(detail_row['Market Cap'])} market cap | "
    f"latest statement {pd.Timestamp(detail_row['Statement Date']).date().isoformat()}"
)

projection_cols = st.columns(4)
projection_cols[0].metric(
    "Current price",
    f"IDR {current_price:,.0f}" if np.isfinite(current_price) else "N/A",
)
projection_cols[1].metric(
    "Upside to mean valuation",
    f"{detail_row['Upside to Mean']:.1f}%",
    help="No-growth estimate using the historical mean P/E and P/S.",
)
projection_cols[2].metric(
    "Implied price in 1Y",
    f"IDR {detail_row['Implied Price 1Y']:,.0f}",
    delta=f"{detail_row['Potential Upside 1Y']:+.1f}% potential",
    help="Uses latest TTM earnings and revenue growth, capped between -20% and 30%.",
)
projection_cols[3].metric(
    "Implied price in 5Y",
    f"IDR {detail_row['Implied Price 5Y']:,.0f}",
    delta=f"{detail_row['Potential Upside 5Y']:+.1f}% potential",
    help="Compounds median five-year earnings and revenue growth, capped between -20% and 30% annually.",
)

with st.expander("How potential upside is estimated"):
    st.markdown(
        f"""
        The model creates two estimates and gives them equal weight:

        - **Earnings value:** historical mean P/E / current P/E x projected earnings growth
        - **Revenue value:** historical mean P/S / current P/S x projected revenue growth
        - **One year:** earnings growth **{detail_row['Earnings Growth Assumption 1Y']:.1f}%**, revenue growth **{detail_row['Revenue Growth Assumption 1Y']:.1f}%**
        - **Five years:** earnings growth **{detail_row['Earnings Growth Assumption 5Y']:.1f}%**, revenue growth **{detail_row['Revenue Growth Assumption 5Y']:.1f}%** per year

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
    chart_cols[1].altair_chart(
        _ratio_chart(detail["ps_history"], "P/S", history_years), width="stretch", theme=None
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
        "This screen identifies candidates, not intrinsic value. Review debt, cash flow, dilution, one-off "
        "earnings, cyclicality, governance, and the latest filing before making an investment decision."
    )
