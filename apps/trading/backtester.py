import concurrent.futures
import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import harvest.data as hd


CACHE_TTL = 6 * 60 * 60
SCAN_SCHEMA_VERSION = 3
PROFILE_BATCH_SIZE = 30
SCAN_WORKERS = 5
MIN_PROJECTION_GROWTH = -20.0
MAX_PROJECTION_GROWTH = 30.0
RESULT_COLUMNS = [
    "Symbol",
    "Company",
    "Sector",
    "Market Cap",
    "Current PE",
    "PE Mean",
    "PE Median",
    "PE Discount",
    "PE Percentile",
    "Current PS",
    "PS Mean",
    "PS Median",
    "PS Discount",
    "PS Percentile",
    "Revenue Growth 5Y",
    "Revenue Growth TTM",
    "Earnings Growth 5Y",
    "Earnings Growth TTM",
    "Margin TTM",
    "Valuation Score",
    "Growth Score",
    "Quality Score",
    "Score",
    "Upside to Mean",
    "Potential Upside 1Y",
    "Potential Upside 5Y",
    "Implied Price 1Y",
    "Implied Price 5Y",
    "Statement Date",
]


st.set_page_config(page_title='Position Trading - Panen Dividen')
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


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _series(df: pd.DataFrame, column: str, default=None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _finite(value, default=np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _text(value, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _median_comparison(discount: float, multiple: str) -> str:
    direction = "below" if discount >= 0 else "above"
    return f"{multiple} is {abs(discount):.1f}% {direction} its median"


def _bounded_growth(value: float) -> float:
    """Convert a percentage to a conservative annual projection rate."""
    return np.clip(_finite(value, 0.0), MIN_PROJECTION_GROWTH, MAX_PROJECTION_GROWTH) / 100


def _project_upside(
    pe_current: float,
    pe_mean: float,
    ps_current: float,
    ps_mean: float,
    earnings_growth: float,
    revenue_growth: float,
    years: int,
) -> dict:
    earnings_rate = _bounded_growth(earnings_growth)
    revenue_rate = _bounded_growth(revenue_growth)
    pe_factor = pe_mean / pe_current * (1 + earnings_rate) ** years
    ps_factor = ps_mean / ps_current * (1 + revenue_rate) ** years
    blended_factor = np.mean([pe_factor, ps_factor])
    return {
        "blended": (blended_factor - 1) * 100,
        "earnings": (pe_factor - 1) * 100,
        "revenue": (ps_factor - 1) * 100,
        "earnings_growth": earnings_rate * 100,
        "revenue_growth": revenue_rate * 100,
    }


def _pct_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    return numeric.rank(pct=True, method="average") * 100


def _format_idr(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000_000_000_000:
        return f"IDR {value / 1_000_000_000_000:.1f}T"
    if abs(value) >= 1_000_000_000:
        return f"IDR {value / 1_000_000_000:.1f}B"
    return f"IDR {value:,.0f}"


def _fetch_profiles(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for start in range(0, len(symbols), PROFILE_BATCH_SIZE):
        try:
            frame = hd.get_company_profile(symbols[start : start + PROFILE_BATCH_SIZE])
            if frame is not None and not frame.empty:
                frames.append(frame)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).loc[lambda frame: ~frame.index.duplicated(keep="first")]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_idx_universe() -> pd.DataFrame:
    listed = hd.get_all_idx_stocks()
    symbols = listed["symbol"].dropna().astype(str).unique().tolist()
    profiles = _fetch_profiles(symbols)
    if profiles.empty:
        return profiles

    profiles = profiles.copy()
    profiles["mktCap"] = pd.to_numeric(_series(profiles, "mktCap"), errors="coerce")
    profiles["price"] = pd.to_numeric(_series(profiles, "price"), errors="coerce")
    profiles["isActivelyTrading"] = (
        _series(profiles, "isActivelyTrading", False).fillna(False).astype(bool)
    )
    profiles["isEtf"] = _series(profiles, "isEtf", False).fillna(False).astype(bool)
    profiles["isFund"] = _series(profiles, "isFund", False).fillna(False).astype(bool)
    profiles = profiles[
        profiles["isActivelyTrading"]
        & ~profiles["isEtf"]
        & ~profiles["isFund"]
        & (profiles["mktCap"] > 0)
        & (profiles["price"] > 0)
    ]
    profiles.index.name = "symbol"
    return profiles.sort_values("mktCap", ascending=False)


def _prepare_financials(financials: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "revenue", "netIncome", "reportedCurrency"}
    if financials is None or financials.empty or not required.issubset(financials.columns):
        return pd.DataFrame()

    frame = financials.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("revenue", "netIncome"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "revenue", "netIncome"])
    return frame.sort_values("date", ascending=False).reset_index(drop=True)


def _financial_history(financials: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ordered = financials.sort_values("date").copy()
    ordered["Revenue TTM"] = ordered["revenue"].rolling(4, min_periods=4).sum()
    ordered["Earnings TTM"] = ordered["netIncome"].rolling(4, min_periods=4).sum()
    ordered["Revenue Growth"] = ordered["Revenue TTM"].pct_change(4) * 100
    ordered["Earnings Growth"] = ordered["Earnings TTM"].pct_change(4) * 100
    ordered["Margin"] = ordered["Earnings TTM"] / ordered["Revenue TTM"] * 100
    valid = ordered.dropna(subset=["Revenue TTM", "Earnings TTM"])
    if valid.empty:
        return ordered, {}

    recent_growth = valid.tail(20)
    stats = {
        "revenue_ttm": _finite(valid["Revenue TTM"].iloc[-1]),
        "earnings_ttm": _finite(valid["Earnings TTM"].iloc[-1]),
        "revenue_growth_ttm": _finite(valid["Revenue Growth"].iloc[-1]),
        "earnings_growth_ttm": _finite(valid["Earnings Growth"].iloc[-1]),
        "revenue_growth_5y": _finite(recent_growth["Revenue Growth"].median()),
        "earnings_growth_5y": _finite(recent_growth["Earnings Growth"].median()),
        "margin_ttm": _finite(valid["Margin"].iloc[-1]),
        "margin_median": _finite(valid.tail(20)["Margin"].median()),
        "statement_date": valid["date"].iloc[-1],
    }
    return ordered, stats


def _ratio_summary(
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    shares: float,
    ratio: str,
    history_years: int,
) -> tuple[pd.DataFrame, dict]:
    reported_currency = str(financials["reportedCurrency"].dropna().iloc[0])
    history = hd.calc_ratio_history(
        prices,
        financials.copy(),
        n_shares=shares,
        ratio=ratio,
        reported_currency=reported_currency,
        target_currency="IDR",
    )
    history = history.rename(columns={"pe": "ratio"})[["date", "ratio"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["ratio"] = pd.to_numeric(history["ratio"], errors="coerce")
    history = history.replace([np.inf, -np.inf], np.nan).dropna()
    history = history[history["ratio"] > 0].sort_values("date")
    if history.empty:
        return history, {}

    cutoff = history["date"].max() - pd.DateOffset(years=history_years)
    window = history[history["date"] >= cutoff]
    if len(window) < 60:
        return history, {}

    current = _finite(history["ratio"].iloc[-1])
    mean = _finite(window["ratio"].mean())
    median = _finite(window["ratio"].median())
    percentile = _finite((window["ratio"] <= current).mean() * 100)
    discount = _finite((median / current - 1) * 100) if current > 0 else np.nan
    return history, {
        "current": current,
        "mean": mean,
        "median": median,
        "percentile": percentile,
        "discount": discount,
        "observations": len(window),
    }


def _analyze_stock(
    symbol: str,
    profile: dict,
    history_years: int,
    include_history: bool = False,
):
    try:
        price = _finite(profile.get("price"))
        market_cap = _finite(profile.get("mktCap"))
        shares = market_cap / price if price > 0 and market_cap > 0 else np.nan
        if not np.isfinite(shares) or shares <= 0:
            return None

        start = (datetime.date.today() - datetime.timedelta(days=(history_years + 1) * 366)).isoformat()
        prices = hd.get_daily_stock_price(symbol, start_from=start)
        financials = _prepare_financials(hd.get_financial_data(symbol, period="quarter"))
        if prices is None or prices.empty or len(financials) < 8:
            return None

        financial_history, growth = _financial_history(financials)
        pe_history, pe = _ratio_summary(prices, financials, shares, "pe", history_years)
        ps_history, ps = _ratio_summary(prices, financials, shares, "ps", history_years)
        if not growth or not pe or not ps:
            return None

        mean_reversion = np.mean(
            [pe["mean"] / pe["current"], ps["mean"] / ps["current"]]
        )
        one_year = _project_upside(
            pe["current"],
            pe["mean"],
            ps["current"],
            ps["mean"],
            growth["earnings_growth_ttm"],
            growth["revenue_growth_ttm"],
            1,
        )
        five_year = _project_upside(
            pe["current"],
            pe["mean"],
            ps["current"],
            ps["mean"],
            growth["earnings_growth_5y"],
            growth["revenue_growth_5y"],
            5,
        )

        row = {
            "Symbol": symbol,
            "Company": _text(profile.get("companyName"), symbol),
            "Sector": _text(profile.get("sector"), "Unknown"),
            "Market Cap": market_cap,
            "Current Price": price,
            "Current PE": pe["current"],
            "PE Mean": pe["mean"],
            "PE Median": pe["median"],
            "PE Discount": pe["discount"],
            "PE Percentile": pe["percentile"],
            "Current PS": ps["current"],
            "PS Mean": ps["mean"],
            "PS Median": ps["median"],
            "PS Discount": ps["discount"],
            "PS Percentile": ps["percentile"],
            "Revenue Growth 5Y": growth["revenue_growth_5y"],
            "Revenue Growth TTM": growth["revenue_growth_ttm"],
            "Earnings Growth 5Y": growth["earnings_growth_5y"],
            "Earnings Growth TTM": growth["earnings_growth_ttm"],
            "Margin TTM": growth["margin_ttm"],
            "Margin Median": growth["margin_median"],
            "Revenue TTM": growth["revenue_ttm"],
            "Earnings TTM": growth["earnings_ttm"],
            "Upside to Mean": (mean_reversion - 1) * 100,
            "Potential Upside 1Y": one_year["blended"],
            "Potential Upside 5Y": five_year["blended"],
            "Earnings Upside 1Y": one_year["earnings"],
            "Revenue Upside 1Y": one_year["revenue"],
            "Earnings Upside 5Y": five_year["earnings"],
            "Revenue Upside 5Y": five_year["revenue"],
            "Earnings Growth Assumption 1Y": one_year["earnings_growth"],
            "Revenue Growth Assumption 1Y": one_year["revenue_growth"],
            "Earnings Growth Assumption 5Y": five_year["earnings_growth"],
            "Revenue Growth Assumption 5Y": five_year["revenue_growth"],
            "Implied Price 1Y": price * (1 + one_year["blended"] / 100),
            "Implied Price 5Y": price * (1 + five_year["blended"] / 100),
            "Statement Date": growth["statement_date"],
        }
        if not include_history:
            return row
        return {
            "row": row,
            "pe_history": pe_history,
            "ps_history": ps_history,
            "financial_history": financial_history,
        }
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def run_fundamental_scan(
    profile_records: tuple[tuple[str, tuple[tuple[str, object], ...]], ...],
    history_years: int,
    schema_version: int,
) -> pd.DataFrame:
    del schema_version  # Included in the cache key to invalidate older result schemas.
    profiles = {symbol: dict(values) for symbol, values in profile_records}
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
        futures = {
            executor.submit(_analyze_stock, symbol, profile, history_years): symbol
            for symbol, profile in profiles.items()
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
            except Exception:
                row = None
            if row:
                rows.append(row)

    if not rows:
        return _empty_results()

    result = pd.DataFrame(rows)
    result["Valuation Score"] = 100 - result[["PE Percentile", "PS Percentile"]].mean(axis=1)
    growth_columns = [
        "Revenue Growth 5Y",
        "Revenue Growth TTM",
        "Earnings Growth 5Y",
        "Earnings Growth TTM",
    ]
    result["Growth Score"] = pd.concat(
        [_pct_rank(result[column]).rename(column) for column in growth_columns], axis=1
    ).mean(axis=1)
    result["Quality Score"] = _pct_rank(result["Margin TTM"])
    result["Score"] = (
        result["Valuation Score"] * 0.55
        + result["Growth Score"] * 0.35
        + result["Quality Score"] * 0.10
    )
    result["Statement Date"] = pd.to_datetime(result["Statement Date"]).dt.date
    return result.sort_values("Score", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_stock_detail(
    symbol: str,
    profile_values: tuple,
    history_years: int,
    schema_version: int,
):
    del schema_version  # Included in the cache key to invalidate older detail schemas.
    return _analyze_stock(symbol, dict(profile_values), history_years, include_history=True)


def _profile_records(profiles: pd.DataFrame) -> tuple:
    fields = ["price", "mktCap", "companyName", "sector", "industry"]
    records = []
    for symbol, row in profiles.iterrows():
        values = tuple((field, row.get(field)) for field in fields)
        records.append((str(symbol), values))
    return tuple(records)


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
        .mark_area(color="#b45309", opacity=0.12)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("lower_bound:Q", title=label, scale=alt.Scale(zero=False)),
            y2="upper_bound:Q",
        )
    )
    line = (
        alt.Chart(chart_data)
        .mark_line(color="#0f766e", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("ratio:Q", title=label, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("ratio:Q", title=label, format=".2f")],
        )
    )
    mean_rule = (
        alt.Chart(pd.DataFrame({"mean": [mean]}))
        .mark_rule(color="#b45309", strokeDash=[6, 5])
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
        "Maximum current P/E", min_value=5.0, max_value=100.0, value=40.0, step=5.0
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
        "Run fundamental screen", type="primary", disabled=universe.empty, use_container_width=True
    )


if submitted:
    market_cap_floor = min_market_cap * 1_000_000_000_000
    selected_profiles = universe[universe["mktCap"] >= market_cap_floor].head(int(max_companies))
    if selected_profiles.empty:
        st.warning("No active companies meet the selected market-cap floor.")
    else:
        with st.spinner(
            f"Analyzing earnings, revenue, and valuation history for {len(selected_profiles)} companies..."
        ):
            raw_results = run_fundamental_scan(
                _profile_records(selected_profiles),
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
required_projection_columns = {
    "Current Price",
    "Upside to Mean",
    "Potential Upside 1Y",
    "Potential Upside 5Y",
    "Implied Price 1Y",
    "Implied Price 5Y",
}
if scan_state and (
    scan_state.get("schema_version") != SCAN_SCHEMA_VERSION
    or not required_projection_columns.issubset(scan_state.get("results", pd.DataFrame()).columns)
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

mask = (
    (raw_results["Current PE"] > 0)
    & (raw_results["Current PE"] <= filters["max_pe"])
    & (raw_results["Current PS"] > 0)
    & (raw_results["Revenue Growth 5Y"] >= filters["min_revenue_growth"])
    & (raw_results["Earnings Growth 5Y"] >= filters["min_earnings_growth"])
    & (raw_results["Earnings TTM"] > 0)
    & (raw_results["Revenue TTM"] > 0)
)
if filters["require_recent_growth"]:
    mask &= (raw_results["Revenue Growth TTM"] > 0) & (raw_results["Earnings Growth TTM"] > 0)
if filters["require_both_discounts"]:
    mask &= (raw_results["PE Discount"] > 0) & (raw_results["PS Discount"] > 0)

results = raw_results[mask].sort_values("Score", ascending=False).reset_index(drop=True)
results.index += 1
results.index.name = "Rank"

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

if results.empty:
    st.warning(
        "No company passes every rule. Relax one constraint at a time, starting with positive TTM "
        "earnings growth, which can be volatile for cyclical businesses."
    )
    st.stop()

display_columns = [
    "Symbol",
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
    results[display_columns],
    column_config={
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
    height=min(720, 38 * len(results) + 38),
)

st.download_button(
    "Download candidates",
    data=results.reset_index().to_csv(index=False).encode("utf-8"),
    file_name=f"idx_growth_at_a_discount_{datetime.date.today().isoformat()}.csv",
    mime="text/csv",
)


st.subheader("Inspect a candidate", divider="green")
symbols = results["Symbol"].tolist()
selected_symbol = st.selectbox(
    "Company",
    symbols,
    format_func=lambda symbol: f"{symbol} - {results.loc[results['Symbol'] == symbol, 'Company'].iloc[0]}",
)
selected_row = results.loc[results["Symbol"] == selected_symbol].iloc[0]
profile_row = scan_state["profiles"].loc[selected_symbol]
profile_values = tuple(
    (field, profile_row.get(field))
    for field in ["price", "mktCap", "companyName", "sector", "industry"]
)

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
required_detail_fields = {
    "Current Price",
    "Upside to Mean",
    "Potential Upside 1Y",
    "Potential Upside 5Y",
    "Implied Price 1Y",
    "Implied Price 5Y",
}
if not required_detail_fields.issubset(detail_row):
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

        Growth assumptions are capped at {MIN_PROJECTION_GROWTH:.0f}% to {MAX_PROJECTION_GROWTH:.0f}% annually.
        The result is a scenario, not a price target or discounted cash-flow valuation.
        """
    )

valuation_tab, growth_tab, checklist_tab = st.tabs(
    ["Valuation history", "Business growth", "Investment checklist"]
)
with valuation_tab:
    chart_cols = st.columns(2)
    chart_cols[0].altair_chart(
        _ratio_chart(detail["pe_history"], "P/E", history_years), use_container_width=True
    )
    chart_cols[1].altair_chart(
        _ratio_chart(detail["ps_history"], "P/S", history_years), use_container_width=True
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
                scale=alt.Scale(domain=["Revenue TTM", "Earnings TTM"], range=["#0f766e", "#b45309"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["date:T", "Series:N", alt.Tooltip("Index:Q", format=".1f")],
        )
        .properties(height=330)
    )
    st.altair_chart(growth_chart, use_container_width=True)
    growth_metrics = st.columns(4)
    growth_metrics[0].metric("Revenue growth TTM", f"{detail_row['Revenue Growth TTM']:.1f}%")
    growth_metrics[1].metric("Earnings growth TTM", f"{detail_row['Earnings Growth TTM']:.1f}%")
    growth_metrics[2].metric("Net margin TTM", f"{detail_row['Margin TTM']:.1f}%")
    growth_metrics[3].metric("Historical margin", f"{detail_row['Margin Median']:.1f}%")

with checklist_tab:
    checks = pd.DataFrame(
        [
            (
                "Earnings valuation",
                detail_row["PE Discount"] > 0,
                _median_comparison(detail_row["PE Discount"], "P/E"),
            ),
            (
                "Revenue valuation",
                detail_row["PS Discount"] > 0,
                _median_comparison(detail_row["PS Discount"], "P/S"),
            ),
            ("Durable revenue growth", detail_row["Revenue Growth 5Y"] > 0, f"5Y median growth is {detail_row['Revenue Growth 5Y']:.1f}%"),
            ("Durable earnings growth", detail_row["Earnings Growth 5Y"] > 0, f"5Y median growth is {detail_row['Earnings Growth 5Y']:.1f}%"),
            ("Recent revenue growth", detail_row["Revenue Growth TTM"] > 0, f"Latest TTM growth is {detail_row['Revenue Growth TTM']:.1f}%"),
            ("Recent earnings growth", detail_row["Earnings Growth TTM"] > 0, f"Latest TTM growth is {detail_row['Earnings Growth TTM']:.1f}%"),
            ("Profitable", detail_row["Earnings TTM"] > 0, f"Net margin is {detail_row['Margin TTM']:.1f}%"),
        ],
        columns=["Test", "Pass", "Evidence"],
    )
    checks["Status"] = np.where(checks["Pass"], "Pass", "Review")
    st.dataframe(checks[["Status", "Test", "Evidence"]], hide_index=True, width="stretch")
    st.warning(
        "This screen identifies candidates, not intrinsic value. Review debt, cash flow, dilution, one-off "
        "earnings, cyclicality, governance, and the latest filing before making an investment decision."
    )
