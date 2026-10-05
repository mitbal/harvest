"""Long Term engine ("Growth at a Discount"): valuation history, growth, scoring and projections.

Each company is compared with its own history: P/E and P/S against their
means and medians over the chosen window, and trailing-twelve-month revenue
and earnings growth to confirm a low multiple is not a shrinking business.

Streamlit-free so the page and the pipeline jobs share one implementation.
"""
import concurrent.futures
import datetime
import hashlib

import numpy as np
import pandas as pd

import harvest.data as hd


SCAN_SCHEMA_VERSION = 3
PROFILE_BATCH_SIZE = 30
SCAN_WORKERS = 5
MIN_PROJECTION_GROWTH = -20.0
MAX_PROJECTION_GROWTH = 30.0
PROFILE_FIELDS = ["price", "mktCap", "companyName", "sector", "industry"]
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
# The page's form defaults; the published screen uses the same values.
DEFAULT_FILTERS = {
    "min_market_cap": 2.0,
    "max_companies": 100,
    "history_years": 5,
    "max_pe": 20.0,
    "min_revenue_growth": 3.0,
    "min_earnings_growth": 0.0,
    "require_recent_growth": True,
    "require_both_discounts": True,
}


class SkipStock(Exception):
    """The stock lacks the data the analysis needs (as opposed to an unexpected failure)."""


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _series(df: pd.DataFrame, column: str, default=None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def finite(value, default=np.nan) -> float:
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


def median_comparison(discount: float, multiple: str) -> str:
    direction = "below" if discount >= 0 else "above"
    return f"{multiple} is {abs(discount):.1f}% {direction} its median"


def bounded_growth(value: float) -> float:
    """Convert a percentage to a conservative annual projection rate."""
    return np.clip(finite(value, 0.0), MIN_PROJECTION_GROWTH, MAX_PROJECTION_GROWTH) / 100


def project_upside(
    pe_current: float,
    pe_mean: float,
    ps_current: float,
    ps_mean: float,
    earnings_growth: float,
    revenue_growth: float,
    years: int,
) -> dict:
    earnings_rate = bounded_growth(earnings_growth)
    revenue_rate = bounded_growth(revenue_growth)
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


def pct_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    return numeric.rank(pct=True, method="average") * 100


def format_idr(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 1_000_000_000_000:
        return f"IDR {value / 1_000_000_000_000:.1f}T"
    if abs(value) >= 1_000_000_000:
        return f"IDR {value / 1_000_000_000:.1f}B"
    return f"IDR {value:,.0f}"


# ── Universe ────────────────────────────────────────────────────────────── #

def fetch_profiles(symbols: list[str]) -> pd.DataFrame:
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


def filter_universe(profiles: pd.DataFrame) -> pd.DataFrame:
    """Actively traded operating companies with a price and market cap, largest first."""
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


def load_idx_universe() -> pd.DataFrame:
    """FMP profiles of every listed IDX company, filtered by ``filter_universe``."""
    listed = hd.get_all_idx_stocks()
    symbols = listed["symbol"].dropna().astype(str).unique().tolist()
    return filter_universe(fetch_profiles(symbols))


def select_universe(universe: pd.DataFrame, min_market_cap: float, max_companies: int) -> pd.DataFrame:
    """The largest ``max_companies`` with a market cap of at least ``min_market_cap`` trillion IDR."""
    market_cap_floor = min_market_cap * 1_000_000_000_000
    return universe[universe["mktCap"] >= market_cap_floor].head(int(max_companies))


def profile_records(profiles: pd.DataFrame) -> tuple:
    records = []
    for symbol, row in profiles.iterrows():
        values = tuple((field, row.get(field)) for field in PROFILE_FIELDS)
        records.append((str(symbol), values))
    return tuple(records)


def profile_snapshot(profiles: pd.DataFrame) -> tuple[str, tuple]:
    records = profile_records(profiles)
    version = hashlib.sha256(repr(records).encode()).hexdigest()
    return version, records


# ── Per-stock analysis ──────────────────────────────────────────────────── #

def prepare_financials(financials: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "revenue", "netIncome", "reportedCurrency"}
    if financials is None or financials.empty or not required.issubset(financials.columns):
        return pd.DataFrame()

    frame = financials.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("revenue", "netIncome"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "revenue", "netIncome"])
    return frame.sort_values("date", ascending=False).reset_index(drop=True)


def financial_history(financials: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
        "revenue_ttm": finite(valid["Revenue TTM"].iloc[-1]),
        "earnings_ttm": finite(valid["Earnings TTM"].iloc[-1]),
        "revenue_growth_ttm": finite(valid["Revenue Growth"].iloc[-1]),
        "earnings_growth_ttm": finite(valid["Earnings Growth"].iloc[-1]),
        "revenue_growth_5y": finite(recent_growth["Revenue Growth"].median()),
        "earnings_growth_5y": finite(recent_growth["Earnings Growth"].median()),
        "margin_ttm": finite(valid["Margin"].iloc[-1]),
        "margin_median": finite(valid.tail(20)["Margin"].median()),
        "statement_date": valid["date"].iloc[-1],
    }
    return ordered, stats


def ratio_summary(
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

    current = finite(history["ratio"].iloc[-1])
    mean = finite(window["ratio"].mean())
    median = finite(window["ratio"].median())
    percentile = finite((window["ratio"] <= current).mean() * 100)
    discount = finite((median / current - 1) * 100) if current > 0 else np.nan
    return history, {
        "current": current,
        "mean": mean,
        "median": median,
        "percentile": percentile,
        "discount": discount,
        "observations": len(window),
    }


def price_start(history_years: int) -> str:
    """First price date the analysis needs: the valuation window plus a year of warm-up."""
    return (datetime.date.today() - datetime.timedelta(days=(history_years + 1) * 366)).isoformat()


def analyze_frames(
    symbol: str,
    profile: dict,
    prices: pd.DataFrame,
    financials: pd.DataFrame,
    history_years: int,
    include_history: bool = False,
):
    """One company's screen row from its prices and raw quarterly income statements.

    Raises ``SkipStock`` when the data is insufficient; other exceptions are
    real failures and are left to the caller.
    """
    price = finite(profile.get("price"))
    market_cap = finite(profile.get("mktCap"))
    shares = market_cap / price if price > 0 and market_cap > 0 else np.nan
    if not np.isfinite(shares) or shares <= 0:
        raise SkipStock("no price or market cap")

    financials = prepare_financials(financials)
    if prices is None or prices.empty:
        raise SkipStock("no prices")
    if len(financials) < 8:
        raise SkipStock("fewer than 8 quarterly statements")

    history, growth = financial_history(financials)
    pe_history, pe = ratio_summary(prices, financials, shares, "pe", history_years)
    ps_history, ps = ratio_summary(prices, financials, shares, "ps", history_years)
    if not growth:
        raise SkipStock("no trailing twelve months")
    if not pe or not ps:
        raise SkipStock("valuation history too short or not positive")

    mean_reversion = np.mean(
        [pe["mean"] / pe["current"], ps["mean"] / ps["current"]]
    )
    one_year = project_upside(
        pe["current"],
        pe["mean"],
        ps["current"],
        ps["mean"],
        growth["earnings_growth_ttm"],
        growth["revenue_growth_ttm"],
        1,
    )
    five_year = project_upside(
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
        "financial_history": history,
    }


def analyze_stock(
    symbol: str,
    profile: dict,
    history_years: int,
    include_history: bool = False,
):
    """``analyze_frames`` on FMP prices and Daguerreo statements; None when anything fails."""
    try:
        prices = hd.get_daily_stock_price(symbol, start_from=price_start(history_years))
        financials = hd.get_financial_data(symbol, period="quarter", source="dag")
        return analyze_frames(symbol, profile, prices, financials, history_years, include_history)
    except Exception:
        return None


# ── Universe scan and scoring ───────────────────────────────────────────── #

def score_results(rows: list[dict]) -> pd.DataFrame:
    """Add valuation, growth, quality and blended scores; best score first."""
    if not rows:
        return empty_results()

    result = pd.DataFrame(rows)
    result["Valuation Score"] = 100 - result[["PE Percentile", "PS Percentile"]].mean(axis=1)
    growth_columns = [
        "Revenue Growth 5Y",
        "Revenue Growth TTM",
        "Earnings Growth 5Y",
        "Earnings Growth TTM",
    ]
    result["Growth Score"] = pd.concat(
        [pct_rank(result[column]).rename(column) for column in growth_columns], axis=1
    ).mean(axis=1)
    result["Quality Score"] = pct_rank(result["Margin TTM"])
    result["Score"] = (
        result["Valuation Score"] * 0.55
        + result["Growth Score"] * 0.35
        + result["Quality Score"] * 0.10
    )
    result["Statement Date"] = pd.to_datetime(result["Statement Date"]).dt.date
    return result.sort_values("Score", ascending=False).reset_index(drop=True)


def scan_profiles(profiles: dict[str, dict], history_years: int) -> pd.DataFrame:
    """Analyze every profile in parallel and score the ones with enough data."""
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
        futures = {
            executor.submit(analyze_stock, symbol, profile, history_years): symbol
            for symbol, profile in profiles.items()
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
            except Exception:
                row = None
            if row:
                rows.append(row)
    return score_results(rows)


# ── Screen rules ────────────────────────────────────────────────────────── #

def screen_rules(results: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """One boolean column per rule, named by the short label shown when a company fails it."""
    rules = {
        "Loss-making": results["Earnings TTM"] > 0,
        "No revenue": results["Revenue TTM"] > 0,
        "Negative P/E": results["Current PE"] > 0,
        f"P/E > {filters['max_pe']:g}": results["Current PE"] <= filters["max_pe"],
        "Negative P/S": results["Current PS"] > 0,
        f"Revenue 5Y < {filters['min_revenue_growth']:g}%": (
            results["Revenue Growth 5Y"] >= filters["min_revenue_growth"]
        ),
        f"Earnings 5Y < {filters['min_earnings_growth']:g}%": (
            results["Earnings Growth 5Y"] >= filters["min_earnings_growth"]
        ),
    }
    if filters["require_recent_growth"]:
        rules["Revenue TTM falling"] = results["Revenue Growth TTM"] > 0
        rules["Earnings TTM falling"] = results["Earnings Growth TTM"] > 0
    if filters["require_both_discounts"]:
        rules["P/E above median"] = results["PE Discount"] > 0
        rules["P/S above median"] = results["PS Discount"] > 0
    return pd.DataFrame(rules)


def apply_screen(results: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """``results`` with ``Passes All`` and a comma-separated ``Failed Rules`` column."""
    results = results.copy()
    rule_checks = screen_rules(results, filters)
    results["Passes All"] = rule_checks.all(axis=1)
    results["Failed Rules"] = rule_checks.apply(
        lambda row: ", ".join(label for label, passed in row.items() if not passed), axis=1
    )
    return results


def ranked(results: pd.DataFrame) -> pd.DataFrame:
    """Sorted by score with a 1-based ``Rank`` index."""
    table = results.sort_values("Score", ascending=False).reset_index(drop=True)
    table.index += 1
    table.index.name = "Rank"
    return table


def checklist(row: dict) -> pd.DataFrame:
    """The detail view's pass/review table for one company."""
    checks = pd.DataFrame(
        [
            (
                "Earnings valuation",
                row["PE Discount"] > 0,
                median_comparison(row["PE Discount"], "P/E"),
            ),
            (
                "Revenue valuation",
                row["PS Discount"] > 0,
                median_comparison(row["PS Discount"], "P/S"),
            ),
            ("Durable revenue growth", row["Revenue Growth 5Y"] > 0, f"5Y median growth is {row['Revenue Growth 5Y']:.1f}%"),
            ("Durable earnings growth", row["Earnings Growth 5Y"] > 0, f"5Y median growth is {row['Earnings Growth 5Y']:.1f}%"),
            ("Recent revenue growth", row["Revenue Growth TTM"] > 0, f"Latest TTM growth is {row['Revenue Growth TTM']:.1f}%"),
            ("Recent earnings growth", row["Earnings Growth TTM"] > 0, f"Latest TTM growth is {row['Earnings Growth TTM']:.1f}%"),
            ("Profitable", row["Earnings TTM"] > 0, f"Net margin is {row['Margin TTM']:.1f}%"),
        ],
        columns=["Test", "Pass", "Evidence"],
    )
    checks["Status"] = np.where(checks["Pass"], "Pass", "Review")
    return checks


def profile_values(profile: pd.Series) -> tuple:
    return tuple((field, profile.get(field)) for field in PROFILE_FIELDS)


def detail_is_current(detail_row: dict) -> bool:
    """False for a cached detail from before the projection columns existed."""
    return {
        "Current Price",
        "Upside to Mean",
        "Potential Upside 1Y",
        "Potential Upside 5Y",
        "Implied Price 1Y",
        "Implied Price 5Y",
    }.issubset(detail_row)
