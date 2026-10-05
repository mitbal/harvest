"""Long Term engine ("Growth at a Discount"): valuation history, growth, scoring and projections.

Each company is compared with its own history: P/E and P/S (P/B for banks)
against their means and medians over the chosen window, and trailing-twelve-
month revenue and earnings growth to confirm a low multiple is not a
shrinking business. Quality checks from the balance sheet and cash flow
statement screen out cheap companies that are cheap for a reason.

Streamlit-free so the page and the pipeline jobs share one implementation.
"""
import concurrent.futures
import datetime
import hashlib
from typing import Optional

import numpy as np
import pandas as pd

import harvest.data as hd


SCAN_SCHEMA_VERSION = 4
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
    "Value Basis",
    "Current PB",
    "PB Mean",
    "PB Median",
    "PB Discount",
    "PB Percentile",
    "Debt to Equity",
    "Interest Coverage",
    "Cash Conversion",
    "ROE",
    "Share Change 1Y",
    "Quality Data",
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
    "require_quality": True,
}
# Quality thresholds. Leverage, interest cover and cash conversion don't describe a bank,
# so banks are held to ROE and dilution only.
MAX_DEBT_TO_EQUITY = 1.5
MIN_INTEREST_COVERAGE = 3.0
MIN_CASH_CONVERSION = 80.0
MIN_ROE = 8.0
MAX_SHARE_CHANGE = 5.0
CASH_CONVERSION_QUARTERS = 12
# FMP statement fields the quality checks and P/B need, by statement.
STATEMENT_FIELDS = {
    "income-statement": ["interestExpense", "incomeBeforeTax", "weightedAverageShsOut"],
    "balance-sheet-statement": ["totalDebt", "totalStockholdersEquity"],
    "cash-flow-statement": ["operatingCashFlow", "netIncome"],
}
MONETARY_FIELDS = [
    "interestExpense", "incomeBeforeTax", "totalDebt", "totalStockholdersEquity",
    "operatingCashFlow", "cashFlowNetIncome",
]


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


def as_dates(values) -> pd.Series:
    """Dates in one resolution: ``merge_asof`` refuses to join R2's and FMP's different units."""
    return pd.to_datetime(values, errors="coerce").astype("datetime64[us]")


def is_bank(profile: dict) -> bool:
    """Banks are valued on book value, not sales (FMP industry "Banks - ...")."""
    return str(profile.get("industry") or "").startswith("Banks")


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
    frame["date"] = as_dates(frame["date"])
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
    return history, summarize_ratio(history, history_years)


def summarize_ratio(history: pd.DataFrame, history_years: int) -> dict:
    """Current, mean, median, percentile and median discount of a positive ratio history."""
    if history.empty:
        return {}
    cutoff = history["date"].max() - pd.DateOffset(years=history_years)
    window = history[history["date"] >= cutoff]
    if len(window) < 60:
        return {}

    current = finite(history["ratio"].iloc[-1])
    mean = finite(window["ratio"].mean())
    median = finite(window["ratio"].median())
    percentile = finite((window["ratio"] <= current).mean() * 100)
    discount = finite((median / current - 1) * 100) if current > 0 else np.nan
    return {
        "current": current,
        "mean": mean,
        "median": median,
        "percentile": percentile,
        "discount": discount,
        "observations": len(window),
    }


# ── Balance sheet and cash flow ─────────────────────────────────────────── #

def load_statements(symbol: str) -> pd.DataFrame:
    """Quarterly FMP fields for the quality checks and P/B; raises when FMP fails."""
    frames = [
        hd.get_fmp_statement(symbol, statement)
        for statement in STATEMENT_FIELDS
    ]
    return prepare_statements(*frames)


def prepare_statements(income: pd.DataFrame, balance: pd.DataFrame, cash_flow: pd.DataFrame) -> pd.DataFrame:
    """One row per quarter, oldest first; the cash flow statement's net income is ``cashFlowNetIncome``.

    ``reportedCurrency`` is the first currency any statement reports.
    """
    merged = None
    currency = None
    for frame, (statement, fields) in zip((income, balance, cash_flow), STATEMENT_FIELDS.items()):
        if frame is None or frame.empty or "date" not in frame:
            frame = pd.DataFrame(columns=["date", *fields])
        if currency is None and "reportedCurrency" in frame and frame["reportedCurrency"].notna().any():
            currency = str(frame["reportedCurrency"].dropna().iloc[0])
        part = frame.reindex(columns=["date", *fields]).copy()
        if statement == "cash-flow-statement":
            part = part.rename(columns={"netIncome": "cashFlowNetIncome"})
        part["date"] = as_dates(part["date"])
        part = part.dropna(subset=["date"]).drop_duplicates("date")
        merged = part if merged is None else merged.merge(part, on="date", how="outer")
    for column in merged.columns.drop("date"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["reportedCurrency"] = currency or "IDR"
    return merged.sort_values("date").reset_index(drop=True)


def usd_idr_rate(currency: Optional[str]) -> float:
    """Multiplier from a reporting currency to IDR (as ``calc_ratio_history`` converts P/E and P/S)."""
    return float(hd.get_usd_idr_rate()) if currency == "USD" else 1.0


def statements_in_idr(statements: pd.DataFrame) -> pd.DataFrame:
    if statements.empty:
        return statements
    rate = usd_idr_rate(statements["reportedCurrency"].iloc[0])
    if rate == 1.0:
        return statements
    converted = statements.copy()
    converted[MONETARY_FIELDS] = converted[MONETARY_FIELDS] * rate
    converted["reportedCurrency"] = "IDR"
    return converted


def book_ratio_summary(
    prices: pd.DataFrame,
    statements: pd.DataFrame,
    shares: float,
    history_years: int,
) -> tuple[pd.DataFrame, dict]:
    """Daily P/B from quarterly equity per share (same share count as P/E and P/S)."""
    book = statements[["date", "totalStockholdersEquity"]].dropna()
    book = book[book["totalStockholdersEquity"] > 0]
    if book.empty or not np.isfinite(shares) or shares <= 0:
        return pd.DataFrame(columns=["date", "ratio"]), {}
    book = book.assign(bvps=book["totalStockholdersEquity"] / shares).sort_values("date")
    daily = prices[["date", "close"]].copy()
    daily["date"] = as_dates(daily["date"])
    daily = pd.merge_asof(daily.sort_values("date"), book[["date", "bvps"]], on="date", direction="backward")
    history = pd.DataFrame({"date": daily["date"], "ratio": daily["close"] / daily["bvps"]})
    history = history.replace([np.inf, -np.inf], np.nan).dropna()
    history = history[history["ratio"] > 0].sort_values("date")
    return history, summarize_ratio(history, history_years)


def book_growth(statements: pd.DataFrame) -> dict:
    """Year-over-year growth of equity per reported share: latest and five-year median."""
    frame = statements[["date", "totalStockholdersEquity", "weightedAverageShsOut"]].dropna()
    frame = frame[(frame["totalStockholdersEquity"] > 0) & (frame["weightedAverageShsOut"] > 0)]
    if len(frame) < 5:
        return {"ttm": np.nan, "5y": np.nan}
    per_share = frame["totalStockholdersEquity"] / frame["weightedAverageShsOut"]
    growth = (per_share.pct_change(4) * 100).dropna()
    return {"ttm": finite(growth.iloc[-1]), "5y": finite(growth.tail(20).median())}


def quality_metrics(
    statements: Optional[pd.DataFrame],
    earnings_ttm: float,
    net_income: Optional[pd.DataFrame] = None,
) -> dict:
    """Leverage, interest cover, cash conversion, ROE and dilution from the latest quarters.

    ROE uses the same trailing earnings as P/E over the average of the latest
    and year-ago equity. Cash conversion divides operating cash flow by
    ``net_income`` (``date, netIncome``, the statements P/E uses) when given,
    else by the cash flow statement's own net income. Interest cover is
    infinite without interest expense. Monetary values must share a currency.
    """
    metrics = {key: np.nan for key in ("Debt to Equity", "Interest Coverage", "Cash Conversion", "ROE", "Share Change 1Y")}
    if statements is None or statements.empty:
        return metrics
    frame = statements.sort_values("date")

    equity = frame["totalStockholdersEquity"].dropna()
    debt = frame.dropna(subset=["totalDebt", "totalStockholdersEquity"])
    if not debt.empty:
        latest = debt.iloc[-1]
        metrics["Debt to Equity"] = (
            finite(latest["totalDebt"] / latest["totalStockholdersEquity"])
            if latest["totalStockholdersEquity"] > 0 else np.inf
        )

    income = frame.dropna(subset=["interestExpense", "incomeBeforeTax"]).tail(4)
    if len(income) == 4:
        interest = income["interestExpense"].abs().sum()
        ebit = income["incomeBeforeTax"].sum() + interest
        metrics["Interest Coverage"] = ebit / interest if interest > 0 else np.inf

    if net_income is not None:
        frame = frame.drop(columns="cashFlowNetIncome").merge(
            net_income[["date", "netIncome"]].rename(columns={"netIncome": "cashFlowNetIncome"}),
            on="date", how="left",
        )
    cash = frame.dropna(subset=["operatingCashFlow", "cashFlowNetIncome"]).tail(CASH_CONVERSION_QUARTERS)
    if len(cash) == CASH_CONVERSION_QUARTERS and cash["cashFlowNetIncome"].sum() > 0:
        metrics["Cash Conversion"] = finite(cash["operatingCashFlow"].sum() / cash["cashFlowNetIncome"].sum() * 100)

    if len(equity) >= 5 and np.isfinite(earnings_ttm):
        average = (equity.iloc[-1] + equity.iloc[-5]) / 2
        if average > 0:
            metrics["ROE"] = finite(earnings_ttm / average * 100)

    shares = frame["weightedAverageShsOut"].dropna()
    shares = shares[shares > 0]
    if len(shares) >= 5:
        metrics["Share Change 1Y"] = finite((shares.iloc[-1] / shares.iloc[-5] - 1) * 100)
    return metrics


def has_quality_data(metrics: dict, bank: bool) -> bool:
    """Every metric the stock is judged on is available (an infinite interest cover counts)."""
    needed = ["ROE", "Share Change 1Y"] if bank else list(metrics)
    return all(not pd.isna(metrics[key]) for key in needed)


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
    statements: Optional[pd.DataFrame] = None,
):
    """One company's screen row from its prices, quarterly income statements and FMP statements.

    ``statements`` (from ``load_statements``) adds P/B and the quality
    metrics; banks need it, other companies are valued without it. Raises
    ``SkipStock`` when the data is insufficient; other exceptions are real
    failures and are left to the caller.
    """
    price = finite(profile.get("price"))
    market_cap = finite(profile.get("mktCap"))
    shares = market_cap / price if price > 0 and market_cap > 0 else np.nan
    if not np.isfinite(shares) or shares <= 0:
        raise SkipStock("no price or market cap")

    financials = prepare_financials(financials)
    if prices is None or prices.empty:
        raise SkipStock("no prices")
    prices = prices.assign(date=as_dates(prices["date"]))
    if len(financials) < 8:
        raise SkipStock("fewer than 8 quarterly statements")

    bank = is_bank(profile)
    statements = statements_in_idr(
        prepare_statements(None, None, None) if statements is None else statements
    )
    history, growth = financial_history(financials)
    pe_history, pe = ratio_summary(prices, financials, shares, "pe", history_years)
    ps_history, ps = ratio_summary(prices, financials, shares, "ps", history_years)
    pb_history, pb = book_ratio_summary(prices, statements, shares, history_years)
    if not growth:
        raise SkipStock("no trailing twelve months")
    second = pb if bank else ps
    if not pe or not second:
        raise SkipStock("valuation history too short or not positive")

    books = book_growth(statements)
    second_growth_ttm = books["ttm"] if bank else growth["revenue_growth_ttm"]
    second_growth_5y = books["5y"] if bank else growth["revenue_growth_5y"]
    mean_reversion = np.mean(
        [pe["mean"] / pe["current"], second["mean"] / second["current"]]
    )
    one_year = project_upside(
        pe["current"],
        pe["mean"],
        second["current"],
        second["mean"],
        growth["earnings_growth_ttm"],
        second_growth_ttm,
        1,
    )
    five_year = project_upside(
        pe["current"],
        pe["mean"],
        second["current"],
        second["mean"],
        growth["earnings_growth_5y"],
        second_growth_5y,
        5,
    )
    # Quality metrics compare income with the statements in one currency (IDR).
    income_rate = usd_idr_rate(str(financials["reportedCurrency"].dropna().iloc[0]))
    quality = quality_metrics(
        statements,
        growth["earnings_ttm"] * income_rate,
        financials[["date", "netIncome"]].assign(netIncome=financials["netIncome"] * income_rate),
    )
    leg = "Book" if bank else "Revenue"
    other = "Revenue" if bank else "Book"

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
        "Current PS": ps.get("current", np.nan),
        "PS Mean": ps.get("mean", np.nan),
        "PS Median": ps.get("median", np.nan),
        "PS Discount": ps.get("discount", np.nan),
        "PS Percentile": ps.get("percentile", np.nan),
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
        f"{leg} Upside 1Y": one_year["revenue"],
        "Earnings Upside 5Y": five_year["earnings"],
        f"{leg} Upside 5Y": five_year["revenue"],
        "Earnings Growth Assumption 1Y": one_year["earnings_growth"],
        f"{leg} Growth Assumption 1Y": one_year["revenue_growth"],
        "Earnings Growth Assumption 5Y": five_year["earnings_growth"],
        f"{leg} Growth Assumption 5Y": five_year["revenue_growth"],
        **{f"{other} {key}": np.nan for key in ("Upside 1Y", "Upside 5Y", "Growth Assumption 1Y", "Growth Assumption 5Y")},
        "Implied Price 1Y": price * (1 + one_year["blended"] / 100),
        "Implied Price 5Y": price * (1 + five_year["blended"] / 100),
        "Statement Date": growth["statement_date"],
        "Value Basis": "P/B" if bank else "P/S",
        "Current PB": pb.get("current", np.nan),
        "PB Mean": pb.get("mean", np.nan),
        "PB Median": pb.get("median", np.nan),
        "PB Discount": pb.get("discount", np.nan),
        "PB Percentile": pb.get("percentile", np.nan),
        "Book Growth TTM": books["ttm"],
        "Book Growth 5Y": books["5y"],
        **quality,
        "Quality Data": has_quality_data(quality, bank),
    }
    if not include_history:
        return row
    return {
        "row": row,
        "pe_history": pe_history,
        "ps_history": ps_history,
        "pb_history": pb_history,
        "financial_history": history,
    }


def analyze_stock(
    symbol: str,
    profile: dict,
    history_years: int,
    include_history: bool = False,
):
    """``analyze_frames`` on FMP prices and statements and Daguerreo income; None when it fails.

    A failed statements request leaves the quality data missing (and skips banks).
    """
    try:
        prices = hd.get_daily_stock_price(symbol, start_from=price_start(history_years))
        financials = hd.get_financial_data(symbol, period="quarter", source="dag")
        try:
            statements = load_statements(symbol)
        except Exception:
            statements = None
        return analyze_frames(
            symbol, profile, prices, financials, history_years, include_history, statements=statements,
        )
    except Exception:
        return None


# ── Universe scan and scoring ───────────────────────────────────────────── #

def score_results(rows: list[dict]) -> pd.DataFrame:
    """Add valuation, growth, quality and blended scores; best score first."""
    if not rows:
        return empty_results()

    result = pd.DataFrame(rows)
    if "Value Basis" not in result:
        result["Value Basis"] = "P/S"
    second = result["PS Percentile"].where(result["Value Basis"] != "P/B", result.get("PB Percentile"))
    result["Valuation Score"] = 100 - pd.concat([result["PE Percentile"], second], axis=1).mean(axis=1)
    growth_columns = [
        "Revenue Growth 5Y",
        "Revenue Growth TTM",
        "Earnings Growth 5Y",
        "Earnings Growth TTM",
    ]
    result["Growth Score"] = pd.concat(
        [pct_rank(result[column]).rename(column) for column in growth_columns], axis=1
    ).mean(axis=1)
    quality_columns = [column for column in ("Margin TTM", "ROE") if column in result]
    result["Quality Score"] = pd.concat(
        [pct_rank(result[column]).rename(column) for column in quality_columns], axis=1
    ).mean(axis=1).fillna(50.0)
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
    """One boolean column per rule, named by the short label shown when a company fails it.

    Banks are judged on P/B where other companies are judged on P/S, and only
    on the quality rules that describe a bank.
    """
    bank = _series(results, "Value Basis", "P/S") == "P/B"
    rules = {
        "Loss-making": results["Earnings TTM"] > 0,
        "No revenue": results["Revenue TTM"] > 0,
        "Negative P/E": results["Current PE"] > 0,
        f"P/E > {filters['max_pe']:g}": results["Current PE"] <= filters["max_pe"],
        "Negative P/S": bank | (results["Current PS"] > 0),
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
        rules["P/S above median"] = bank | (results["PS Discount"] > 0)
        rules["P/B above median"] = ~bank | (_series(results, "PB Discount") > 0)
    if filters.get("require_quality"):
        rules["Quality data missing"] = _series(results, "Quality Data", False).fillna(False).astype(bool)
        # A missing metric is reported once, by the rule above.
        leverage = _series(results, "Debt to Equity")
        coverage = _series(results, "Interest Coverage")
        conversion = _series(results, "Cash Conversion")
        roe = _series(results, "ROE")
        dilution = _series(results, "Share Change 1Y")
        rules[f"Debt/equity > {MAX_DEBT_TO_EQUITY:g}"] = bank | leverage.isna() | (leverage <= MAX_DEBT_TO_EQUITY)
        rules[f"Interest cover < {MIN_INTEREST_COVERAGE:g}x"] = bank | coverage.isna() | (coverage >= MIN_INTEREST_COVERAGE)
        rules[f"Cash flow < {MIN_CASH_CONVERSION:g}% of profit"] = (
            bank | conversion.isna() | (conversion >= MIN_CASH_CONVERSION)
        )
        rules[f"ROE < {MIN_ROE:g}%"] = roe.isna() | (roe >= MIN_ROE)
        rules[f"Shares up > {MAX_SHARE_CHANGE:g}%"] = dilution.isna() | (dilution <= MAX_SHARE_CHANGE)
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


def _metric_check(test: str, value, passed, evidence: str):
    if pd.isna(value):
        return (test, False, "Not reported")
    return (test, bool(passed), evidence)


def checklist(row: dict) -> pd.DataFrame:
    """The detail view's pass/review table for one company."""
    bank = row.get("Value Basis") == "P/B"
    second = (
        ("Book valuation", row.get("PB Discount", np.nan) > 0, median_comparison(row.get("PB Discount", np.nan), "P/B"))
        if bank else
        ("Revenue valuation", row["PS Discount"] > 0, median_comparison(row["PS Discount"], "P/S"))
    )
    rows = [
        (
            "Earnings valuation",
            row["PE Discount"] > 0,
            median_comparison(row["PE Discount"], "P/E"),
        ),
        second,
        ("Durable revenue growth", row["Revenue Growth 5Y"] > 0, f"5Y median growth is {row['Revenue Growth 5Y']:.1f}%"),
        ("Durable earnings growth", row["Earnings Growth 5Y"] > 0, f"5Y median growth is {row['Earnings Growth 5Y']:.1f}%"),
        ("Recent revenue growth", row["Revenue Growth TTM"] > 0, f"Latest TTM growth is {row['Revenue Growth TTM']:.1f}%"),
        ("Recent earnings growth", row["Earnings Growth TTM"] > 0, f"Latest TTM growth is {row['Earnings Growth TTM']:.1f}%"),
        ("Profitable", row["Earnings TTM"] > 0, f"Net margin is {row['Margin TTM']:.1f}%"),
    ]
    if "ROE" in row:
        roe, dilution = row["ROE"], row["Share Change 1Y"]
        rows.append(_metric_check("Return on equity", roe, roe >= MIN_ROE, f"ROE is {roe:.1f}% (at least {MIN_ROE:g}%)"))
        rows.append(_metric_check(
            "No heavy dilution", dilution, dilution <= MAX_SHARE_CHANGE,
            f"Share count {'up' if dilution >= 0 else 'down'} {abs(dilution):.1f}% in a year",
        ))
        if not bank:
            leverage, coverage, conversion = row["Debt to Equity"], row["Interest Coverage"], row["Cash Conversion"]
            rows.append(_metric_check(
                "Moderate debt", leverage, leverage <= MAX_DEBT_TO_EQUITY,
                f"Debt is {leverage:.2f}x equity (at most {MAX_DEBT_TO_EQUITY:g}x)",
            ))
            rows.append(_metric_check(
                "Interest covered", coverage, coverage >= MIN_INTEREST_COVERAGE,
                "No interest expense" if coverage == np.inf
                else f"Profit covers interest {coverage:.1f}x (at least {MIN_INTEREST_COVERAGE:g}x)",
            ))
            rows.append(_metric_check(
                "Profit backed by cash", conversion, conversion >= MIN_CASH_CONVERSION,
                f"Operating cash flow is {conversion:.0f}% of net income over 3 years",
            ))
    checks = pd.DataFrame(rows, columns=["Test", "Pass", "Evidence"])
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
