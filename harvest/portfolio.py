"""Portfolio calculations behind the Portfolio Analytics page.

No Streamlit, Redis or network calls, so the page's maths can be tested
directly. Prices and dividends are per share; IDX trades in
lots of ``LOT_SIZE`` shares.
"""
import io
import json
import logging

import numpy as np
import pandas as pd

import harvest.data as hd

logger = logging.getLogger(__name__)

LOT_SIZE = 100
REQUIRED_COLUMNS = {'Symbol', 'Available Lot', 'Average Price'}
PASTE_RAW_FIELDS_PER_ROW = 11  # expected tokens per stock row in raw paste format
PROFILE_COLUMNS = ['price', 'sector', 'lastDiv']
OPTIONAL_PROFILE_COLUMNS = ['price_date']  # absent from payloads published before it existed
DIVIDEND_HISTORY_COLUMNS = {'date', 'adjDividend', 'fiscal_year', 'dividend_type'}


def clean_symbols(symbols) -> pd.Series:
    """Upper-case, trim and drop the ``.JK`` suffix so tickers match across sources."""
    return (
        pd.Series(symbols, dtype='string')
        .str.strip()
        .str.upper()
        .str.removesuffix('.JK')
    )


def normalize_portfolio(portfolio) -> pd.DataFrame:
    """Validate portfolio input and combine duplicate symbols at weighted cost."""
    df = pd.DataFrame(portfolio).copy(deep=True)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        missing = ', '.join(sorted(missing_cols))
        raise ValueError(f'Missing required columns: {missing}.')

    df = df[['Symbol', 'Available Lot', 'Average Price']].dropna(how='all')
    df['Symbol'] = clean_symbols(df['Symbol']).values
    df = df[df['Symbol'].notna() & (df['Symbol'] != '')].copy()
    if df.empty:
        raise ValueError('Add at least one stock with a symbol, lot count, and average price.')

    for column in ['Available Lot', 'Average Price']:
        cleaned = df[column].astype('string').str.replace(',', '', regex=False).str.strip()
        df[column] = pd.to_numeric(cleaned, errors='coerce')
        invalid = df[column].isna() | ~np.isfinite(df[column]) | (df[column] <= 0)
        if invalid.any():
            symbols = ', '.join(df.loc[invalid, 'Symbol'].astype(str).tolist())
            raise ValueError(f'{column} must be a positive number. Check: {symbols}.')

    df['_invested'] = df['Available Lot'] * df['Average Price']
    grouped = df.groupby('Symbol', sort=False, as_index=False).agg(
        {'Available Lot': 'sum', '_invested': 'sum'}
    )
    grouped['Average Price'] = grouped['_invested'] / grouped['Available Lot']
    return grouped[['Symbol', 'Available Lot', 'Average Price']]


def parse_raw_portfolio(raw: str) -> pd.DataFrame:
    """Parse a Stockbit portfolio table copied as whitespace-delimited text."""
    if not raw or not raw.strip():
        raise ValueError('Paste your Stockbit portfolio data before loading it.')

    rows = raw.split()
    if len(rows) % PASTE_RAW_FIELDS_PER_ROW != 0:
        raise ValueError(
            f'Pasted data contains {len(rows)} fields; expected a multiple of '
            f'{PASTE_RAW_FIELDS_PER_ROW}. Copy the complete Stockbit table and try again.'
        )

    portfolio = pd.DataFrame({
        'Symbol': rows[0::PASTE_RAW_FIELDS_PER_ROW],
        'Available Lot': rows[1::PASTE_RAW_FIELDS_PER_ROW],
        'Average Price': rows[3::PASTE_RAW_FIELDS_PER_ROW],
    })
    return normalize_portfolio(portfolio)


def portfolio_to_records(portfolio) -> list[dict]:
    """Return a JSON-safe, record-oriented portfolio payload."""
    normalized = normalize_portfolio(portfolio)
    return json.loads(normalized.to_json(orient='records'))


def parse_company_profiles(raw) -> pd.DataFrame:
    """Parse the ``div_score_jkse`` Redis payload (parquet or JSON) into one row per symbol."""
    try:
        if isinstance(raw, bytes) and raw.startswith(b'PAR1'):
            profiles = pd.read_parquet(io.BytesIO(raw))
        else:
            payload = json.loads(raw)
            if isinstance(payload, dict) and 'content' in payload:
                profiles = pd.DataFrame(json.loads(payload['content']))
            else:
                profiles = pd.DataFrame(payload)
    except Exception as e:
        raise ValueError(f'Failed to parse company profile data from Redis: {e}') from e

    if 'symbol' in profiles.columns:
        profiles = profiles.set_index('symbol')

    for col in PROFILE_COLUMNS:
        if col not in profiles.columns:
            raise KeyError(f'Expected column "{col}" not found in company profile data.')

    return with_symbol_column(profiles)


def with_symbol_column(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add a cleaned ``Symbol`` column from a symbol-indexed profile frame, keeping one row per symbol."""
    profiles = profiles.copy(deep=True)
    if 'symbol' in profiles.columns:
        profiles = profiles.set_index('symbol')
    profiles['Symbol'] = clean_symbols(pd.Index(profiles.index).astype(str)).values
    for col in PROFILE_COLUMNS:
        if col not in profiles.columns:
            profiles[col] = np.nan
    return profiles.drop_duplicates(subset='Symbol', keep='first')


def annual_dividend_from_history(div_df, fallback: float) -> float:
    """Return the latest finalized fiscal-year dividend, or ``fallback`` when the history can't say."""
    if div_df is None or not DIVIDEND_HISTORY_COLUMNS.issubset(div_df.columns):
        return fallback
    return hd.calc_latest_finalized_dividend_sum(div_df)


def merge_market_data(portfolio: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join holdings to market profiles; return matched rows and the symbols with no profile."""
    optional = [column for column in OPTIONAL_PROFILE_COLUMNS if column in profiles.columns]
    df = portfolio.merge(
        profiles[['Symbol'] + PROFILE_COLUMNS + optional],
        on='Symbol',
        how='left',
        validate='one_to_one',
        indicator=True,
    )
    df = df.rename(columns={'lastDiv': 'div_rate', 'price': 'last_price'})

    unknown = df.loc[df['_merge'] == 'left_only', 'Symbol'].tolist()
    if len(unknown) == len(df):
        raise ValueError(
            'Market data is unavailable for: '
            f'{", ".join(unknown)}. Remove or correct these symbols before continuing.'
        )

    return df[df['_merge'] == 'both'].drop(columns='_merge').reset_index(drop=True), unknown


def validate_market_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop holdings with unusable market fields; return the valid rows and the excluded symbols."""
    validated = df.copy(deep=True)
    rules = {
        'last_price': False,  # must be > 0
        'div_rate': True,     # may be 0
    }
    invalid = pd.Series(False, index=validated.index)
    for column, allow_zero in rules.items():
        validated[column] = pd.to_numeric(validated[column], errors='coerce')
        column_invalid = validated[column].isna() | ~np.isfinite(validated[column])
        column_invalid |= validated[column] < 0 if allow_zero else validated[column] <= 0
        invalid |= column_invalid

    excluded = validated.loc[invalid, 'Symbol'].astype(str).tolist()
    validated = validated[~invalid].reset_index(drop=True)
    if validated.empty:
        raise ValueError(f'Market price or dividend data is invalid for: {", ".join(excluded)}.')

    validated['sector'] = validated['sector'].fillna('Unclassified').replace('', 'Unclassified')
    return validated, excluded


def _format_date(date: pd.Timestamp) -> str:
    return f'{date.day} {date:%b %Y}'


def market_data_caption(df: pd.DataFrame) -> str:
    """Describe which close the holdings' prices come from, naming any holding off that date."""
    dates = pd.to_datetime(df['price_date'], errors='coerce') if 'price_date' in df.columns else None
    if dates is None or dates.isna().all():
        return 'Market prices and dividends come from the daily data pipeline.'

    as_of = dates.max()
    caption = f'Market prices are closing prices as of **{_format_date(as_of)}**.'
    older = df.loc[dates < as_of, 'Symbol'].astype(str).tolist()
    if older:
        caption += f' Older close for {", ".join(older)}.'
    live = df.loc[dates.isna(), 'Symbol'].astype(str).tolist()
    if live:
        caption += f' Live quote for {", ".join(live)}.'
    return caption


def enrich_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-holding cost, value, yield and income columns."""
    enriched = df.copy(deep=True)
    enriched['current_lot'] = enriched['Available Lot'].astype(float)
    enriched['avg_price'] = enriched['Average Price'].astype(float)
    shares = enriched['current_lot'] * LOT_SIZE

    enriched['total_invested'] = shares * enriched['avg_price']
    enriched['market_value'] = shares * enriched['last_price']
    enriched['yield_on_cost'] = enriched['div_rate'] / enriched['avg_price'] * 100
    enriched['yield_on_price'] = enriched['div_rate'] / enriched['last_price'] * 100
    enriched['total_dividend'] = (shares * enriched['div_rate']).astype(int)
    return enriched


def summarize_portfolio(holdings: pd.DataFrame, target_million: float) -> dict:
    """Return portfolio-level KPIs from ``enrich_holdings`` output.

    ``target_million`` is the annual income goal in millions of IDR.
    """
    annual_dividend = holdings['total_dividend'].sum()
    total_invested = holdings['total_invested'].sum()
    market_value = holdings['market_value'].sum()
    return {
        'annual_dividend': annual_dividend,
        'total_invested': total_invested,
        'market_value': market_value,
        'market_delta': market_value - total_invested,
        'target_progress': annual_dividend / (target_million * 1_000_000) * 100 if target_million > 0 else 0.0,
        'yield_on_cost': annual_dividend / total_invested * 100 if total_invested > 0 else 0.0,
        'yield_on_price': annual_dividend / market_value * 100 if market_value > 0 else 0.0,
    }


def build_last_year_dividends(holdings: pd.DataFrame, divs: dict, today=None) -> list[pd.DataFrame]:
    """Return each holding's dividend payments from the trailing 12 months, tagged with symbol and lots."""
    end_date = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    start_date = end_date - pd.Timedelta(days=365)

    div_lists = []
    for row in holdings.itertuples(index=False):
        symbol = row.Symbol
        history = divs.get(f'{symbol}.JK')
        if history is None:
            continue

        try:
            div_df = pd.DataFrame(history)
            if div_df.empty or 'date' not in div_df.columns or 'adjDividend' not in div_df.columns:
                continue

            # Accept strings, naive or tz-aware timestamps
            div_df['date'] = pd.to_datetime(div_df['date'], errors='coerce', utc=True).dt.tz_localize(None)
            in_window = (div_df['date'] >= start_date) & (div_df['date'] < end_date)
            last_year_div = div_df[in_window].copy(deep=True)
            if last_year_div.empty:
                continue

            last_year_div['Symbol'] = symbol
            last_year_div['Lot'] = row.current_lot
            last_year_div['yield'] = last_year_div['adjDividend'] / row.last_price * 100
            div_lists.append(last_year_div)
        except Exception as e:
            logger.warning(f'Skipping dividend timeline entry for {symbol}: {e}')

    return div_lists


def project_income(annual_dividend: float, initial_cost: float, growth_pct: float, years: int) -> pd.DataFrame:
    """Project income when all dividends are reinvested and income grows at growth_pct per year.

    Year N is N years from today. Yield is measured against the original cost plus all
    dividends reinvested before that year, since reinvested cash is also invested capital.
    """
    year_numbers = np.arange(1, years + 1)
    returns = annual_dividend * (1 + growth_pct / 100) ** year_numbers
    # Income received in years 0..N-1 has been reinvested by year N
    previous_income = np.concatenate([[annual_dividend], returns[:-1]])
    cost_basis = initial_cost + np.cumsum(previous_income)
    return pd.DataFrame({
        'years': [f'Year {n:02d}' for n in year_numbers],
        'returns': returns,
        'yield': returns / cost_basis * 100,
    })


def format_idr(value: float, signed: bool = False) -> str:
    """Format rupiah compactly for KPI cards, e.g. ``IDR 49.2M``, ``IDR 240M`` or ``-IDR 1.25B``."""
    sign = '-' if value < 0 else ('+' if signed and value > 0 else '')
    amount = abs(value)
    for threshold, suffix, decimals in ((1e12, 'T', 2), (1e9, 'B', 2), (1e6, 'M', 1), (1e3, 'K', 1)):
        if amount >= threshold:
            number = f'{amount / threshold:,.{decimals}f}'.rstrip('0').rstrip('.')
            return f'{sign}IDR {number}{suffix}'
    return f'{sign}IDR {amount:,.0f}'
