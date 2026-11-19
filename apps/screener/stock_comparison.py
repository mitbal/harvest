import io
import json
import logging
import mimetypes
import os
from typing import NamedTuple

import altair as alt
import numpy as np
import pandas as pd
import redis
import streamlit as st

import harvest.data as hd
import harvest.plot as hp
from harvest import page_ui as ui
from harvest.utils import setup_logging


mimetypes.add_type('image/svg+xml', '.svg')

st.set_page_config(page_title='Stock Comparison - Panen Dividen')
hp.enable_chart_theme()
ui.page_title(
    'Stock Comparison',
    'Compare 2-5 stocks across income, valuation, quality, growth, returns, and risk, or over time in History. '
    'Unavailable or financially invalid values are excluded rather than treated as zero.',
)

MAX_STOCKS = 5
_STOCK_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
_VIEW_LABELS = {'table': 'Table', 'dist': 'Distribution', 'scatter': 'Scatter', 'history': 'History'}
_HIST_PERIODS = {'1Y': 1, '3Y': 3, '5Y': 5, '10Y': 10}
_HIST_DEFAULT_PERIOD = '5Y'
_HIST_YEARS = 10
_DIV_MEASURES = ('Per share', 'Growth index', 'Yield at year-end price')
# A one-year window ends on the last trading day, which can be a few days (or a long
# holiday) short of 365 calendar days; still annualise those rather than show N/A.
_MIN_ANNUALISE_DAYS = 350
_MAX_CHART_POINTS = 800
_VIEW_KEYS = {label: key for key, label in _VIEW_LABELS.items()}
_MARKET_QP_MAP = {'jkse': 'Indonesian Stock', 'sp500': 'S&P 500 (US and World Stock)'}
_MARKET_QP_REV = {value: key for key, value in _MARKET_QP_MAP.items()}


class MetricSpec(NamedTuple):
    source: str
    group: str
    format: str
    unit: str
    description: str
    direction: str
    validity: str = 'finite'


METRIC_OPTIONS = {
    'Dividend Yield (%)': MetricSpec('yield', 'Dividend', '{:.2f}%', '%', 'Latest finalized fiscal-year ordinary dividend yield.', 'higher_better', 'nonnegative'),
    'Last Dividend': MetricSpec('lastDiv', 'Dividend', '{:,.2f}', 'currency/share', 'Ordinary dividends per share for the latest finalized fiscal year. Compare dividend yield instead when share prices differ.', 'neutral', 'nonnegative'),
    'Div Growth (Annual)': MetricSpec('avgFlatAnnualDivIncrease', 'Dividend', '{:,.2f}', 'currency/year', 'Average annual dividend increase per share in currency terms. Higher-priced shares naturally show larger increases.', 'neutral'),
    'Years Paying Dividend': MetricSpec('numDividendYear', 'Dividend', '{:.0f}', 'years', 'Number of years with a dividend payment.', 'higher_better', 'nonnegative'),
    'Years Raised Dividend': MetricSpec('positiveYear', 'Dividend', '{:.0f}', 'years', 'Number of years the dividend increased.', 'higher_better', 'nonnegative'),
    'Dividend Payout Ratio (%)': MetricSpec('dividendPayoutRatio', 'Dividend', '{:.1f}%', '%', 'Dividend as a share of trailing earnings. Sustainability depends on sector and cash flow.', 'neutral', 'positive'),
    'PE Ratio': MetricSpec('peRatio', 'Valuation', '{:.1f}x', 'x', 'Price divided by trailing earnings. Non-positive values are not comparable.', 'lower_better', 'positive'),
    'PS Ratio': MetricSpec('psRatio', 'Valuation', '{:.2f}x', 'x', 'Price divided by trailing sales. Non-positive values are not comparable.', 'lower_better', 'positive'),
    'Revenue Growth (5Y)': MetricSpec('revenueGrowth', 'Growth', '{:+.1f}%', '%', 'Revenue growth over the five-year measurement period.', 'higher_better'),
    'Revenue CAGR (5Y)': MetricSpec('revenueCAGR5Y', 'Growth', '{:+.1f}%', '%', 'Compound annual revenue growth over five years.', 'higher_better'),
    'Net Income Growth (5Y)': MetricSpec('netIncomeGrowth', 'Growth', '{:+.1f}%', '%', 'Net income growth over the five-year measurement period.', 'higher_better'),
    'Revenue Growth (TTM)': MetricSpec('revenueGrowthTTM', 'Growth', '{:+.1f}%', '%', 'Trailing-twelve-month revenue growth.', 'higher_better'),
    'Net Income Growth (TTM)': MetricSpec('netIncomeGrowthTTM', 'Growth', '{:+.1f}%', '%', 'Trailing-twelve-month net income growth.', 'higher_better'),
    'Revenue (TTM)': MetricSpec('revenueTTM', 'Growth', '{:,.2f}', 'currency', 'Revenue reported over the trailing twelve months. Reflects company size, not quality.', 'neutral'),
    'Net Income (TTM)': MetricSpec('earningTTM', 'Profitability', '{:,.2f}', 'currency', 'Net income reported over the trailing twelve months. Reflects company size, not quality.', 'neutral'),
    'Profit Margin (Median)': MetricSpec('medianProfitMargin', 'Profitability', '{:.1f}%', '%', 'Median historical profit margin.', 'higher_better'),
    'Profit Margin (TTM)': MetricSpec('marginTTM', 'Profitability', '{:.1f}%', '%', 'Trailing-twelve-month net profit margin.', 'higher_better'),
    '1M Return': MetricSpec('return_1m', 'Returns', '{:+.1f}%', '%', 'Price return over one month.', 'higher_better'),
    '1Y Return': MetricSpec('return_1y', 'Returns', '{:+.1f}%', '%', 'Price return over one year.', 'higher_better'),
    'Total 1Y Return': MetricSpec('total_return_1y', 'Returns', '{:+.1f}%', '%', 'One-year return including dividends.', 'higher_better'),
    '10Y Return': MetricSpec('return_10y', 'Returns', '{:+.1f}%', '%', 'Price return over ten years.', 'higher_better'),
    'Price': MetricSpec('price', 'General', '{:,.2f}', 'currency/share', 'Latest market price per share.', 'neutral', 'positive'),
    'Market Cap': MetricSpec('mktCap', 'General', '{:,.2f}', 'currency', 'Latest equity market capitalization.', 'neutral', 'positive'),
    'Beta': MetricSpec('beta', 'Risk', '{:.2f}', 'ratio', 'Price sensitivity relative to the broad market.', 'neutral'),
}

METRIC_PRESETS = {
    'Balanced': [
        'Dividend Yield (%)', 'Dividend Payout Ratio (%)', 'PE Ratio',
        'Profit Margin (TTM)', 'Revenue CAGR (5Y)', 'Total 1Y Return', 'Beta',
    ],
    'Dividend income': [
        'Dividend Yield (%)', 'Last Dividend', 'Years Paying Dividend',
        'Years Raised Dividend', 'Dividend Payout Ratio (%)',
    ],
    'Quality & growth': [
        'Profit Margin (Median)', 'Profit Margin (TTM)', 'Revenue CAGR (5Y)',
        'Revenue Growth (TTM)', 'Net Income Growth (TTM)',
    ],
    'Valuation': ['PE Ratio', 'PS Ratio', 'Dividend Yield (%)', 'Profit Margin (TTM)'],
}
TABLE_DEFAULT_METRICS = METRIC_PRESETS['Balanced']
CHART_METRIC_OPTIONS = {
    label: spec.source for label, spec in METRIC_OPTIONS.items()
    if spec.direction != 'neutral' or label in (
        'Dividend Payout Ratio (%)', 'Beta', 'Market Cap', 'Div Growth (Annual)', 'Revenue (TTM)', 'Net Income (TTM)',
    )
}

# Stocks that investors routinely weigh against each other, per market. At most MAX_STOCKS each.
STOCK_GROUPS = {
    'JKSE': {
        'Big banks': ['BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK'],
        'Regional banks': ['BJBR.JK', 'BJTM.JK', 'BNGA.JK', 'BDMN.JK', 'NISP.JK'],
        'Telco': ['TLKM.JK', 'ISAT.JK', 'EXCL.JK'],
        'Coal': ['ADRO.JK', 'ITMG.JK', 'PTBA.JK', 'BSSR.JK', 'HRUM.JK'],
        'Consumer': ['ICBP.JK', 'INDF.JK', 'UNVR.JK', 'MYOR.JK', 'SIDO.JK'],
        'Tobacco': ['HMSP.JK', 'GGRM.JK'],
        'Astra group': ['ASII.JK', 'UNTR.JK', 'AUTO.JK', 'AALI.JK'],
        'Cement': ['SMGR.JK', 'INTP.JK'],
        'Poultry': ['CPIN.JK', 'JPFA.JK'],
    },
    'S&P500': {
        'Big tech': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'],
        'Banks': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
        'Payments': ['V', 'MA', 'AXP'],
        'Oil majors': ['XOM', 'CVX', 'COP'],
        'Pharma': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY'],
        'Beverages': ['KO', 'PEP'],
        'Retail': ['WMT', 'COST', 'TGT'],
        'Home improvement': ['HD', 'LOW'],
        'Telecom': ['T', 'VZ', 'TMUS'],
    },
}


@st.cache_resource
def get_logger(name, level=logging.INFO):
    return setup_logging(name, level)


logger = get_logger('comparison')


@st.cache_resource
def connect_redis(redis_url):
    return redis.from_url(
        redis_url,
        socket_connect_timeout=10,
        socket_timeout=30,
        socket_keepalive=True,
        retry_on_timeout=True,
    )


@st.cache_data(max_entries=4, ttl=60 * 10, show_spinner='Loading stock universe...')
def get_div_score_table(key='jkse_div_score'):
    redis_url = os.environ['REDIS_URL']
    rjson = connect_redis(redis_url).get(key)
    if rjson is None:
        final_df = pd.read_csv('dividend_historical.csv')
    elif isinstance(rjson, bytes) and rjson.startswith(b'PAR1'):
        final_df = pd.read_parquet(io.BytesIO(rjson))
    else:
        raw = json.loads(rjson)
        final_df = pd.DataFrame(json.loads(raw['content'])) if isinstance(raw, dict) and 'content' in raw else pd.DataFrame(raw)

    final_df.rename(columns={'symbol': 'stock'}, inplace=True)
    cp_df = hd.get_company_profile(final_df['stock'].to_list())
    if 'price' in final_df.columns:
        final_df.rename(columns={'price': 'snapshotPrice'}, inplace=True)
    final_df = final_df.merge(
        cp_df[['price', 'changes', 'beta']], left_on='stock', right_on='symbol', how='left'
    )
    return final_df.set_index('stock')


@st.cache_data(max_entries=64, ttl=60 * 60, show_spinner=False)
def get_stock_history(stock, market):
    """Return (daily price DataFrame, dividend DataFrame or None) for about the last ten years."""
    start_from = (pd.Timestamp.today() - pd.DateOffset(years=_HIST_YEARS, days=30)).strftime('%Y-%m-%d')
    price_df = hd.get_daily_stock_price(stock, start_from=start_from)
    # raise_errors: a failed FMP request must not be cached for an hour as "pays no dividends".
    div_df = hd.get_dividend_history_single_stock(
        stock, source='dag' if market == 'JKSE' else 'fmp', raise_errors=True
    )
    return price_df, div_df


_KEEP_COLS = [
    'price', 'snapshotPrice', 'changes', 'sector', 'industry', 'mktCap', 'ipoDate', 'yield', 'lastDiv',
    'avgFlatAnnualDivIncrease', 'numDividendYear', 'positiveYear', 'numOfYear',
    'maximumCutPct', 'max10CutPct', 'peRatio', 'psRatio', 'revenueGrowth',
    'revenueCAGR5Y', 'netIncomeGrowth', 'medianProfitMargin', 'earningTTM', 'revenueTTM',
    'revenueGrowthTTM', 'netIncomeGrowthTTM', 'beta', 'return_7d', 'return_1m',
    'return_1y', 'return_10y', 'total_return_1y', 'total_return_10y', 'is_syariah',
]


@st.cache_data(max_entries=16, show_spinner=False)
def get_processed_df(df):
    df = df.copy()
    revenue = pd.to_numeric(df['revenueTTM'], errors='coerce')
    earnings = pd.to_numeric(df['earningTTM'], errors='coerce')
    df['marginTTM'] = np.where(revenue != 0, earnings / revenue * 100, np.nan)

    market_cap = pd.to_numeric(df['mktCap'], errors='coerce')
    # Shares outstanding must come from the price the market cap was computed with,
    # not the live price, or the payout ratio drifts with every price move.
    price = pd.to_numeric(df['price'], errors='coerce')
    if 'snapshotPrice' in df.columns:
        price = pd.to_numeric(df['snapshotPrice'], errors='coerce').fillna(price)
    eps_ttm = np.where((market_cap > 0) & (price > 0), earnings / (market_cap / price), np.nan)
    df['dividendPayoutRatio'] = np.where(
        eps_ttm > 0,
        pd.to_numeric(df['lastDiv'], errors='coerce') / eps_ttm * 100,
        np.nan,
    )
    df['mc_penalty'] = 1 / (1 + np.exp(-2 * (market_cap / 3_000_000_000_000 - 1)))

    for col in ('maximumCutPct', 'max10CutPct'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(upper=0) * -1
    for col in ('return_7d', 'return_1m', 'return_1y', 'return_10y', 'total_return_1y', 'total_return_10y'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') * 100

    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df


def is_valid_metric_value(value, spec):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(numeric):
        return False
    if spec.validity == 'positive':
        return numeric > 0
    if spec.validity == 'nonnegative':
        return numeric >= 0
    return True


def valid_metric_series(df, spec):
    if spec.source not in df.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(df[spec.source], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if spec.validity == 'positive':
        values = values.where(values > 0)
    elif spec.validity == 'nonnegative':
        values = values.where(values >= 0)
    return values.dropna()


def format_metric_value(label, value, divisor=1, currency_suffix=''):
    spec = METRIC_OPTIONS[label]
    if not is_valid_metric_value(value, spec):
        return 'N/A'
    numeric = float(value)
    if spec.unit == 'currency':
        return f'{numeric / divisor:,.2f} {currency_suffix}'.strip()
    if spec.unit in ('currency/share', 'currency/year'):
        return f'{spec.format.format(numeric)} {currency_suffix.split()[-1] if currency_suffix else ""}'.strip()
    return spec.format.format(numeric)


def _ordinal(number):
    if 10 <= number % 100 <= 20:
        return f'{number}th'
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return f'{number}{suffix}'


def direction_aware_percentile(peers, value, direction, exclude=None):
    valid = pd.to_numeric(pd.Series(peers), errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if exclude is not None:
        valid = valid.drop(exclude, errors='ignore')
    if value is None or not np.isfinite(value) or valid.empty:
        return {'percent': None, 'peer_count': 0, 'tie_count': 0, 'text': 'No valid peer comparison'}
    tie_count = int(np.isclose(valid.to_numpy(dtype=float), float(value)).sum())
    if direction == 'higher_better':
        percent = float((valid < value).mean() * 100)
        text = f'Better than {percent:.0f}% of {len(valid)} valid peers'
    elif direction == 'lower_better':
        percent = float((valid > value).mean() * 100)
        text = f'Better than {percent:.0f}% of {len(valid)} valid peers'
    else:
        percent = float((valid <= value).mean() * 100)
        text = f'{_ordinal(round(percent))} percentile of {len(valid)} valid peers'
    if tie_count:
        text += f'; tied with {tie_count}'
    return {'percent': percent, 'peer_count': len(valid), 'tie_count': tie_count, 'text': text}


def comparison_states(values, spec):
    valid = {key: float(value) for key, value in values.items() if is_valid_metric_value(value, spec)}
    if spec.direction == 'neutral' or not valid:
        return {key: '' for key in values}
    target_best = max(valid.values()) if spec.direction == 'higher_better' else min(valid.values())
    target_worst = min(valid.values()) if spec.direction == 'higher_better' else max(valid.values())
    if np.isclose(target_best, target_worst):
        return {key: 'Tied' if key in valid else '' for key in values}
    states = {}
    for key, value in values.items():
        if key not in valid:
            states[key] = ''
        elif np.isclose(valid[key], target_best):
            states[key] = 'Best'
        elif np.isclose(valid[key], target_worst):
            states[key] = 'Weakest'
        else:
            states[key] = ''
    return states


def comparison_cell_style(value):
    text = str(value)
    if text.endswith(' - Best'):
        return 'background-color: rgba(22, 163, 74, 0.09); font-weight: 600;'
    if text.endswith(' - Weakest'):
        return 'background-color: rgba(220, 38, 38, 0.07); font-weight: 600;'
    if text.endswith(' - Tied'):
        return 'background-color: rgba(100, 116, 139, 0.06); font-weight: 600;'
    return ''


def normalize_view(value):
    value = str(value or '').lower()
    return value if value in _VIEW_LABELS else 'table'


def parse_stock_query(raw, valid_stocks, limit=MAX_STOCKS):
    valid_set = set(valid_stocks)
    stocks = []
    for item in str(raw or '').split(','):
        stock = item.strip().upper()
        if stock and stock in valid_set and stock not in stocks:
            stocks.append(stock)
    return stocks[:limit], len(stocks) > limit


def available_stock_groups(market, valid_stocks, limit=MAX_STOCKS):
    """Preset groups for a market, keeping only tickers in the universe and groups with at least two left."""
    valid_set = set(valid_stocks)
    groups = {}
    for name, stocks in STOCK_GROUPS.get(market, {}).items():
        kept = [stock for stock in stocks if stock in valid_set][:limit]
        if len(kept) >= 2:
            groups[name] = kept
    return groups


def split_valid_size(df, spec):
    """Return (rows with a valid size value, tickers dropped for an invalid one)."""
    values = pd.to_numeric(df[spec.source], errors='coerce') if spec.source in df.columns else pd.Series(np.nan, index=df.index)
    mask = values.apply(lambda value: is_valid_metric_value(value, spec))
    kept = df[mask].copy()
    kept[spec.source] = values[mask]
    return kept, df.loc[~mask, 'stock'].tolist()


def _clean_prices(price_df):
    """Daily closes as (date, close): parsed, positive, one row per date, oldest first."""
    if not isinstance(price_df, pd.DataFrame) or price_df.empty or not {'date', 'close'} <= set(price_df.columns):
        return pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'), 'close': pd.Series(dtype=float)})
    prices = pd.DataFrame({
        'date': pd.to_datetime(price_df['date'], errors='coerce'),
        'close': pd.to_numeric(price_df['close'], errors='coerce'),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices[prices['close'] > 0]
    prices['date'] = prices['date'].dt.normalize().astype('datetime64[ns]')
    prices = prices.sort_values('date', kind='stable').drop_duplicates('date', keep='last')
    return prices.reset_index(drop=True)


def _clean_dividends(div_df):
    """Dividends as (date, adjDividend): parsed, non-negative, summed per ex-date, oldest first."""
    if not isinstance(div_df, pd.DataFrame) or div_df.empty or not {'date', 'adjDividend'} <= set(div_df.columns):
        return pd.DataFrame({'date': pd.Series(dtype='datetime64[ns]'), 'adjDividend': pd.Series(dtype=float)})
    divs = pd.DataFrame({
        'date': pd.to_datetime(div_df['date'], errors='coerce'),
        'adjDividend': pd.to_numeric(div_df['adjDividend'], errors='coerce'),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    divs = divs[divs['adjDividend'] >= 0]
    divs['date'] = divs['date'].dt.normalize().astype('datetime64[ns]')
    # Exact repeats are feed duplicates; distinct amounts on one date are separate payments.
    divs = divs.drop_duplicates()
    divs = divs.groupby('date', as_index=False)['adjDividend'].sum()
    return divs.sort_values('date').reset_index(drop=True)


def build_performance_frame(price_df, div_df, include_dividends=True):
    """Daily price and total-return series for one stock.

    Uses the unadjusted ``close`` plus ``adjDividend`` (the convention of
    harvest.simulator and the pipeline's total_return_*); ``adjClose`` already
    folds dividends in, so adding dividends to it would count them twice.
    Each dividend is reinvested at the close on its ex-date, or on the next
    trading day when the ex-date has no price. Returns columns ``date``,
    ``price`` (close), ``total`` (close x accumulated shares, starting from one
    share) and ``value`` (``total`` or ``price`` per ``include_dividends``).
    """
    prices = _clean_prices(price_df)
    if prices.empty:
        return pd.DataFrame({
            'date': pd.Series(dtype='datetime64[ns]'),
            'price': pd.Series(dtype=float), 'total': pd.Series(dtype=float), 'value': pd.Series(dtype=float),
        })
    closes = prices['close'].to_numpy(dtype=float)
    growth = np.ones(len(prices))
    divs = _clean_dividends(div_df)
    divs = divs[divs['date'] >= prices['date'].iloc[0]]
    if not divs.empty:
        positions = prices['date'].searchsorted(divs['date'], side='left')
        for position, amount in zip(positions, divs['adjDividend'].to_numpy(dtype=float)):
            if position < len(prices):
                growth[position] *= 1 + amount / closes[position]
    total = closes * np.cumprod(growth)
    return pd.DataFrame({
        'date': prices['date'].to_numpy(),
        'price': closes,
        'total': total,
        'value': total if include_dividends else closes,
    })


def rebase_performance(frames_by_stock, period_start=None):
    """Rebase every stock's series to 100 on one common start date.

    The common start is the latest of ``period_start`` and each stock's first
    price date. Each stock's base is its last observation on or before that
    date, so every line starts at exactly 100 on the same day. Returns
    ``(long_df, common_start, limiting_stock)``; ``limiting_stock`` names the
    stock that pushed the start past ``period_start``, else None.
    """
    columns = ['stock', 'date', 'price', 'total', 'value']
    usable = {stock: frame for stock, frame in frames_by_stock.items() if frame is not None and not frame.empty}
    if not usable:
        return pd.DataFrame(columns=columns), None, None
    firsts = {stock: pd.Timestamp(frame['date'].iloc[0]) for stock, frame in usable.items()}
    limiting_stock = max(firsts, key=firsts.get)
    common_start = firsts[limiting_stock]
    if period_start is None:
        limiting_stock = None
    elif pd.Timestamp(period_start) >= common_start:
        common_start, limiting_stock = pd.Timestamp(period_start).normalize(), None

    pieces = []
    for stock, frame in usable.items():
        position = int(frame['date'].searchsorted(common_start, side='right')) - 1
        base = frame.iloc[position]
        after = frame.iloc[position + 1:]
        piece = {'stock': stock, 'date': [common_start, *after['date']]}
        for col in ('price', 'total', 'value'):
            piece[col] = [100.0, *(after[col].to_numpy(dtype=float) / float(base[col]) * 100)]
        pieces.append(pd.DataFrame(piece))
    long_df = pd.concat(pieces, ignore_index=True)
    long_df['date'] = pd.to_datetime(long_df['date'])
    return long_df[columns], common_start, limiting_stock


def performance_summary(long_df, stocks=None):
    """Price return, total return, annualised total return and max drawdown per stock, as fractions (NaN = N/A)."""
    columns = ['Price return', 'Total return', 'Annualised total return', 'Max drawdown']
    if stocks is None:
        stocks = list(dict.fromkeys(long_df['stock'])) if not long_df.empty else []
    rows = []
    for stock in stocks:
        group = long_df[long_df['stock'] == stock].sort_values('date') if not long_df.empty else long_df
        row = dict.fromkeys(columns, np.nan)
        if len(group) >= 2:
            price, total = group['price'].to_numpy(dtype=float), group['total'].to_numpy(dtype=float)
            row['Price return'] = price[-1] / price[0] - 1
            row['Total return'] = total[-1] / total[0] - 1
            days = (pd.Timestamp(group['date'].iloc[-1]) - pd.Timestamp(group['date'].iloc[0])).days
            if days >= _MIN_ANNUALISE_DAYS and total[-1] > 0:
                row['Annualised total return'] = (total[-1] / total[0]) ** (365.25 / days) - 1
            row['Max drawdown'] = float(np.min(total / np.maximum.accumulate(total) - 1))
        rows.append({'Stock': stock, **row})
    return pd.DataFrame(rows, columns=['Stock', *columns]).set_index('Stock')


def format_signed_pct(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 'N/A'
    if not np.isfinite(numeric):
        return 'N/A'
    return f'{numeric * 100:+.1f}%'


def thin_series(long_df, max_points=_MAX_CHART_POINTS):
    """Keep at most about ``max_points`` evenly spaced rows per stock, always keeping the first and last."""
    if long_df.empty:
        return long_df
    pieces = []
    for _, group in long_df.groupby('stock', sort=False):
        if len(group) > max_points:
            keep = np.unique(np.linspace(0, len(group) - 1, max_points).round().astype(int))
            group = group.iloc[keep]
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def yearly_dividends(div_df, price_df, years, measure='Per share', today=None):
    """Yearly dividend figures for one stock, grouped by ex-date calendar year.

    ``measure`` is one of ``_DIV_MEASURES``: ``Per share`` sums ``adjDividend``;
    ``Growth index`` rebases to 100 in the first shown year with a positive
    dividend; ``Yield at year-end price`` divides by that year's last close
    (the latest close for the current year) x 100. Years from the stock's
    first dividend year onward are zero-filled; earlier years are absent, as
    are years without a price for the yield measure. Ex-dates after ``today``
    are ignored. Returns columns ``year``, ``value`` and ``is_ytd``.
    """
    if measure not in _DIV_MEASURES:
        raise ValueError(f'Unknown dividend measure: {measure}')
    empty = pd.DataFrame({'year': pd.Series(dtype=int), 'value': pd.Series(dtype=float), 'is_ytd': pd.Series(dtype=bool)})
    today = (pd.Timestamp.today() if today is None else pd.Timestamp(today)).normalize()
    years = sorted({int(year) for year in years})
    divs = _clean_dividends(div_df)
    divs = divs[divs['date'] <= today]
    if divs.empty or not years:
        return empty
    per_year = divs.groupby(divs['date'].dt.year)['adjDividend'].sum()
    first_year = int(per_year.index.min())
    shown = [year for year in years if year >= first_year]
    if not shown:
        return empty
    amounts = pd.Series([float(per_year.get(year, 0.0)) for year in shown], index=shown, dtype=float)

    if measure == 'Per share':
        values = amounts
    elif measure == 'Growth index':
        positive = amounts[amounts > 0]
        if positive.empty:
            return empty
        values = amounts[amounts.index >= positive.index[0]] / positive.iloc[0] * 100
    else:
        prices = _clean_prices(price_df)
        prices = prices[prices['date'] <= today]
        if prices.empty:
            return empty
        year_end = prices.groupby(prices['date'].dt.year)['close'].last()
        values = (amounts / year_end.reindex(amounts.index) * 100).dropna()

    if values.empty:
        return empty
    return pd.DataFrame({
        'year': values.index.astype(int),
        'value': values.to_numpy(dtype=float),
        'is_ytd': values.index == today.year,
    })


def _sync_query(key, value, remove_empty=False):
    current = st.query_params.get(key)
    if remove_empty and not value:
        if key in st.query_params:
            del st.query_params[key]
    elif current != value:
        st.query_params[key] = value


def _metric_hint(label):
    spec = METRIC_OPTIONS[label]
    direction = {'higher_better': 'Higher is generally more favorable.', 'lower_better': 'Lower positive values are generally more favorable.', 'neutral': 'Contextual; no automatic best or weakest.'}[spec.direction]
    return f'{spec.description} {direction}'


market_key = str(st.query_params.get('market', 'jkse')).lower()
market_key = market_key if market_key in _MARKET_QP_MAP else 'jkse'
# Also reseed when the widget state is gone (Streamlit drops it after leaving the page):
# a segmented control has no first-option fallback like a radio, so it would come back as None.
if (
    st.session_state.get('_comp_last_qp_market') != market_key
    or st.session_state.get('comp_sl') not in _MARKET_QP_REV
):
    st.session_state['comp_sl'] = _MARKET_QP_MAP[market_key]
    st.session_state['_comp_last_qp_market'] = market_key


def _market_changed():
    selected = st.session_state.get('comp_sl', _MARKET_QP_MAP['jkse'])
    query_value = _MARKET_QP_REV.get(selected, 'jkse')
    _sync_query('market', query_value)
    st.session_state['_comp_last_qp_market'] = query_value


market_col, syariah_col = st.columns([3, 1], vertical_alignment='bottom')
stock_select = market_col.segmented_control(
    'Stock market',
    list(_MARKET_QP_MAP.values()),
    key='comp_sl',
    on_change=_market_changed,
    required=True,
)
market = 'JKSE' if stock_select == _MARKET_QP_MAP['jkse'] else 'S&P500'
_sync_query('market', _MARKET_QP_REV[stock_select])
st.session_state['_comp_last_qp_market'] = _MARKET_QP_REV[stock_select]

previous_market = st.session_state.get('_comp_logged_market')
if previous_market != market:
    logger.info('COMPARISON | event=market_select | market=%s', market)
    st.session_state['_comp_logged_market'] = market

if market == 'JKSE':
    data_key, divisor, currency_suffix = 'div_score_jkse', 1_000_000_000_000, 'T IDR'
else:
    data_key, divisor, currency_suffix = 'div_score_sp500', 1_000_000_000, 'B USD'

try:
    final_df = get_div_score_table(data_key)
except Exception as exc:
    logger.exception('COMPARISON | event=universe_load_error | market=%s', market)
    st.error('The stock universe could not be loaded. Your URL selection is unchanged; retry in a moment.')
    st.caption(f'Data service response: {exc}')
    st.stop()

if market != 'JKSE':
    final_df = final_df.drop('GOOGL', errors='ignore')
if market == 'JKSE' and 'is_syariah' in final_df.columns:
    if syariah_col.toggle('Syariah only', key='comp_syariah'):
        final_df = final_df[final_df['is_syariah'].eq(True)]

filtered_df = get_processed_df(final_df[[col for col in _KEEP_COLS if col in final_df.columns]])
stock_options = sorted(filtered_df.index.tolist())
query_stocks, query_was_limited = parse_stock_query(st.query_params.get('stocks', ''), stock_options)
if query_was_limited:
    st.warning('This shared comparison contained more than five valid stocks. The first five were kept in URL order for readability.')
    _sync_query('stocks', ','.join(query_stocks))

canonical_query_stocks = ','.join(query_stocks)
if st.session_state.get('_comp_last_qp_stocks') != canonical_query_stocks:
    st.session_state['comp_stocks'] = query_stocks
    st.session_state['_comp_last_qp_stocks'] = canonical_query_stocks


def _stocks_changed():
    selected = [stock for stock in st.session_state.get('comp_stocks', []) if stock in stock_options][:MAX_STOCKS]
    query_value = ','.join(selected)
    _sync_query('stocks', query_value, remove_empty=True)
    st.session_state['_comp_last_qp_stocks'] = query_value


def _clear_selection():
    st.session_state['comp_stocks'] = []
    st.session_state['_comp_last_qp_stocks'] = ''
    if 'stocks' in st.query_params:
        del st.query_params['stocks']


select_col, clear_col = st.columns([5, 1])
clear_col.button('Clear selection', on_click=_clear_selection, use_container_width=True)
selected_stocks = select_col.multiselect(
    'Stocks to compare',
    options=stock_options,
    max_selections=MAX_STOCKS,
    placeholder='Search ticker symbols',
    key='comp_stocks',
    on_change=_stocks_changed,
    help='Choose at least two and no more than five stocks. Selection order sets a stable chart color.',
)
_sync_query('stocks', ','.join(selected_stocks), remove_empty=True)
st.session_state['_comp_last_qp_stocks'] = ','.join(selected_stocks)

stock_groups = available_stock_groups(market, stock_options)


def _group_picked():
    group = st.session_state.get('comp_group')
    if group not in stock_groups:
        return
    stocks = stock_groups[group]
    query_value = ','.join(stocks)
    st.session_state['comp_stocks'] = list(stocks)
    st.session_state['_comp_last_qp_stocks'] = query_value
    _sync_query('stocks', query_value, remove_empty=True)
    logger.info('COMPARISON | event=group_select | market=%s | group=%s', market, group)


if stock_groups:
    # Derive the highlight from the current selection so it clears after manual edits and matches shared URLs.
    st.session_state['comp_group'] = next(
        (name for name, stocks in stock_groups.items() if set(stocks) == set(selected_stocks)), None
    )
    st.pills(
        'Popular comparisons',
        options=list(stock_groups),
        selection_mode='single',
        key='comp_group',
        on_change=_group_picked,
        help='Fill the selection with a group of stocks that are commonly compared with each other.',
    )

if len(selected_stocks) < 2:
    st.info('Select at least two stocks to start. A focused comparison supports up to five.')
    st.stop()

selection_tuple = tuple(selected_stocks)
if st.session_state.get('_comp_logged_stocks') != selection_tuple:
    logger.info('COMPARISON | event=stock_select | market=%s | stocks=%s | count=%d', market, ','.join(selected_stocks), len(selected_stocks))
    st.session_state['_comp_logged_stocks'] = selection_tuple

stock_colors = {stock: _STOCK_PALETTE[index] for index, stock in enumerate(selected_stocks)}
comp_df = filtered_df.loc[selected_stocks].copy()

view_key = normalize_view(st.query_params.get('tab', 'table'))
if st.query_params.get('tab') != view_key:
    _sync_query('tab', view_key)
if st.session_state.get('_comp_last_qp_view') != view_key:
    st.session_state['comp_view'] = _VIEW_LABELS[view_key]
    st.session_state['_comp_last_qp_view'] = view_key


def _view_changed():
    selected = st.session_state.get('comp_view', _VIEW_LABELS['table'])
    query_value = _VIEW_KEYS.get(selected, 'table')
    _sync_query('tab', query_value)
    st.session_state['_comp_last_qp_view'] = query_value


view_label = st.segmented_control(
    'Comparison view',
    list(_VIEW_LABELS.values()),
    required=True,
    key='comp_view',
    on_change=_view_changed,
)
active_view = _VIEW_KEYS.get(view_label, 'table')
_sync_query('tab', active_view)
st.session_state['_comp_last_qp_view'] = active_view
if st.session_state.get('_comp_logged_tab') != active_view:
    logger.info('COMPARISON | event=view_change | view=%s | stocks=%s', active_view, ','.join(selected_stocks))
    st.session_state['_comp_logged_tab'] = active_view


if active_view == 'table':
    control_a, control_b = st.columns(2)
    preset = control_a.selectbox('Metric preset', [*METRIC_PRESETS, 'Custom'], key='comp_metric_preset')
    if preset == 'Custom':
        selected_metrics = control_b.multiselect(
            'Custom metrics',
            list(METRIC_OPTIONS),
            default=TABLE_DEFAULT_METRICS,
            format_func=lambda label: f'{METRIC_OPTIONS[label].group} - {label}',
            key='comp_table_metrics',
        )
    else:
        selected_metrics = METRIC_PRESETS[preset]
        control_b.caption(f'{len(selected_metrics)} recommended metrics. Choose Custom to build a different set.')

    if not selected_metrics:
        st.info('Choose at least one metric to compare.')
    else:
        rows = []
        for label in selected_metrics:
            spec = METRIC_OPTIONS[label]
            if spec.source not in comp_df.columns:
                continue
            values = {stock: comp_df.loc[stock, spec.source] for stock in selected_stocks}
            states = comparison_states(values, spec)
            row = {'Metric': label}
            for stock, value in values.items():
                formatted = format_metric_value(label, value, divisor, currency_suffix)
                row[stock] = f'{formatted} - {states[stock]}' if states[stock] else formatted
            rows.append(row)
        if not rows:
            st.warning('None of the selected metrics are available in this dataset.')
        else:
            table_df = pd.DataFrame(rows).set_index('Metric')
            st.dataframe(table_df.style.map(comparison_cell_style), width='stretch')
            st.caption('Best and Weakest use only valid values. Tied marks equal valid values. N/A means missing or financially invalid data.')
            with st.expander('Metric definitions and comparison rules'):
                for label in selected_metrics:
                    st.markdown(f'**{label}** ({METRIC_OPTIONS[label].group}): {_metric_hint(label)}')


elif active_view == 'dist':
    metric_labels = list(CHART_METRIC_OPTIONS)
    default_metric = st.query_params.get('dist_metric', 'Dividend Yield (%)')
    default_metric = default_metric if default_metric in metric_labels else 'Dividend Yield (%)'
    ctrl_a, ctrl_b = st.columns(2)
    selected_label = ctrl_a.selectbox('Metric', metric_labels, index=metric_labels.index(default_metric), key='comp_dist_metric')
    spec = METRIC_OPTIONS[selected_label]
    _sync_query('dist_metric', selected_label)

    sectors = ['All']
    if 'sector' in filtered_df.columns:
        sectors += sorted(filtered_df['sector'].dropna().astype(str).unique().tolist())
    requested_sector = st.query_params.get('dist_sector', 'All')
    requested_sector = requested_sector if requested_sector in sectors else 'All'
    selected_sector = ctrl_b.selectbox('Peer sector', sectors, index=sectors.index(requested_sector), key='comp_dist_sector')
    _sync_query('dist_sector', selected_sector)

    toggle_a, toggle_b = st.columns(2)
    show_universe = toggle_a.toggle('Show peer distribution', value=True, key='comp_dist_universe')
    exclude_zero_yield = toggle_b.toggle(
        'Exclude 0% yield', value=False, disabled=selected_label != 'Dividend Yield (%)', key='comp_dist_excl_zero'
    )

    universe_df = filtered_df
    if selected_sector != 'All':
        universe_df = universe_df[universe_df['sector'].astype(str).eq(selected_sector)]
    if exclude_zero_yield and selected_label == 'Dividend Yield (%)':
        before = len(universe_df)
        universe_df = universe_df[universe_df['yield'] > 0]
        st.caption(f'{before - len(universe_df)} zero-yield stocks excluded from this peer group.')

    valid_peers = valid_metric_series(universe_df, spec)
    comparison_vals = {
        stock: float(comp_df.loc[stock, spec.source])
        for stock in selected_stocks
        if spec.source in comp_df.columns and is_valid_metric_value(comp_df.loc[stock, spec.source], spec)
    }
    invalid_selected = [stock for stock in selected_stocks if stock not in comparison_vals]
    filtered_selected = [stock for stock in comparison_vals if stock not in universe_df.index]
    if invalid_selected:
        st.warning(f'No valid {selected_label} value for: {", ".join(invalid_selected)}.')
    if filtered_selected:
        st.info(f'Outside the selected peer filter: {", ".join(filtered_selected)}. Their values remain listed but are not part of the peer distribution.')

    if valid_peers.empty:
        st.warning('No valid peer values remain for this metric and filter.')
    else:
        kpi_cols = st.columns(4)
        for col, label, value in zip(
            kpi_cols,
            ('Peer mean', 'Peer median', '25th percentile', '75th percentile'),
            (valid_peers.mean(), valid_peers.median(), valid_peers.quantile(.25), valid_peers.quantile(.75)),
        ):
            col.metric(label, format_metric_value(selected_label, value, divisor, currency_suffix))

        q05, q95 = float(valid_peers.quantile(.05)), float(valid_peers.quantile(.95))
        outliers = [stock for stock, value in comparison_vals.items() if value < q05 or value > q95]
        st.subheader(f'{selected_label} distribution')
        st.caption(f'Peer chart shows the 5th-95th percentile range across {len(valid_peers)} valid values. {_metric_hint(selected_label)}')
        if outliers:
            st.info(f'Clipped to the chart edge as outliers: {", ".join(outliers)}. Tooltips retain actual values.')

        if show_universe:
            chart_df = universe_df.loc[valid_peers.index]
            chart = hp.plot_card_distribution(
                chart_df,
                spec.source,
                color='#64748B',
                height=420,
                show_axis=True,
                comparison_vals=comparison_vals,
                comparison_colors=stock_colors,
                x_range=(q05, q95),
                fill_opacity=.2,
                show_median=True,
            )
        else:
            bar_df = pd.DataFrame({'Stock': list(comparison_vals), 'Value': list(comparison_vals.values())})
            color_scale = alt.Scale(domain=selected_stocks, range=[stock_colors[stock] for stock in selected_stocks])
            chart = alt.Chart(bar_df).mark_bar().encode(
                x=alt.X('Stock:N', sort=None),
                y=alt.Y('Value:Q', title=selected_label),
                color=alt.Color('Stock:N', scale=color_scale, legend=None),
                tooltip=['Stock:N', alt.Tooltip('Value:Q', title=selected_label, format='.2f')],
            ).properties(height=380)
        st.altair_chart(chart, width='stretch', theme=None)

        summary = []
        for stock in selected_stocks:
            value = comparison_vals.get(stock)
            rank = direction_aware_percentile(valid_peers, value, spec.direction, exclude=stock) if value is not None else None
            summary.append({
                'Stock': stock,
                'Value': format_metric_value(selected_label, value, divisor, currency_suffix) if value is not None else 'N/A',
                'Peer context': rank['text'] if rank else 'No valid peer comparison',
            })
        st.dataframe(pd.DataFrame(summary).set_index('Stock'), width='stretch')


elif active_view == 'history':
    today_ts = pd.Timestamp.today().normalize()
    histories, failed = {}, []
    with st.spinner('Loading price and dividend history...'):
        for stock in selected_stocks:
            try:
                histories[stock] = get_stock_history(stock, market)
            except Exception:
                logger.exception('COMPARISON | event=history_load_error | market=%s | stock=%s', market, stock)
                failed.append(stock)
    no_prices = [stock for stock, (price_df, _) in histories.items() if _clean_prices(price_df).empty]
    problems = []
    if failed:
        problems.append(f'History could not be loaded for: {", ".join(failed)}.')
    if no_prices:
        problems.append(f'No price history is available for: {", ".join(no_prices)}.')
    if problems:
        st.warning(' '.join(problems) + ' The other stocks are still shown.')

    st.subheader('Performance')
    requested_period = str(st.query_params.get('hist_period', _HIST_DEFAULT_PERIOD)).upper()
    requested_period = requested_period if requested_period in _HIST_PERIODS else _HIST_DEFAULT_PERIOD
    ctrl_a, ctrl_b = st.columns([3, 2])
    period = ctrl_a.segmented_control(
        'Period', list(_HIST_PERIODS), default=requested_period, required=True, key='comp_hist_period'
    ) or requested_period
    _sync_query('hist_period', period)
    if st.session_state.get('_comp_logged_hist_period') != period:
        logger.info('COMPARISON | event=history_period | period=%s | stocks=%s', period, ','.join(selected_stocks))
        st.session_state['_comp_logged_hist_period'] = period
    include_dividends = ctrl_b.toggle(
        'Include dividends', value=True, key='comp_hist_include_div',
        help='On: total return, with each dividend reinvested at the close on its ex-date. Off: price change only.',
    )

    period_start = today_ts - pd.DateOffset(years=_HIST_PERIODS[period])
    frames = {
        stock: build_performance_frame(price_df, div_df, include_dividends)
        for stock, (price_df, div_df) in histories.items()
    }
    frames = {stock: frame for stock, frame in frames.items() if not frame.empty}
    perf_df, common_start, limiting_stock = rebase_performance(frames, period_start)
    plotted = [stock for stock in selected_stocks if stock in frames]
    color_scale = alt.Scale(domain=plotted, range=[stock_colors[stock] for stock in plotted])

    if perf_df.empty:
        st.info('No price history is available for the selected stocks.')
    else:
        if limiting_stock:
            st.caption(
                f'{limiting_stock} has price history only from {common_start:%d %b %Y}, so every line starts '
                f'at 100 on that date rather than at the start of the {period} period.'
            )
        value_title = 'Total return index' if include_dividends else 'Price index'
        chart = alt.Chart(thin_series(perf_df)).mark_line(strokeWidth=2).encode(
            x=alt.X('date:T', title=None),
            y=alt.Y('value:Q', title=f'{value_title} (start = 100)', scale=alt.Scale(zero=False)),
            color=alt.Color('stock:N', title='Stock', scale=color_scale, sort=plotted),
            tooltip=[
                alt.Tooltip('stock:N', title='Stock'),
                alt.Tooltip('date:T', title='Date', format='%d %b %Y'),
                alt.Tooltip('value:Q', title=value_title, format=',.1f'),
            ],
        ).properties(height=420)
        st.altair_chart(chart, width='stretch', theme=None)

        summary = performance_summary(perf_df, plotted)
        st.dataframe(summary.map(format_signed_pct), width='stretch')
        st.caption(
            f'From {common_start:%d %b %Y} to each stock\'s latest close. Total return reinvests every dividend '
            'at the close on its ex-date; taxes and fees are ignored. Max drawdown is the largest peak-to-trough '
            'fall in total return. Annualised return is N/A for windows shorter than one year.'
        )

    st.subheader('Dividends by year')
    measure = st.segmented_control(
        'Measure', list(_DIV_MEASURES), default=_DIV_MEASURES[0], required=True, key='comp_hist_div_measure'
    ) or _DIV_MEASURES[0]
    years = list(range(today_ts.year - _HIST_YEARS + 1, today_ts.year + 1))
    pieces = []
    for stock, (price_df, div_df) in histories.items():
        yearly = yearly_dividends(div_df, price_df, years, measure, today=today_ts)
        if not yearly.empty:
            pieces.append(yearly.assign(stock=stock))
    div_chart_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    if div_chart_df.empty or not (div_chart_df['value'] > 0).any():
        st.info(f'None of the selected stocks paid a dividend between {years[0]} and {years[-1]}.')
    else:
        currency = currency_suffix.split()[-1]
        if measure == 'Per share':
            div_chart_df['display'] = div_chart_df['value'].map(lambda value: f'{value:,.2f} {currency}')
            y_title = f'Dividend per share ({currency})'
        elif measure == 'Growth index':
            div_chart_df['display'] = div_chart_df['value'].map(lambda value: f'{value:,.1f}')
            y_title = 'Dividend growth index (first year = 100)'
        else:
            div_chart_df['display'] = div_chart_df['value'].map(lambda value: f'{value:.2f}%')
            y_title = 'Dividend yield at year-end price (%)'
        div_stocks = [stock for stock in selected_stocks if stock in set(div_chart_df['stock'])]
        div_color_scale = alt.Scale(domain=div_stocks, range=[stock_colors[stock] for stock in div_stocks])
        chart = alt.Chart(div_chart_df).mark_bar().encode(
            x=alt.X('year:O', title=None, scale=alt.Scale(domain=years), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset('stock:N', sort=div_stocks, scale=alt.Scale(domain=div_stocks)),
            y=alt.Y('value:Q', title=y_title),
            color=alt.Color('stock:N', title='Stock', scale=div_color_scale, sort=div_stocks),
            opacity=alt.condition('datum.is_ytd', alt.value(.45), alt.value(1)),
            tooltip=[
                alt.Tooltip('stock:N', title='Stock'),
                alt.Tooltip('year:O', title='Year'),
                alt.Tooltip('display:N', title=measure),
            ],
        ).properties(height=380)
        st.altair_chart(chart, width='stretch', theme=None)
        caption = (
            f'Dividends are grouped by ex-date calendar year and include all dividend types, so the latest figure '
            f'can differ from the table\'s Last Dividend (latest finalized fiscal year, ordinary dividends only). '
            f'{today_ts.year} is year to date and drawn lighter.'
        )
        if measure == 'Growth index':
            caption += ' Each stock is rebased to 100 in its first year with a dividend in this window.'
        elif measure == 'Yield at year-end price':
            caption += f' Yield uses each year\'s last close; {today_ts.year} uses the latest close.'
        st.caption(caption)


else:
    metric_labels = list(CHART_METRIC_OPTIONS)
    default_x = st.query_params.get('sc_x', 'Dividend Yield (%)')
    default_y = st.query_params.get('sc_y', 'PE Ratio')
    default_x = default_x if default_x in metric_labels else 'Dividend Yield (%)'
    default_y = default_y if default_y in metric_labels else 'PE Ratio'
    size_none = 'None (uniform size)'
    size_logo = 'Company logo'
    size_options = [size_none, size_logo, *metric_labels]
    default_size = st.query_params.get('sc_size', 'Market Cap')
    default_size = default_size if default_size in size_options else 'Market Cap'

    ctrl_a, ctrl_b = st.columns(2)
    x_label = ctrl_a.selectbox('X axis', metric_labels, index=metric_labels.index(default_x), key='comp_sc_x')
    y_label = ctrl_b.selectbox('Y axis', metric_labels, index=metric_labels.index(default_y), key='comp_sc_y')
    ctrl_c, ctrl_d = st.columns(2)
    size_label = ctrl_c.selectbox('Point size', size_options, index=size_options.index(default_size), key='comp_sc_size')
    show_universe = ctrl_d.toggle('Show peer universe', value=False, key='comp_sc_universe')
    _sync_query('sc_x', x_label)
    _sync_query('sc_y', y_label)
    _sync_query('sc_size', size_label)

    x_spec, y_spec = METRIC_OPTIONS[x_label], METRIC_OPTIONS[y_label]
    if x_spec.source == y_spec.source:
        st.warning('Choose different metrics for the X and Y axes.')
    else:
        x_valid = valid_metric_series(filtered_df, x_spec)
        y_valid = valid_metric_series(filtered_df, y_spec)
        valid_index = x_valid.index.intersection(y_valid.index)
        plot_df = filtered_df.loc[valid_index].copy()
        selected_plot = plot_df.loc[plot_df.index.intersection(selected_stocks)].reset_index()
        excluded = [stock for stock in selected_stocks if stock not in selected_plot['stock'].tolist()]
        if excluded:
            st.warning(f'Excluded because an axis value is missing or invalid: {", ".join(excluded)}.')
        if size_label not in (size_none, size_logo) and not selected_plot.empty:
            sized_plot, size_dropped = split_valid_size(selected_plot, METRIC_OPTIONS[size_label])
            if sized_plot.empty:
                st.warning(f'No selected stock has a valid {size_label} value; using uniform point size.')
                size_label = size_none
            else:
                if size_dropped:
                    st.warning(f'Excluded because the {size_label} value used for point size is missing or invalid: {", ".join(size_dropped)}.')
                selected_plot = sized_plot
        if selected_plot.empty:
            st.warning('No selected stocks have valid values for both axes.')
        else:
            x_hint = 'right' if x_spec.direction == 'higher_better' else ('left' if x_spec.direction == 'lower_better' else 'contextual')
            y_hint = 'up' if y_spec.direction == 'higher_better' else ('down' if y_spec.direction == 'lower_better' else 'contextual')
            st.caption(f'Preferred direction: X is {x_hint}; Y is {y_hint}. Contextual axes should not be read as automatic recommendations.')

            if show_universe and not plot_df.empty:
                x05, x95 = plot_df[x_spec.source].quantile([.05, .95])
                y05, y95 = plot_df[y_spec.source].quantile([.05, .95])
                peer_plot = plot_df[
                    plot_df[x_spec.source].between(x05, x95) & plot_df[y_spec.source].between(y05, y95)
                ].reset_index()
            else:
                peer_plot = pd.DataFrame()

            color_scale = alt.Scale(domain=selected_stocks, range=[stock_colors[stock] for stock in selected_stocks])
            layers = []
            if not peer_plot.empty:
                layers.append(alt.Chart(peer_plot).mark_circle(size=45, color='#94A3B8', opacity=.25).encode(
                    x=alt.X(f'{x_spec.source}:Q', title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f'{y_spec.source}:Q', title=y_label, scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip('stock:N', title='Stock')],
                ))
                x_median, y_median = peer_plot[x_spec.source].median(), peer_plot[y_spec.source].median()
                layers.extend([
                    alt.Chart(pd.DataFrame({x_spec.source: [x_median]})).mark_rule(color='#94A3B8', strokeDash=[5, 5]).encode(x=f'{x_spec.source}:Q'),
                    alt.Chart(pd.DataFrame({y_spec.source: [y_median]})).mark_rule(color='#94A3B8', strokeDash=[5, 5]).encode(y=f'{y_spec.source}:Q'),
                ])

            tooltip = [
                alt.Tooltip('stock:N', title='Stock'),
                alt.Tooltip(f'{x_spec.source}:Q', title=x_label, format='.2f'),
                alt.Tooltip(f'{y_spec.source}:Q', title=y_label, format='.2f'),
            ]
            size_encoding = alt.value(500)
            if size_label not in (size_none, size_logo):
                size_spec = METRIC_OPTIONS[size_label]
                # zero=False maps the smallest value (even a negative one) to the smallest point
                size_encoding = alt.Size(f'{size_spec.source}:Q', title=size_label, scale=alt.Scale(range=[180, 1400], zero=False))
                tooltip.append(alt.Tooltip(f'{size_spec.source}:Q', title=size_label, format='.2f'))

            if size_label == size_logo:
                logo_base = 'jkse' if market == 'JKSE' else 'sp500'
                selected_plot['_logo_url'] = selected_plot['stock'].map(
                    lambda stock: f'https://raw.githubusercontent.com/mitbal/daguerreo-data/refs/heads/main/{logo_base}/logos/{stock.split(".")[0]}.svg'
                )
                selected_layer = alt.Chart(selected_plot).mark_image(width=42, height=42).encode(
                    x=alt.X(f'{x_spec.source}:Q', title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f'{y_spec.source}:Q', title=y_label, scale=alt.Scale(zero=False)),
                    url='_logo_url:N', tooltip=tooltip,
                )
            else:
                selected_layer = alt.Chart(selected_plot).mark_circle(stroke='white', strokeWidth=1.5).encode(
                    x=alt.X(f'{x_spec.source}:Q', title=x_label, scale=alt.Scale(zero=False)),
                    y=alt.Y(f'{y_spec.source}:Q', title=y_label, scale=alt.Scale(zero=False)),
                    size=size_encoding,
                    color=alt.Color('stock:N', title='Stock', scale=color_scale),
                    tooltip=tooltip,
                )
            labels = alt.Chart(selected_plot).mark_text(dy=-24, fontWeight='bold').encode(
                x=f'{x_spec.source}:Q', y=f'{y_spec.source}:Q', text='stock:N',
                color=alt.Color('stock:N', scale=color_scale, legend=None),
            )
            layers.extend([selected_layer, labels])

            if x_spec.direction != 'neutral' and y_spec.direction != 'neutral':
                quadrant_source = peer_plot if not peer_plot.empty else selected_plot
                x_low, x_high = quadrant_source[x_spec.source].quantile([.1, .9])
                y_low, y_high = quadrant_source[y_spec.source].quantile([.1, .9])
                x_mid = quadrant_source[x_spec.source].median()
                y_mid = quadrant_source[y_spec.source].median()
                if peer_plot.empty:
                    layers.extend([
                        alt.Chart(pd.DataFrame({x_spec.source: [x_mid]})).mark_rule(color='#94A3B8', strokeDash=[5, 5]).encode(x=f'{x_spec.source}:Q'),
                        alt.Chart(pd.DataFrame({y_spec.source: [y_mid]})).mark_rule(color='#94A3B8', strokeDash=[5, 5]).encode(y=f'{y_spec.source}:Q'),
                    ])
                quadrant_rows = []
                for x_value, x_is_high in ((x_low, False), (x_high, True)):
                    for y_value, y_is_high in ((y_low, False), (y_high, True)):
                        x_favorable = x_is_high == (x_spec.direction == 'higher_better')
                        y_favorable = y_is_high == (y_spec.direction == 'higher_better')
                        if x_favorable and y_favorable:
                            quadrant = 'More favorable on both'
                        elif x_favorable:
                            quadrant = 'X more favorable'
                        elif y_favorable:
                            quadrant = 'Y more favorable'
                        else:
                            quadrant = 'Less favorable on both'
                        quadrant_rows.append({x_spec.source: x_value, y_spec.source: y_value, 'quadrant': quadrant})
                layers.append(alt.Chart(pd.DataFrame(quadrant_rows)).mark_text(
                    color='#64748B', fontSize=11, opacity=.8
                ).encode(x=f'{x_spec.source}:Q', y=f'{y_spec.source}:Q', text='quadrant:N'))
            st.altair_chart(alt.layer(*layers).properties(height=520).interactive(), width='stretch', theme=None)

            summary = []
            for stock in selected_stocks:
                if stock not in comp_df.index:
                    continue
                summary.append({
                    'Stock': stock,
                    x_label: format_metric_value(x_label, comp_df.loc[stock, x_spec.source], divisor, currency_suffix),
                    y_label: format_metric_value(y_label, comp_df.loc[stock, y_spec.source], divisor, currency_suffix),
                })
            st.dataframe(pd.DataFrame(summary).set_index('Stock'), width='stretch')
