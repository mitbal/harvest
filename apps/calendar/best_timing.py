import os
import io
import json
import hashlib
import logging
import calendar
import concurrent.futures
from datetime import datetime

import numpy as np
import redis
import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd
from harvest.utils import setup_logging


current_year = datetime.today().year

st.set_page_config(page_title='Best Timing - Panen Dividen')

st.title('Best Timing')
sl = st.radio('Stock List', ['JKSE', 'S&P500'], index=0, horizontal=True)

if sl is None:
    st.stop()

exch = 'jkse' if sl == 'JKSE' else 'sp500'

div_years_key = f'div_cal_years_{exch}'
div_score_key = f'div_score_{exch}'


def get_logger(name, level=logging.INFO):
    return setup_logging(name, level)


def _safe_error_reason(exc):
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    return f'HTTP {status}' if status is not None else type(exc).__name__


logger = get_logger('best_timing')

url = os.getenv('REDIS_URL')
if not url:
    logger.error('REDIS_URL is not configured')
    st.error('Dividend data is unavailable because Redis is not configured.')
    st.stop()


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url, socket_connect_timeout=10, socket_timeout=30, socket_keepalive=True, retry_on_timeout=True)
    return r

r = connect_redis(url)


@st.cache_data(max_entries=8, ttl=60, show_spinner='Downloading dividend data')
def get_data_from_redis(key):
    try:
        value = r.get(key)
        if value is None:
            logger.warning(f'Missing redis key: {key}')
            return None

        if isinstance(value, bytes) and value.startswith(b'PAR1'):
            return pd.read_parquet(io.BytesIO(value))

        decoded = json.loads(value)
        if isinstance(decoded, dict) and 'date' in decoded and 'content' in decoded:
            decoded = json.loads(decoded['content'])
        if not isinstance(decoded, (dict, list)):
            raise ValueError(f'Expected an object or array, received {type(decoded).__name__}')
        return pd.DataFrame(decoded)
    except Exception as exc:
        logger.error(f'Failed to load Redis key {key}: {exc}')
        st.error(f'Dividend data for "{key}" is temporarily unavailable.')
        return None


# ── Year selection ───────────────────────────────────────────────────────────
years_df = get_data_from_redis(div_years_key)
if years_df is not None and 'year' in years_df.columns:
    available_years = sorted(
        pd.to_numeric(years_df['year'], errors='coerce').dropna().astype(int).unique().tolist()
    )
else:
    available_years = [current_year]

if len(available_years) == 0:
    available_years = [current_year]
selected_year = max(available_years)

# ── Load calendar & sector data ──────────────────────────────────────────────
div_cal_key = f'div_cal_{exch}_{selected_year}'
df = get_data_from_redis(div_cal_key)
if df is None or not {'symbol', 'date'}.issubset(df.columns):
    df = pd.DataFrame(columns=['symbol', 'date'])
else:
    df = df[['symbol', 'date']].copy()
    df['symbol'] = df['symbol'].astype('string').str.strip()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['symbol', 'date'])
    df = df[df['symbol'] != '']

_div_score_df = get_data_from_redis(div_score_key)

_sector_map = {}
if _div_score_df is not None and 'sector' in _div_score_df.columns and 'symbol' in _div_score_df.columns:
    _sector_symbols = _div_score_df['symbol'].astype('string').str.strip()
    _sector_names = _div_score_df['sector'].astype('string').fillna('Unknown').str.strip().replace('', 'Unknown')
    _sector_map = dict(zip(_sector_symbols, _sector_names))

_all_symbols = sorted(df['symbol'].unique().tolist()) if not df.empty else []
_using_downloaded_stock_list = False

if not _all_symbols:
    with st.spinner('Downloading stock list...'):
        try:
            if exch == 'jkse':
                _stock_df = hd.get_all_idx_stocks()
            else:
                _stock_df = hd.get_all_sp500_stocks()
            _all_symbols = sorted(
                _stock_df['symbol'].astype('string').dropna().str.strip().replace('', pd.NA).dropna().unique().tolist()
            )
            _using_downloaded_stock_list = True
            st.info(
                'Using downloaded stock list. For richer data (sector mapping, historical calendar), '
                'run the pipeline first.'
            )
        except Exception as e:
            reason = _safe_error_reason(e)
            logger.error(f'Failed to download stock list: {reason}')
            st.error(f'Cannot load stock list: {reason}')
            st.stop()

if _using_downloaded_stock_list:
    st.write(
        f'Pools historical price data across **{len(_all_symbols)}** listed stocks to reveal '
        'which months are consistently cheaper and how prices behave around available ex-dates.'
    )
else:
    st.write(
        f'Pools historical price data across all **{len(_all_symbols)}** dividend-paying stocks '
        f'in the **{selected_year}** calendar to reveal which months are consistently cheaper '
        f'and how many days before / after ex-date prices dip and recover.'
    )

# =========================================================================== #
# Aggregate Seasonality — Best Month to Buy Across All Dividend Stocks         #
# =========================================================================== #

MONTH_ORDER = list(calendar.month_abbr[1:])


def _density_frame(values, domain_min, domain_max, points=256):
    values = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return pd.DataFrame(columns=['x', 'density'])

    grid = np.linspace(domain_min, domain_max, points)
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    bandwidth = max(1.06 * std * values.size ** -0.2, 1.0)
    density = np.zeros_like(grid)
    for chunk in np.array_split(values, max(1, int(np.ceil(values.size / 2000)))):
        scaled = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * scaled ** 2).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2 * np.pi)
    return pd.DataFrame({'x': grid, 'density': density})


def _kaplan_meier_quantile(recovery_df, probability):
    if recovery_df is None or recovery_df.empty:
        return None
    work = recovery_df[['days_after', 'recovered']].copy()
    work['days_after'] = pd.to_numeric(work['days_after'], errors='coerce')
    work = work.dropna(subset=['days_after'])
    if work.empty:
        return None

    work['recovered'] = work['recovered'].fillna(False).astype(bool)
    at_risk = len(work)
    survival = 1.0
    for duration, group in work.groupby('days_after', sort=True):
        events = int(group['recovered'].sum())
        if events:
            survival *= 1 - events / at_risk
            if 1 - survival >= probability:
                return int(duration)
        at_risk -= len(group)
        if at_risk == 0:
            break
    return None


def _calc_low_entry_recovery_days(price_df, best_events, max_lookforward=365):
    """Measure post-ex recovery against each event's modeled pre-ex low entry."""
    if price_df is None or price_df.empty or not best_events:
        return []

    pdf = price_df[['date', 'close']].copy()
    pdf['date'] = pd.to_datetime(pdf['date'], errors='coerce')
    pdf['close'] = pd.to_numeric(pdf['close'], errors='coerce')
    pdf = pdf.replace([np.inf, -np.inf], np.nan).dropna(subset=['date', 'close'])
    pdf = pdf[pdf['close'] > 0].sort_values('date').reset_index(drop=True)
    if pdf.empty:
        return []

    recovery_days = []
    for event in best_events:
        ex_date = pd.to_datetime(event.get('ex_date'), errors='coerce')
        entry_date = pd.to_datetime(event.get('low_date'), errors='coerce')
        entry_price = pd.to_numeric(event.get('low_price'), errors='coerce')
        if pd.isna(ex_date) or pd.isna(entry_date) or pd.isna(entry_price) or entry_price <= 0:
            continue

        on_cum = pdf[pdf['date'] < ex_date]
        if on_cum.empty:
            continue
        cum_row = on_cum.iloc[-1]

        window_end = ex_date + pd.Timedelta(days=max_lookforward)
        forward = pdf[(pdf['date'] >= ex_date) & (pdf['date'] <= window_end)].copy()
        if forward.empty:
            continue
        forward['days_after'] = (forward['date'] - ex_date).dt.days
        recovered = forward[forward['close'] >= entry_price]

        detail = {
            'ex_date': ex_date.strftime('%Y-%m-%d'),
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'entry_price': float(entry_price),
            'cum_date': cum_row['date'].strftime('%Y-%m-%d'),
            'cum_price': float(cum_row['close']),
        }
        if recovered.empty:
            observation_end = min(pdf['date'].max(), window_end)
            detail.update({
                'recover_date': None,
                'recover_price': None,
                'days_after': max(0, int((observation_end - ex_date).days)),
                'recovered': False,
            })
        else:
            recover_row = recovered.iloc[0]
            detail.update({
                'recover_date': recover_row['date'].strftime('%Y-%m-%d'),
                'recover_price': float(recover_row['close']),
                'days_after': int(recover_row['days_after']),
                'recovered': True,
            })
        recovery_days.append(detail)

    return recovery_days


def _recovery_stats(recovery_df):
    if recovery_df is None or recovery_df.empty:
        return {
            'median': None, 'p25': None, 'p75': None, 'p90': None,
            'recovered': 0, 'censored': 0,
        }
    recovered = int(recovery_df['recovered'].sum())
    return {
        'median': _kaplan_meier_quantile(recovery_df, 0.50),
        'p25': _kaplan_meier_quantile(recovery_df, 0.25),
        'p75': _kaplan_meier_quantile(recovery_df, 0.75),
        'p90': _kaplan_meier_quantile(recovery_df, 0.90),
        'recovered': recovered,
        'censored': len(recovery_df) - recovered,
    }


def _source_data_version(div_cal_df, sector_map, symbols):
    calendar_rows = div_cal_df[div_cal_df['symbol'].isin(symbols)][['symbol', 'date']].copy()
    calendar_rows['date'] = pd.to_datetime(calendar_rows['date'], errors='coerce')
    calendar_rows = calendar_rows.dropna().sort_values(['symbol', 'date'])
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(calendar_rows, index=False).values.tobytes())
    digest.update(repr(sorted((symbol, sector_map.get(symbol, 'Unknown')) for symbol in symbols)).encode())
    return digest.hexdigest()


@st.cache_data(max_entries=128, ttl=60 * 60 * 6, show_spinner=False)
def _summarize_stock(symbol, event_start, event_end, data_version, sector, _calendar_dates):
    del data_version  # The source-data digest is intentionally part of the cache key.
    event_start_ts = pd.Timestamp(event_start)
    event_end_ts = pd.Timestamp(event_end)
    price_start = event_start_ts - pd.Timedelta(days=180)
    price_end = event_end_ts + pd.Timedelta(days=365)

    raw_pdf = hd.get_daily_stock_history_yahoo(
        symbol,
        start_from=price_start.strftime('%Y-%m-%d'),
        end_at=price_end.strftime('%Y-%m-%d'),
    )

    if raw_pdf is None or raw_pdf.empty or not {'date', 'close'}.issubset(raw_pdf.columns):
        raise ValueError('price history is empty or missing required columns')

    raw_pdf = raw_pdf.copy()
    raw_pdf['date'] = pd.to_datetime(raw_pdf['date'], errors='coerce')
    raw_pdf['close'] = pd.to_numeric(raw_pdf['close'], errors='coerce')
    raw_pdf = raw_pdf.replace([np.inf, -np.inf], np.nan).dropna(subset=['date', 'close'])
    raw_pdf = raw_pdf[(raw_pdf['date'] >= price_start) & (raw_pdf['date'] <= price_end)]
    raw_pdf = raw_pdf[raw_pdf['close'] > 0].sort_values('date')
    if raw_pdf.empty:
        raise ValueError('price history has no valid rows in the requested range')

    price_df = raw_pdf[['date', 'close']].copy()
    seasonality_df = price_df
    if 'adjClose' in raw_pdf.columns:
        adjusted = pd.to_numeric(raw_pdf['adjClose'], errors='coerce')
        valid_adjusted = adjusted.notna() & np.isfinite(adjusted) & (adjusted > 0)
        if valid_adjusted.any():
            seasonality_df = raw_pdf.loc[valid_adjusted, ['date']].copy()
            seasonality_df['close'] = adjusted.loc[valid_adjusted]

    dividend_source = 'dag' if symbol.endswith('.JK') else 'fmp'
    ddf = hd.get_dividend_history_single_stock(symbol, source=dividend_source)
    has_dividend_history = ddf is not None and not ddf.empty and 'date' in ddf.columns
    if has_dividend_history:
        ddf = ddf.copy()
        ddf['date'] = pd.to_datetime(ddf['date'], errors='coerce')
        ddf = ddf.dropna(subset=['date'])
        ddf = ddf[(ddf['date'] >= event_start_ts) & (ddf['date'] <= event_end_ts)]
        ddf = ddf[ddf['date'] <= price_df['date'].max()]
        if 'dividend' in ddf.columns:
            ddf['dividend_amount'] = pd.to_numeric(ddf['dividend'], errors='coerce')
        elif 'adjDividend' in ddf.columns:
            ddf['dividend_amount'] = pd.to_numeric(ddf['adjDividend'], errors='coerce')
        else:
            ddf['dividend_amount'] = np.nan
        div_dates = ddf[['date', 'dividend_amount']].reset_index(drop=True)
    else:
        div_dates = pd.DataFrame(columns=['date', 'dividend_amount'])

    monthly = hd.calc_aggregate_seasonality([{'symbol': symbol, 'price_df': seasonality_df}])
    monthly_rows = pd.DataFrame()
    if not monthly.empty:
        monthly_rows = monthly[['month', 'month_name', 'median']].rename(
            columns={'median': 'rel_price'}
        )
        monthly_rows.insert(0, 'sector', sector)
        monthly_rows.insert(0, 'symbol', symbol)
        monthly_rows.insert(0, 'record_type', 'monthly')

    if not div_dates.empty:
        event_dates = div_dates[['date']].copy()
    else:
        event_dates = pd.DataFrame({'date': pd.to_datetime(_calendar_dates, errors='coerce')})
    event_dates = event_dates.dropna(subset=['date'])
    event_dates = event_dates[event_dates['date'] <= price_df['date'].max()]
    price_min = price_df['date'].min()
    eligible_dates = event_dates[
        event_dates['date'] - pd.Timedelta(days=180) >= price_min
    ].copy()

    best_detail = hd.calc_pre_ex_best_days(
        price_df, eligible_dates, pre_ex_days=180, detail=True
    )
    low_detail = _calc_low_entry_recovery_days(price_df, best_detail, max_lookforward=365)
    matched_events = pd.DataFrame({'date': [event['ex_date'] for event in best_detail]})
    cum_detail = hd.calc_post_ex_recovery_days(
        price_df, matched_events, max_lookforward=365, detail=True
    )
    low_by_date = {event['ex_date']: event for event in low_detail}
    cum_by_date = {event['ex_date']: event for event in cum_detail}
    dividend_by_date = {}
    if not div_dates.empty:
        dividend_by_date = {
            pd.Timestamp(row.date).strftime('%Y-%m-%d'): row.dividend_amount
            for row in div_dates.itertuples()
        }

    event_columns = [
        'record_type', 'symbol', 'sector', 'ex_date', 'dividend_amount',
        'ex_price', 'low_date', 'low_price', 'days_before', 'entry_date',
        'entry_price', 'low_recover_date', 'low_recover_price',
        'low_days_after', 'low_recovered', 'cum_date', 'cum_price',
        'cum_recover_date', 'cum_recover_price', 'cum_days_after',
        'cum_recovered',
    ]
    event_rows = []
    for best in best_detail:
        ex_date = best['ex_date']
        low = low_by_date.get(ex_date, {})
        cum = cum_by_date.get(ex_date, {})
        event_rows.append({
            'record_type': 'event',
            'symbol': symbol,
            'sector': sector,
            'ex_date': ex_date,
            'dividend_amount': dividend_by_date.get(ex_date),
            'ex_price': best.get('ex_price'),
            'low_date': best.get('low_date'),
            'low_price': best.get('low_price'),
            'days_before': best.get('days_before'),
            'entry_date': low.get('entry_date'),
            'entry_price': low.get('entry_price'),
            'low_recover_date': low.get('recover_date'),
            'low_recover_price': low.get('recover_price'),
            'low_days_after': low.get('days_after'),
            'low_recovered': low.get('recovered'),
            'cum_date': cum.get('cum_date'),
            'cum_price': cum.get('cum_price'),
            'cum_recover_date': cum.get('recover_date'),
            'cum_recover_price': cum.get('recover_price'),
            'cum_days_after': cum.get('days_after'),
            'cum_recovered': cum.get('recovered'),
        })
    event_rows = pd.DataFrame(event_rows, columns=event_columns)
    stock_event_table = pd.concat([monthly_rows, event_rows], ignore_index=True, sort=False)
    return stock_event_table, has_dividend_history


def _fetch_all_summaries(symbols_tuple, event_start, event_end, data_version, sector_map, calendar_by_symbol):
    results = []
    failures = []
    missing_dividend_history = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(
                _summarize_stock,
                symbol,
                event_start,
                event_end,
                data_version,
                sector_map.get(symbol, 'Unknown'),
                tuple(calendar_by_symbol.get(symbol, ())),
            ): symbol
            for symbol in symbols_tuple
        }
        for fut in concurrent.futures.as_completed(futures):
            symbol = futures[fut]
            try:
                summary, has_dividend_history = fut.result()
                results.append(summary)
                missing_dividend_history += not has_dividend_history
            except Exception as exc:
                failures.append(symbol)
                reason = _safe_error_reason(exc)
                logger.warning(f'Failed to fetch timing data for {symbol}: {reason}')
    stock_event_table = pd.concat(results, ignore_index=True, sort=False) if results else pd.DataFrame()
    return stock_event_table, failures, missing_dividend_history


def _seasonality_for_sector(stock_event_table, sector='All'):
    sector = sector or 'All'
    rows = stock_event_table
    if sector != 'All' and not rows.empty:
        rows = rows[rows['sector'] == sector]
    if rows.empty:
        return tuple(pd.DataFrame() for _ in range(7))

    monthly = rows[rows['record_type'] == 'monthly'].copy()
    if monthly.empty:
        agg_df = pd.DataFrame()
    else:
        agg_df = monthly.groupby(['month', 'month_name'])['rel_price'].agg(
            mean='mean', median='median', std='std',
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        ).reset_index()

    events = rows[rows['record_type'] == 'event'].copy()
    best_days_df = events[['days_before']].dropna() if not events.empty else pd.DataFrame()
    low_recovery_df = events[['low_days_after', 'low_recovered']].rename(
        columns={'low_days_after': 'days_after', 'low_recovered': 'recovered'}
    ).dropna(subset=['days_after']) if not events.empty else pd.DataFrame()
    cum_recovery_df = events[['cum_days_after', 'cum_recovered']].rename(
        columns={'cum_days_after': 'days_after', 'cum_recovered': 'recovered'}
    ).dropna(subset=['days_after']) if not events.empty else pd.DataFrame()
    best_raw_df = events
    low_recovery_raw_df = events.rename(columns={
        'low_recover_date': 'recover_date', 'low_recover_price': 'recover_price',
        'low_days_after': 'days_after', 'low_recovered': 'recovered',
    })
    cum_recovery_raw_df = events.rename(columns={
        'cum_recover_date': 'recover_date', 'cum_recover_price': 'recover_price',
        'cum_days_after': 'days_after', 'cum_recovered': 'recovered',
    })
    return (
        agg_df, best_days_df, low_recovery_df, cum_recovery_df,
        best_raw_df, low_recovery_raw_df, cum_recovery_raw_df,
    )


def _season_sector_table(
    stock_event_table, sectors, all_symbols
):
    rows = []
    available_symbols = set(stock_event_table['symbol'].dropna()) if not stock_event_table.empty else set()
    for sec in sectors:
        sec_symbols = {
            s for s in all_symbols
            if s in available_symbols
            and not stock_event_table[
                (stock_event_table['symbol'] == s) & (stock_event_table['sector'] == sec)
            ].empty
        }
        n_stocks = len(sec_symbols)
        if n_stocks == 0:
            continue
        agg_df, best_df, low_rec_df, cum_rec_df, _, _, _ = _seasonality_for_sector(
            stock_event_table, sec
        )
        if agg_df.empty:
            rows.append({
                'sector': sec, 'n_stocks': n_stocks, 'best_month': '—',
                'avg_rel_price': None, 'median_days': None, 'p25_days': None, 'p75_days': None, 'n_events': 0,
                'median_recovery': None, 'p25_recovery': None, 'p75_recovery': None, 'p90_recovery': None,
                'n_recovery': 0, 'n_censored': 0,
                'cum_median_recovery': None, 'cum_n_recovery': 0, 'cum_n_censored': 0,
            })
            continue
        best_row = agg_df.loc[agg_df['median'].idxmin()]
        if not best_df.empty:
            median_days = int(best_df['days_before'].median())
            p25_days = int(best_df['days_before'].quantile(0.25))
            p75_days = int(best_df['days_before'].quantile(0.75))
            n_events = len(best_df)
        else:
            median_days = p25_days = p75_days = None
            n_events = 0
        low_stats = _recovery_stats(low_rec_df)
        cum_stats = _recovery_stats(cum_rec_df)
        rows.append({
            'sector': sec,
            'n_stocks': n_stocks,
            'best_month': best_row['month_name'],
            'avg_rel_price': best_row['median'],
            'median_days': median_days,
            'p25_days': p25_days,
            'p75_days': p75_days,
            'n_events': n_events,
            'median_recovery': low_stats['median'],
            'p25_recovery': low_stats['p25'],
            'p75_recovery': low_stats['p75'],
            'p90_recovery': low_stats['p90'],
            'n_recovery': low_stats['recovered'],
            'n_censored': low_stats['censored'],
            'cum_median_recovery': cum_stats['median'],
            'cum_n_recovery': cum_stats['recovered'],
            'cum_n_censored': cum_stats['censored'],
        })
    return pd.DataFrame(rows, columns=[
        'sector', 'n_stocks', 'best_month', 'avg_rel_price',
        'median_days', 'p25_days', 'p75_days', 'n_events',
        'median_recovery', 'p25_recovery', 'p75_recovery', 'p90_recovery', 'n_recovery', 'n_censored',
        'cum_median_recovery', 'cum_n_recovery', 'cum_n_censored',
    ])


_n_stocks = len(_all_symbols)

_sector_options = ['All']
if _sector_map and _all_symbols:
    _present_sectors = sorted({_sector_map.get(s, 'Unknown') for s in _all_symbols if s in _sector_map})
    _sector_options += _present_sectors

if _n_stocks == 0:
    st.info('No stocks available for seasonality analysis.')
else:
    if not _sector_map:
        st.info('Sector information unavailable — will analyse all stocks together without sector breakdown.')

    _n_sectors = len(_sector_options) - 1 if _sector_map else 0
    _year_range_start = list(range(2015, current_year + 1))
    _year_range_end = list(range(2015, current_year + 1))
    _default_end_year = max(2015, current_year - 1)
    with st.form('cal_seasonality_inputs'):
        _yr_col1, _yr_col2, _sector_col, _submit_col = st.columns(4)
        _start_year = _yr_col1.selectbox(
            'Start Year', _year_range_start,
            index=_year_range_start.index(2024),
            help='First year of price history to include in the distribution.'
        )
        _end_year = _yr_col2.selectbox(
            'End Year', _year_range_end,
            index=_year_range_end.index(_default_end_year),
            help='Last dividend-event year to include. Completed years are recommended.'
        )
        _input_sector = _sector_col.selectbox(
            'Sector',
            _sector_options,
            key='cal_input_sector',
            index=6,
            help='Only stocks in this sector will be downloaded and analysed.',
        )
        _calculate_submitted = _submit_col.form_submit_button(
            'Calculate Seasonality', width='stretch'
        )

    if _end_year < _start_year:
        st.warning('End year must be >= start year. Please adjust the range.')
        _end_year = _start_year
    if _input_sector == 'All':
        _request_symbols = list(_all_symbols)
        _request_sectors = _sector_options[1:]
    else:
        _request_symbols = [
            symbol for symbol in _all_symbols
            if _sector_map.get(symbol) == _input_sector
        ]
        _request_sectors = [_input_sector]
    _request_stock_count = len(_request_symbols)
    st.caption(
        f'Will analyse **{_request_stock_count} stocks**'
        + (f' in **{_input_sector}**.' if _input_sector != 'All' else
           (f' across **{_n_sectors} sectors**.' if _sector_map else '.'))
        + ' Source: **Yahoo Finance**. Submitted results remain cached while using chart controls.'
    )
    _event_start = f'{_start_year}-01-01'
    _event_end = f'{_end_year}-12-31'
    _div_cal_df = df[['symbol', 'date']].copy()
    _div_cal_df['date'] = pd.to_datetime(_div_cal_df['date'], errors='coerce')
    _div_cal_df = _div_cal_df.dropna(subset=['symbol', 'date'])
    _div_cal_df = _div_cal_df[
        (_div_cal_df['date'] >= pd.Timestamp(_event_start))
        & (_div_cal_df['date'] <= pd.Timestamp(_event_end))
        & (_div_cal_df['symbol'].isin(_request_symbols))
    ]
    _data_version = _source_data_version(_div_cal_df, _sector_map, _request_symbols)
    _request_key = (
        exch, selected_year, _start_year, _end_year, _input_sector,
        _data_version, 'compact_summary_v4',
    )
    if _calculate_submitted:
        st.session_state['cal_seasonality_request'] = _request_key

    _active_request = st.session_state.get('cal_seasonality_request')
    if _active_request is not None and _active_request != _request_key:
        st.info('The inputs changed. Select Calculate Seasonality to refresh the analysis.')

    if _active_request == _request_key:
        _result_cache = st.session_state.setdefault('cal_seasonality_result_cache', {})
        _result_payload = _result_cache.get(_request_key)
        if _result_payload is None:
            with st.spinner(
                f'Fetching price data for {_request_stock_count} stocks '
                f'({_start_year}–{_end_year})…'
            ):
                _symbols_tuple = tuple(_request_symbols)
                _calendar_by_symbol = {
                    symbol: tuple(group['date'])
                    for symbol, group in _div_cal_df.groupby('symbol')
                }
                _stock_event_table, _fetch_failures, _missing_dividend_history = _fetch_all_summaries(
                    _symbols_tuple,
                    _event_start,
                    _event_end,
                    _data_version,
                    _sector_map,
                    _calendar_by_symbol,
                )
                _sector_table = _season_sector_table(
                    _stock_event_table, _request_sectors, _request_symbols
                )
            _result_payload = {
                'sector_table': _sector_table,
                'stock_event_table': _stock_event_table,
                'failures': _fetch_failures,
                'metadata': {
                    'exchange': exch,
                    'selected_year': selected_year,
                    'start_year': _start_year,
                    'end_year': _end_year,
                    'sector': _input_sector,
                    'requested_stocks': _request_stock_count,
                    'missing_dividend_history': _missing_dividend_history,
                    'data_version': _data_version,
                },
            }
            _result_cache[_request_key] = _result_payload
            while len(_result_cache) > 1:
                _result_cache.pop(next(iter(_result_cache)))
        else:
            _stock_event_table = _result_payload['stock_event_table']
            _fetch_failures = _result_payload['failures']
            _missing_dividend_history = _result_payload['metadata']['missing_dividend_history']

        if _fetch_failures:
            st.warning(
                f'Loaded {_request_stock_count - len(_fetch_failures)} of '
                f'{_request_stock_count} stocks. '
                f'{len(_fetch_failures)} failed because price or dividend data was unavailable.'
            )
        if _missing_dividend_history:
            st.warning(
                f'{_missing_dividend_history} stocks had no historical dividend feed; '
                'their event analysis uses only matching dates from the selected Redis calendar.'
            )

        if _stock_event_table.empty:
            st.warning('Could not compute seasonality — price data unavailable.')
        else:
            # ── Sector breakdown table — best time to buy distribution ── #
            with st.expander(
                f'🏆 Best Time to Buy by Sector ({len(_request_sectors)} sectors)',
                expanded=True,
            ):
                _sector_table = _result_payload['sector_table'].copy()
                if _sector_table.empty:
                    st.info('No sector breakdown available.')
                else:
                    _sector_table = _sector_table.sort_values('avg_rel_price', ascending=True).reset_index(drop=True)
                    _sector_table.insert(0, 'rank', range(1, len(_sector_table) + 1))
                    st.dataframe(
                        _sector_table,
                        hide_index=True,
                        width='stretch',
                        column_config={
                            'rank': st.column_config.NumberColumn('Rank', format='%d'),
                            'sector': st.column_config.TextColumn('Sector'),
                            'n_stocks': st.column_config.NumberColumn('Stocks', format='%d'),
                            'best_month': st.column_config.TextColumn('Best Month to Buy'),
                            'avg_rel_price': st.column_config.NumberColumn(
                                'Median Rel Price (%)', help='<100 means below the detrended baseline',
                                format='%.1f'
                            ),
                            'median_days': st.column_config.NumberColumn(
                                'Median Days Before', help='Median calendar days before ex-date to buy', format='%d'
                            ),
                            'p25_days': st.column_config.NumberColumn('Q25 Days', format='%d'),
                            'p75_days': st.column_config.NumberColumn('Q75 Days', format='%d'),
                            'n_events': st.column_config.NumberColumn('Events', format='%d'),
                            'median_recovery': st.column_config.NumberColumn(
                                'Low Entry Median Recovery',
                                help=(
                                    'Median calendar days after ex-date to regain the modeled '
                                    '180-day pre-ex low purchase price'
                                ),
                                format='%d'
                            ),
                            'p25_recovery': st.column_config.NumberColumn('Low Entry Q25', format='%d'),
                            'p75_recovery': st.column_config.NumberColumn('Low Entry Q75', format='%d'),
                            'p90_recovery': st.column_config.NumberColumn('Low Entry Q90', format='%d'),
                            'n_recovery': st.column_config.NumberColumn(
                                'Recovered from Low Entry', format='%d'
                            ),
                            'n_censored': st.column_config.NumberColumn(
                                'Low Entry Traps',
                                help=(
                                    'Events that did not regain the modeled 180-day pre-ex low '
                                    'purchase price within their observed follow-up window'
                                ),
                                format='%d',
                            ),
                            'cum_median_recovery': st.column_config.NumberColumn(
                                'Cum Entry Median Recovery',
                                help='Median calendar days after ex-date to regain the cum-date price',
                                format='%d',
                            ),
                            'cum_n_recovery': st.column_config.NumberColumn(
                                'Recovered from Cum Entry', format='%d'
                            ),
                            'cum_n_censored': st.column_config.NumberColumn(
                                'Cum Entry Traps',
                                help='Events that did not regain the cum-date price during follow-up',
                                format='%d',
                            ),
                        },
                    )
                    st.caption(
                        'Relative price is detrended within complete stock-years and gives each stock equal weight. '
                        'Low-entry and cum-entry recovery use the same eligible events. Kaplan-Meier '
                        'estimates keep unrecovered events censored.'
                    )

            # ── Analyse the cohort selected before downloading ── #
            _selected_sector = _input_sector
            (
                _agg_df, _best_days_df, _low_recovery_df, _cum_recovery_df,
                _best_raw_df, _low_recovery_raw_df, _cum_recovery_raw_df,
            ) = _seasonality_for_sector(_stock_event_table, _selected_sector)

            if _agg_df.empty:
                st.warning(f'No seasonality data for sector "{_selected_sector}".')
            else:
                _scope_label = 'all dividend stocks' if _selected_sector == 'All' else f'the **{_selected_sector}** sector'
                _chart_cols = st.columns(2)

                # ── Chart A — Aggregate monthly bar + IQR band ── #
                with _chart_cols[0]:
                    st.markdown('#### Aggregate Monthly Relative Price')
                    st.caption(f'Median ± IQR across {_scope_label}. Green highlight = historically cheapest month.')

                    _best_row = _agg_df.loc[_agg_df['median'].idxmin()]
                    _best_month_name = _best_row['month_name']

                    _base = alt.Chart(_agg_df)

                    _band = _base.mark_area(opacity=0.18, color='#2ecc71').encode(
                        x=alt.X('month_name:O', sort=MONTH_ORDER, title='Month'),
                        y=alt.Y('q25:Q', title='Relative Price (%)'),
                        y2=alt.Y2('q75:Q'),
                    )
                    _line = _base.mark_line(point=True, color='#27ae60', strokeWidth=2.5).encode(
                        x=alt.X('month_name:O', sort=MONTH_ORDER),
                        y=alt.Y('median:Q', scale=alt.Scale(zero=False)),
                        tooltip=[
                            alt.Tooltip('month_name:O', title='Month'),
                            alt.Tooltip('mean:Q', title='Avg Relative Price', format='.2f'),
                            alt.Tooltip('median:Q', title='Median', format='.2f'),
                            alt.Tooltip('q25:Q', title='Q25', format='.2f'),
                            alt.Tooltip('q75:Q', title='Q75', format='.2f'),
                        ]
                    )
                    _ref = alt.Chart(pd.DataFrame({'y': [100]})).mark_rule(
                        color='#aaaaaa', strokeDash=[6, 4], strokeWidth=1
                    ).encode(y='y:Q')

                    _best_data = _agg_df[_agg_df['month_name'] == _best_month_name]
                    _best_bar = alt.Chart(_best_data).mark_bar(color='#1abc9c', opacity=0.4, width=40).encode(
                        x=alt.X('month_name:O', sort=MONTH_ORDER),
                        y=alt.Y('q25:Q'),
                        y2=alt.Y2('q75:Q'),
                    )

                    _agg_chart = (_band + _best_bar + _line + _ref).properties(height=320)
                    st.altair_chart(_agg_chart, width='stretch')

                    _best_val = _best_row['median']
                    st.success(
                        f'🏆 **{_best_month_name}** is historically the cheapest month for {_scope_label} '
                        f'(median detrended relative price: **{_best_val:.1f}%**)'
                    )

                # ── Chart B — KDE of best days before ex-date ── #
                with _chart_cols[1]:
                    st.markdown('#### Distribution: Days Before Ex-Date to Buy')
                    st.caption(
                        f'For each historical dividend event across {_scope_label}, shows how many '
                        'calendar days before ex-date the price hit its lowest within a 180-day window.'
                    )

                    if not _best_days_df.empty:
                        _median_days = int(_best_days_df['days_before'].median())
                        _best_density = _density_frame(_best_days_df['days_before'], 0, 180)

                        _kde_chart = alt.Chart(_best_density).mark_area(
                            color=alt.Gradient(
                                gradient='linear',
                                stops=[
                                    alt.GradientStop(color='#1a5276', offset=0),
                                    alt.GradientStop(color='#3498db', offset=1),
                                ],
                                x1=1, x2=1, y1=1, y2=0
                            ),
                            line={'color': '#2e86c1'},
                            opacity=0.75,
                        ).encode(
                            x=alt.X('x:Q', title='Calendar Days Before Ex-Date',
                                    scale=alt.Scale(domain=[0, 180])),
                            y=alt.Y('density:Q', title='',
                                    axis=alt.Axis(tickSize=0, domain=False, labelFontSize=0)),
                            tooltip=[alt.Tooltip('x:Q', format='.0f', title='Days Before')]
                        )

                        _median_rule = alt.Chart(
                            pd.DataFrame({'x': [_median_days], 'label': [f'Median: {_median_days}d']})
                        ).mark_rule(color='#f39c12', strokeWidth=2, strokeDash=[5, 3]).encode(
                            x='x:Q'
                        )
                        _median_text = alt.Chart(
                            pd.DataFrame({'x': [_median_days + 1.5], 'y': [0], 'label': [f'Median: {_median_days}d']})
                        ).mark_text(
                            align='left', color='#f39c12', fontSize=11, fontWeight='bold', dy=-8
                        ).encode(x='x:Q', y=alt.Y('y:Q', impute=alt.ImputeParams(value=0)), text='label:N')

                        _kde_full = (_kde_chart + _median_rule + _median_text).properties(height=320)
                        st.altair_chart(_kde_full, width='stretch')

                        _n_events = len(_best_days_df)
                        _p25 = int(_best_days_df['days_before'].quantile(0.25))
                        _p75 = int(_best_days_df['days_before'].quantile(0.75))
                        st.success(
                            f'🎯 Buy **{_median_days} days** before ex-date (median across {_n_events} events). '
                            f'Middle 50% range: **{_p25}–{_p75} days** before.'
                        )
                    else:
                        st.info('No pre-ex best-day data available for these stocks.')

                # ── Chart C — KDE of days after ex-date to recover ── #
                _low_stats = _recovery_stats(_low_recovery_df)
                _cum_stats = _recovery_stats(_cum_recovery_df)
                _low_total = len(_low_recovery_df)
                _cum_total = len(_cum_recovery_df)
                if _low_total or _cum_total:
                    st.markdown('#### Recovery Entry Comparison')
                    _comparison_metrics = st.columns(4)
                    _comparison_metrics[0].metric(
                        'Low Entry Recovered',
                        f"{_low_stats['recovered']:,} / {_low_total:,}",
                    )
                    _comparison_metrics[1].metric(
                        'Low Entry Traps',
                        f"{_low_stats['censored']:,} / {_low_total:,}",
                    )
                    _comparison_metrics[2].metric(
                        'Cum Entry Recovered',
                        f"{_cum_stats['recovered']:,} / {_cum_total:,}",
                    )
                    _comparison_metrics[3].metric(
                        'Cum Entry Traps',
                        f"{_cum_stats['censored']:,} / {_cum_total:,}",
                    )
                    _additional_recoveries = _low_stats['recovered'] - _cum_stats['recovered']
                    st.caption(
                        f'Using the modeled 180-day low entry recovers '
                        f'**{_additional_recoveries:,} additional events** versus entering on the cum date. '
                        'Both calculations use the same eligible dividend events and follow-up window.'
                    )

                _recovery_entry = st.radio(
                    'Recovery entry benchmark',
                    ['Cum-date entry', 'Lowest-date entry'],
                    horizontal=True,
                    key='cal_recovery_entry',
                    help='Select which purchase price drives the detailed recovery charts below.',
                )
                if _recovery_entry == 'Lowest-date entry':
                    _recovery_df = _low_recovery_df
                    _recovery_raw_df = _low_recovery_raw_df
                    _recovery_stats_selected = _low_stats
                    _recovery_target = 'modeled 180-day pre-ex low purchase price'
                    _benchmark_price_col = 'entry_price'
                    _benchmark_yield_title = 'Dividend Yield at Low Entry (%)'
                else:
                    _recovery_df = _cum_recovery_df
                    _recovery_raw_df = _cum_recovery_raw_df
                    _recovery_stats_selected = _cum_stats
                    _recovery_target = 'last trading close before the ex-date (cum-date price)'
                    _benchmark_price_col = 'cum_price'
                    _benchmark_yield_title = 'Dividend Yield at Cum Entry (%)'

                st.markdown('#### Distribution: Days After Ex-Date to Recover')
                st.caption(
                    f'For historical dividend events across {_scope_label}, recovery means the post-ex '
                    f'price reached the {_recovery_target}. '
                    'Summary quantiles also account for events still unrecovered.'
                )

                if not _recovery_df.empty:
                    _recovered_only = _recovery_df[_recovery_df['recovered']].copy()
                    _n_rec = _recovery_stats_selected['recovered']
                    _n_censored = _recovery_stats_selected['censored']
                    _trap_rate = _n_censored / len(_recovery_df) * 100
                    st.caption(
                        f'For **{_recovery_entry}**, {_n_rec:,} events recovered and {_n_censored:,} '
                        f'(**{_trap_rate:.1f}%**) are classified as dividend traps. Recent events may '
                        'have less than 365 calendar days of follow-up.'
                    )
                    _median_rec = _recovery_stats_selected['median']
                    _r_p25 = _recovery_stats_selected['p25']
                    _r_p75 = _recovery_stats_selected['p75']
                    _r_p90 = _recovery_stats_selected['p90']

                    if not _recovered_only.empty:
                        _rec_p99 = float(_recovered_only['days_after'].quantile(0.99))
                        _rec_domain_max = max(120, min(365, int(_rec_p99) + 10))
                        _rec_density = _density_frame(
                            _recovered_only['days_after'], 0, _rec_domain_max
                        )
                        _rec_kde = alt.Chart(_rec_density).mark_area(
                            color=alt.Gradient(
                                gradient='linear',
                                stops=[
                                    alt.GradientStop(color='#4a235a', offset=0),
                                    alt.GradientStop(color='#c39bd3', offset=1),
                                ],
                                x1=1, x2=1, y1=1, y2=0
                            ),
                            line={'color': '#6c3483'},
                            opacity=0.75,
                        ).encode(
                            x=alt.X('x:Q', title='Calendar Days After Ex-Date',
                                    scale=alt.Scale(domain=[0, _rec_domain_max])),
                            y=alt.Y('density:Q', title='',
                                    axis=alt.Axis(tickSize=0, domain=False, labelFontSize=0)),
                            tooltip=[alt.Tooltip('x:Q', format='.0f', title='Days After')]
                        )
                        _rec_layers = _rec_kde
                        if _median_rec is not None:
                            _rec_median_rule = alt.Chart(
                                pd.DataFrame({'x': [_median_rec]})
                            ).mark_rule(
                                color='#f39c12', strokeWidth=2, strokeDash=[5, 3]
                            ).encode(x='x:Q')
                            _rec_layers += _rec_median_rule
                        st.altair_chart(_rec_layers.properties(height=320), width='stretch')

                    if _median_rec is None:
                        st.info(
                            f'The Kaplan-Meier median has not been reached. '
                            f'{_n_rec} events recovered and {_n_censored} remain censored.'
                        )
                    else:
                        _quantile_parts = []
                        if _r_p25 is not None:
                            _quantile_parts.append(f'Q25: **{_r_p25} days**')
                        if _r_p75 is not None:
                            _quantile_parts.append(f'Q75: **{_r_p75} days**')
                        if _r_p90 is not None:
                            _quantile_parts.append(f'Q90: **{_r_p90} days**')
                        _quantile_text = ', '.join(_quantile_parts)
                        st.success(
                            f'📈 Estimated median recovery: **{_median_rec} days** after ex-date. '
                            f'{_n_rec} events recovered; {_n_censored} remain censored.'
                            + (f' {_quantile_text}.' if _quantile_text else '')
                        )
                else:
                    st.info('No post-ex recovery data available for these stocks.')

                # ── Scatter — yield at modeled entry vs. days to recover ── #
                if not _recovery_raw_df.empty:
                    st.markdown('#### Benchmark Yield vs. Recovery or Dividend Trap Follow-Up')
                    st.caption(
                        f'Each point is one historical dividend event across {_scope_label}. Dividend traps '
                        f'can be shown as red diamonds and remain excluded from the regression. Yield and '
                        f'recovery use the selected **{_recovery_entry}** benchmark.'
                    )
                    _scatter_src = _recovery_raw_df.copy()
                    _scatter_src['ex_date'] = pd.to_datetime(_scatter_src['ex_date'])
                    _scatter_df = _scatter_src.dropna(subset=['dividend_amount'])
                    if not _scatter_df.empty and 'dividend_amount' in _scatter_df.columns:
                        _scatter_df = _scatter_df.copy()
                        _scatter_df['recovered'] = _scatter_df['recovered'].fillna(False).astype(bool)
                        _scatter_df['recovery_status'] = np.where(
                            _scatter_df['recovered'], 'Recovered', 'Dividend Trap'
                        )
                        for _numeric_col in ['dividend_amount', _benchmark_price_col, 'days_after']:
                            _scatter_df[_numeric_col] = pd.to_numeric(
                                _scatter_df[_numeric_col], errors='coerce'
                            )
                        _scatter_df['benchmark_yield'] = (
                            _scatter_df['dividend_amount'] / _scatter_df[_benchmark_price_col] * 100
                        )
                        _scatter_clean = _scatter_df.replace([np.inf, -np.inf], np.nan).dropna(
                            subset=['benchmark_yield', 'days_after']
                        )
                        _scatter_clean = _scatter_clean[
                            (_scatter_clean['benchmark_yield'] > 0)
                            & (_scatter_clean[_benchmark_price_col] > 0)
                        ]
                        _scatter_controls = st.columns(2)
                        _show_scatter_traps = _scatter_controls[0].toggle(
                            'Show dividend traps',
                            value=False,
                            key='cal_scatter_show_traps',
                            help=(
                                'Adds unrecovered events at their observed follow-up duration. '
                                'These points can substantially expand the chart domain.'
                            ),
                        )
                        _scatter_axis_layout = _scatter_controls[1].selectbox(
                            'Axis layout',
                            ['Dividend yield on X', 'Recovery days on X'],
                            index=0,
                            key='cal_scatter_axis_layout',
                        )

                        if _show_scatter_traps:
                            _scatter_plot_source = _scatter_clean
                        else:
                            _scatter_plot_source = _scatter_clean[_scatter_clean['recovered']]

                        _scatter_display = _scatter_plot_source
                        if len(_scatter_display) > 5000:
                            _scatter_display = _scatter_display.sample(5000, random_state=42)

                        _duration_title = (
                            'Recovery or Observed Follow-Up Days'
                            if _show_scatter_traps else 'Recovery Days'
                        )
                        if _scatter_axis_layout == 'Recovery days on X':
                            _point_x = alt.X(
                                'days_after:Q', title=_duration_title,
                                scale=alt.Scale(zero=False),
                            )
                            _point_y = alt.Y(
                                'benchmark_yield:Q', title=_benchmark_yield_title,
                                scale=alt.Scale(zero=False),
                            )
                            _regression_x = 'days_after:Q'
                            _regression_y = 'benchmark_yield:Q'
                        else:
                            _point_x = alt.X(
                                'benchmark_yield:Q', title=_benchmark_yield_title,
                                scale=alt.Scale(zero=False),
                            )
                            _point_y = alt.Y(
                                'days_after:Q', title=_duration_title,
                                scale=alt.Scale(zero=False),
                            )
                            _regression_x = 'benchmark_yield:Q'
                            _regression_y = 'days_after:Q'

                        _point_color = alt.Color(
                            'recovery_status:N',
                            title='Status',
                            scale=alt.Scale(
                                domain=['Recovered', 'Dividend Trap'],
                                range=['#2e86c1', '#e74c3c'],
                            ),
                        ) if _show_scatter_traps else alt.value('#2e86c1')
                        _point_shape = alt.Shape(
                            'recovery_status:N',
                            title='Status',
                            scale=alt.Scale(
                                domain=['Recovered', 'Dividend Trap'],
                                range=['circle', 'diamond'],
                            ),
                        ) if _show_scatter_traps else alt.value('circle')

                        _scatter_chart = alt.Chart(_scatter_display).mark_point(
                            filled=True, opacity=0.65, size=70
                        ).encode(
                            x=_point_x,
                            y=_point_y,
                            color=_point_color,
                            shape=_point_shape,
                            tooltip=[
                                alt.Tooltip('symbol:N', title='Symbol'),
                                alt.Tooltip('ex_date:T', title='Ex Date', format='%Y-%m-%d'),
                                alt.Tooltip('recovery_status:N', title='Status'),
                                alt.Tooltip('benchmark_yield:Q', format='.2f', title='Benchmark Yield %'),
                                alt.Tooltip('days_after:Q', format='.0f', title='Days Observed'),
                            ]
                        )

                        _scatter_layers = _scatter_chart
                        _regression_source = _scatter_clean[_scatter_clean['recovered']]
                        if len(_regression_source) >= 2 and _regression_source['benchmark_yield'].nunique() >= 2:
                            _slope, _intercept = np.polyfit(
                                _regression_source['benchmark_yield'], _regression_source['days_after'], 1
                            )
                            _x_min = float(_regression_source['benchmark_yield'].min())
                            _x_max = float(_regression_source['benchmark_yield'].max())
                            _regression_df = pd.DataFrame({
                                'benchmark_yield': [_x_min, _x_max],
                                'days_after': [
                                    _slope * _x_min + _intercept,
                                    _slope * _x_max + _intercept,
                                ],
                            })
                            _regression = alt.Chart(_regression_df).mark_line(
                                color='#f39c12', strokeWidth=2.5, strokeDash=[4, 3]
                            ).encode(x=_regression_x, y=_regression_y)
                            _scatter_layers += _regression
                            st.success(
                                f'Among recovered events, each **+1%** in dividend yield is associated '
                                f'with **{_slope:+.1f}** recovery days in a descriptive linear fit.'
                            )
                        else:
                            st.info('At least two distinct yield values are required for a regression line.')

                        _scatter_navigation = alt.selection_interval(
                            name='scatter_navigation', bind='scales'
                        )
                        _scatter_layers = _scatter_layers.add_params(_scatter_navigation)
                        st.altair_chart(_scatter_layers, width='stretch')
                        _sample_note = ' The chart displays a 5,000-point sample.' if len(_scatter_plot_source) > 5000 else ''
                        if _show_scatter_traps:
                            _scatter_traps = int((~_scatter_plot_source['recovered']).sum())
                            _point_summary = (
                                f'{len(_scatter_plot_source)} matched events, including '
                                f'{_scatter_traps} dividend traps.'
                            )
                        else:
                            _point_summary = f'{len(_scatter_plot_source)} recovered events.'
                        st.caption(
                            f'Drag to pan, scroll or pinch to zoom, and double-click to reset. '
                            f'{_point_summary}{_sample_note}'
                        )
                    else:
                        st.info('Could not match recovery events to dividend yield data.')
                else:
                    st.info('No recovery data for scatter plot.')

                # ── Raw per-event data — merged best-time-to-buy & recovery table ── #
                if st.toggle('👁 Show raw per-event data', key='cal_show_raw_data',
                             help='Show the underlying symbol × ex-date records behind '
                                  'the two distributions above, for sanity checking.'):
                    st.caption(
                        f'Per-event records pooled from {_start_year} through {_end_year}. '
                        'Each row shows, for one dividend event, the best time to buy before '
                        'ex-date and compares recovery from the low-date and cum-date entries.'
                    )

                    _best_view = _best_raw_df[['symbol', 'ex_date', 'ex_price', 'low_date', 'days_before', 'low_price']].copy() \
                        if not _best_raw_df.empty else pd.DataFrame(
                            columns=['symbol', 'ex_date', 'ex_price', 'low_date', 'days_before', 'low_price'])
                    _low_rec_view = _low_recovery_raw_df[
                        ['symbol', 'ex_date', 'recover_date', 'recover_price', 'days_after', 'recovered']
                    ].copy() if not _low_recovery_raw_df.empty else pd.DataFrame(
                        columns=['symbol', 'ex_date', 'recover_date', 'recover_price', 'days_after', 'recovered'])
                    _low_rec_view = _low_rec_view.rename(columns={
                        'recover_date': 'low_recover_date',
                        'recover_price': 'low_recover_price',
                        'days_after': 'low_days_after',
                        'recovered': 'low_recovered',
                    })
                    _cum_rec_view = _cum_recovery_raw_df[
                        ['symbol', 'ex_date', 'cum_date', 'cum_price', 'recover_date',
                         'recover_price', 'days_after', 'recovered']
                    ].copy() if not _cum_recovery_raw_df.empty else pd.DataFrame(
                        columns=['symbol', 'ex_date', 'cum_date', 'cum_price', 'recover_date',
                                 'recover_price', 'days_after', 'recovered'])
                    _cum_rec_view = _cum_rec_view.rename(columns={
                        'recover_date': 'cum_recover_date',
                        'recover_price': 'cum_recover_price',
                        'days_after': 'cum_days_after',
                        'recovered': 'cum_recovered',
                    })

                    _merged_raw = _best_view.merge(
                        _low_rec_view, on=['symbol', 'ex_date'], how='outer'
                    ).merge(_cum_rec_view, on=['symbol', 'ex_date'], how='outer')

                    _div_lookup = _stock_event_table[
                        _stock_event_table['record_type'] == 'event'
                    ][['symbol', 'ex_date', 'dividend_amount']].drop_duplicates(
                        subset=['symbol', 'ex_date']
                    )
                    if not _div_lookup.empty:
                        _merged_raw = _merged_raw.merge(_div_lookup, on=['symbol', 'ex_date'], how='left')
                    else:
                        _merged_raw['dividend_amount'] = None

                    # Price difference between cum-date and ex-date
                    _merged_raw['cum_ex_diff'] = _merged_raw['ex_price'] - _merged_raw['cum_price']
                    _merged_raw['low_recovery_status'] = np.select(
                        [
                            _merged_raw['low_recovered'].eq(True).fillna(False),
                            _merged_raw['low_recovered'].eq(False).fillna(False),
                        ],
                        ['Recovered', 'Dividend Trap'],
                        default='No Recovery Record',
                    )
                    _merged_raw['cum_recovery_status'] = np.select(
                        [
                            _merged_raw['cum_recovered'].eq(True).fillna(False),
                            _merged_raw['cum_recovered'].eq(False).fillna(False),
                        ],
                        ['Recovered', 'Dividend Trap'],
                        default='No Recovery Record',
                    )

                    _col_order = [
                        'symbol', 'ex_date', 'dividend_amount',
                        'ex_price', 'low_date', 'low_price', 'days_before',
                        'cum_date', 'cum_price', 'cum_ex_diff',
                        'low_recovery_status', 'low_recover_date', 'low_recover_price',
                        'low_days_after', 'low_recovered',
                        'cum_recovery_status', 'cum_recover_date', 'cum_recover_price',
                        'cum_days_after', 'cum_recovered',
                    ]
                    _merged_raw = _merged_raw.reindex(columns=_col_order)
                    _merged_raw = _merged_raw.sort_values(['symbol', 'ex_date']).reset_index(drop=True)

                    if _merged_raw.empty:
                        st.info('No per-event data available.')
                    else:
                        st.dataframe(
                            _merged_raw,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                'symbol': st.column_config.TextColumn('Symbol'),
                                'ex_date': st.column_config.TextColumn('Ex Date'),
                                'dividend_amount': st.column_config.NumberColumn('Dividend', format='%.2f'),
                                'ex_price': st.column_config.NumberColumn('Ex Price', format='%.2f'),
                                'low_date': st.column_config.TextColumn('Low Date'),
                                'low_price': st.column_config.NumberColumn('Low Price', format='%.2f'),
                                'days_before': st.column_config.NumberColumn('Days Before', format='%d'),
                                'cum_date': st.column_config.TextColumn('Cum Date'),
                                'cum_price': st.column_config.NumberColumn('Cum Price', format='%.2f'),
                                'cum_ex_diff': st.column_config.NumberColumn('Cum→Ex Diff', format='%.2f'),
                                'low_recovery_status': st.column_config.TextColumn('Low Entry Status'),
                                'low_recover_date': st.column_config.TextColumn('Low Entry Recover Date'),
                                'low_recover_price': st.column_config.NumberColumn(
                                    'Low Entry Recover Price', format='%.2f'
                                ),
                                'low_days_after': st.column_config.NumberColumn(
                                    'Low Entry Days Observed', format='%d'
                                ),
                                'low_recovered': st.column_config.CheckboxColumn('Low Entry Recovered'),
                                'cum_recovery_status': st.column_config.TextColumn('Cum Entry Status'),
                                'cum_recover_date': st.column_config.TextColumn('Cum Entry Recover Date'),
                                'cum_recover_price': st.column_config.NumberColumn(
                                    'Cum Entry Recover Price', format='%.2f'
                                ),
                                'cum_days_after': st.column_config.NumberColumn(
                                    'Cum Entry Days Observed', format='%d'
                                ),
                                'cum_recovered': st.column_config.CheckboxColumn('Cum Entry Recovered'),
                            },
                        )
                        st.caption(f'{len(_merged_raw)} events')
