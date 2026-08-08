import os
import io
import json
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


@st.cache_data(max_entries=512, ttl=60 * 60 * 6, show_spinner=False)
def _fetch_price_for_stock(symbol, event_start, event_end, data_source):
    event_start_ts = pd.Timestamp(event_start)
    event_end_ts = pd.Timestamp(event_end)
    price_start = event_start_ts - pd.Timedelta(days=180)
    price_end = event_end_ts + pd.Timedelta(days=365)

    source_used = data_source
    if data_source == 'Yahoo Finance':
        raw_pdf = hd.get_daily_stock_history_yahoo(
            symbol,
            start_from=price_start.strftime('%Y-%m-%d'),
            end_at=price_end.strftime('%Y-%m-%d'),
        )
    else:
        try:
            raw_pdf = hd.get_daily_stock_price(
                symbol, start_from=price_start.strftime('%Y-%m-%d')
            )
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status != 429:
                raise
            logger.warning(f'FMP quota reached for {symbol}; using Yahoo Finance fallback')
            source_used = 'Yahoo Finance fallback'
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

    rec = {
        'symbol': symbol,
        'price_df': price_df,
        'seasonality_df': seasonality_df,
        'source_used': source_used,
    }
    if source_used.startswith('Yahoo Finance'):
        if 'dividend' in raw_pdf.columns:
            raw_pdf['dividend'] = pd.to_numeric(raw_pdf['dividend'], errors='coerce')
            ddf = raw_pdf.loc[raw_pdf['dividend'] > 0, ['date', 'dividend']].copy()
        else:
            ddf = pd.DataFrame(columns=['date', 'dividend'])
    else:
        dividend_source = 'dag' if symbol.endswith('.JK') else 'fmp'
        ddf = hd.get_dividend_history_single_stock(symbol, source=dividend_source)
    if ddf is not None and not ddf.empty and 'date' in ddf.columns:
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
        rec['div_dates'] = ddf[['date', 'dividend_amount']].reset_index(drop=True)
        rec['has_dividend_history'] = True
    else:
        rec['div_dates'] = pd.DataFrame(columns=['date', 'dividend_amount'])
        rec['has_dividend_history'] = False
    return rec


def _fetch_all_prices(symbols_tuple, event_start, event_end, data_source):
    results = []
    failures = []
    max_workers = 4 if data_source == 'Yahoo Finance' else 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _fetch_price_for_stock, symbol, event_start, event_end, data_source
            ): symbol
            for symbol in symbols_tuple
        }
        for fut in concurrent.futures.as_completed(futures):
            symbol = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                failures.append(symbol)
                reason = _safe_error_reason(exc)
                logger.warning(f'Failed to fetch timing data for {symbol}: {reason}')
    return results, failures


def _seasonality_for_sector(price_results, div_cal_df, sector_map, sector='All', pre_ex_days=180):
    sector = sector or 'All'
    if sector != 'All':
        sector_symbols = {s for s, sec in sector_map.items() if sec == sector}
        results = [r for r in price_results if r['symbol'] in sector_symbols]
    else:
        results = list(price_results)

    if not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    seasonality_results = [
        {'symbol': rec['symbol'], 'price_df': rec.get('seasonality_df', rec['price_df'])}
        for rec in results
    ]
    agg_df = hd.calc_aggregate_seasonality(seasonality_results)

    all_best_days = []
    all_recovery_days = []
    best_raw = []
    recovery_raw = []
    for rec in results:
        sym = rec['symbol']
        if 'div_dates' in rec and not rec['div_dates'].empty:
            sdf_sym = rec['div_dates'].copy()
        else:
            sdf_sym = div_cal_df[div_cal_df['symbol'] == sym][['date']].copy()

        sdf_sym['date'] = pd.to_datetime(sdf_sym['date'], errors='coerce')
        sdf_sym = sdf_sym.dropna(subset=['date'])
        price_min = rec['price_df']['date'].min()
        price_max = rec['price_df']['date'].max()
        sdf_sym = sdf_sym[sdf_sym['date'] <= price_max]
        pre_sdf = sdf_sym[sdf_sym['date'] - pd.Timedelta(days=pre_ex_days) >= price_min].copy()

        best_detail = hd.calc_pre_ex_best_days(rec['price_df'], pre_sdf, pre_ex_days=pre_ex_days, detail=True)
        for ev in best_detail:
            ev['symbol'] = sym
            best_raw.append(ev)
            all_best_days.append(ev['days_before'])

        rec_detail = hd.calc_post_ex_recovery_days(rec['price_df'], sdf_sym, max_lookforward=365, detail=True)
        for ev in rec_detail:
            ev['symbol'] = sym
            recovery_raw.append(ev)
            all_recovery_days.append({
                'days_after': ev['days_after'],
                'recovered': ev['recovered'],
            })

    best_days_df = pd.DataFrame({'days_before': all_best_days}) if all_best_days else pd.DataFrame()
    recovery_days_df = pd.DataFrame(all_recovery_days) if all_recovery_days else pd.DataFrame()
    best_raw_df = pd.DataFrame(best_raw) if best_raw else pd.DataFrame()
    recovery_raw_df = pd.DataFrame(recovery_raw) if recovery_raw else pd.DataFrame()
    return agg_df, best_days_df, recovery_days_df, best_raw_df, recovery_raw_df


def _season_sector_table(
    price_results, div_cal_df, sector_map, sectors, all_symbols, analysis_cache=None
):
    rows = []
    analysis_cache = analysis_cache if analysis_cache is not None else {}
    available_symbols = {rec['symbol'] for rec in price_results}
    for sec in sectors:
        sec_symbols = {
            s for s, v in sector_map.items()
            if v == sec and s in all_symbols and s in available_symbols
        }
        n_stocks = len(sec_symbols)
        if n_stocks == 0:
            continue
        if sec not in analysis_cache:
            analysis_cache[sec] = _seasonality_for_sector(
                price_results, div_cal_df, sector_map, sec
            )
        agg_df, best_df, rec_df, _, _ = analysis_cache[sec]
        if agg_df.empty:
            rows.append({
                'sector': sec, 'n_stocks': n_stocks, 'best_month': '—',
                'avg_rel_price': None, 'median_days': None, 'p25_days': None, 'p75_days': None, 'n_events': 0,
                'median_recovery': None, 'p25_recovery': None, 'p75_recovery': None, 'p90_recovery': None,
                'n_recovery': 0, 'n_censored': 0,
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
        if not rec_df.empty:
            median_recovery = _kaplan_meier_quantile(rec_df, 0.50)
            p25_recovery = _kaplan_meier_quantile(rec_df, 0.25)
            p75_recovery = _kaplan_meier_quantile(rec_df, 0.75)
            p90_recovery = _kaplan_meier_quantile(rec_df, 0.90)
            n_recovery = int(rec_df['recovered'].sum())
            n_censored = len(rec_df) - n_recovery
        else:
            median_recovery = p25_recovery = p75_recovery = p90_recovery = None
            n_recovery = 0
            n_censored = 0
        rows.append({
            'sector': sec,
            'n_stocks': n_stocks,
            'best_month': best_row['month_name'],
            'avg_rel_price': best_row['median'],
            'median_days': median_days,
            'p25_days': p25_days,
            'p75_days': p75_days,
            'n_events': n_events,
            'median_recovery': median_recovery,
            'p25_recovery': p25_recovery,
            'p75_recovery': p75_recovery,
            'p90_recovery': p90_recovery,
            'n_recovery': n_recovery,
            'n_censored': n_censored,
        })
    return pd.DataFrame(rows, columns=[
        'sector', 'n_stocks', 'best_month', 'avg_rel_price',
        'median_days', 'p25_days', 'p75_days', 'n_events',
        'median_recovery', 'p25_recovery', 'p75_recovery', 'p90_recovery', 'n_recovery', 'n_censored'
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
        _yr_col1, _yr_col2, _source_col = st.columns(3)
        _start_year = _yr_col1.selectbox(
            'Start Year', _year_range_start,
            index=_year_range_start.index(2015),
            help='First year of price history to include in the distribution.'
        )
        _end_year = _yr_col2.selectbox(
            'End Year', _year_range_end,
            index=_year_range_end.index(_default_end_year),
            help='Last dividend-event year to include. Completed years are recommended.'
        )
        _data_source = _source_col.selectbox(
            'Market Data Source',
            ['Yahoo Finance', 'Financial Modeling Prep'],
            index=0,
            help=(
                'Yahoo Finance is recommended for bulk analysis because it does not consume the FMP API quota. '
                'FMP automatically falls back to Yahoo Finance when HTTP 429 is returned.'
            ),
        )
        _calculate_submitted = st.form_submit_button('🔍 Calculate Seasonality', width='stretch')

    if _end_year < _start_year:
        st.warning('End year must be >= start year. Please adjust the range.')
        _end_year = _start_year
    st.caption(
        f'Will analyse up to **{_n_stocks} stocks**'
        + (f' across **{_n_sectors} sectors**.' if _sector_map else '.')
        + f' Source: **{_data_source}**. Submitted results remain cached while using chart controls.'
    )
    _event_start = f'{_start_year}-01-01'
    _event_end = f'{_end_year}-12-31'
    _request_key = (exch, selected_year, _start_year, _end_year, _data_source)
    if _calculate_submitted:
        st.session_state['cal_seasonality_request'] = _request_key

    _active_request = st.session_state.get('cal_seasonality_request')
    if _active_request is not None and _active_request != _request_key:
        st.info('The inputs changed. Select Calculate Seasonality to refresh the analysis.')

    if _active_request == _request_key:
        _result_cache = st.session_state.setdefault('cal_seasonality_result_cache', {})
        _result_payload = _result_cache.get(_request_key)
        if _result_payload is None:
            with st.spinner(f'Fetching price data for {_n_stocks} stocks ({_start_year}–{_end_year})…'):
                _symbols_tuple = tuple(_all_symbols)
                _price_results, _fetch_failures = _fetch_all_prices(
                    _symbols_tuple, _event_start, _event_end, _data_source
                )
                _div_cal_df = df[['symbol', 'date']].copy()
                _div_cal_df['date'] = pd.to_datetime(_div_cal_df['date'], errors='coerce')
                _div_cal_df = _div_cal_df.dropna(subset=['symbol', 'date'])
                _div_cal_df = _div_cal_df[
                    (_div_cal_df['date'] >= pd.Timestamp(_event_start))
                    & (_div_cal_df['date'] <= pd.Timestamp(_event_end))
                ]
            _result_payload = {
                'price_results': _price_results,
                'fetch_failures': _fetch_failures,
                'div_cal_df': _div_cal_df,
                'sector_analysis': {},
            }
            _result_cache[_request_key] = _result_payload
            while len(_result_cache) > 1:
                _result_cache.pop(next(iter(_result_cache)))
        else:
            _price_results = _result_payload['price_results']
            _fetch_failures = _result_payload['fetch_failures']
            _div_cal_df = _result_payload['div_cal_df']

        if _fetch_failures:
            st.warning(
                f'Loaded {_n_stocks - len(_fetch_failures)} of {_n_stocks} stocks. '
                f'{len(_fetch_failures)} failed because price or dividend data was unavailable.'
            )
        _fmp_fallbacks = sum(
            rec.get('source_used') == 'Yahoo Finance fallback' for rec in _price_results
        )
        if _fmp_fallbacks:
            st.warning(
                f'FMP quota was reached for {_fmp_fallbacks} stocks; Yahoo Finance fallback was used.'
            )
        _missing_dividend_history = sum(
            not rec.get('has_dividend_history', False) for rec in _price_results
        )
        if _missing_dividend_history:
            st.warning(
                f'{_missing_dividend_history} stocks had no historical dividend feed; '
                'their event analysis uses only matching dates from the selected Redis calendar.'
            )

        if not _price_results:
            st.warning('Could not compute seasonality — price data unavailable.')
        else:
            # ── Sector breakdown table — best time to buy distribution ── #
            with st.expander(f'🏆 Best Time to Buy by Sector ({len(_sector_options) - 1} sectors)', expanded=True):
                if 'sector_table' not in _result_payload:
                    with st.spinner('Calculating sector summaries...'):
                        _result_payload['sector_table'] = _season_sector_table(
                            _price_results, _div_cal_df, _sector_map,
                            _sector_options[1:], _all_symbols,
                            analysis_cache=_result_payload['sector_analysis'],
                        )
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
                                'Median Days to Recover',
                                help='Median calendar days after ex-date to return to the cum-date price',
                                format='%d'
                            ),
                            'p25_recovery': st.column_config.NumberColumn('Q25 Recover', format='%d'),
                            'p75_recovery': st.column_config.NumberColumn('Q75 Recover', format='%d'),
                            'p90_recovery': st.column_config.NumberColumn('Q90 Recover', format='%d'),
                            'n_recovery': st.column_config.NumberColumn('Recoveries', format='%d'),
                            'n_censored': st.column_config.NumberColumn(
                                'Dividend Traps',
                                help='Events not recovered within their observed follow-up window',
                                format='%d',
                            ),
                        },
                    )
                    st.caption(
                        'Relative price is detrended within complete stock-years and gives each stock equal weight. '
                        'Recovery quantiles use Kaplan-Meier estimates so unrecovered events remain censored.'
                    )

            # ── Drill into a single sector with the interactive charts ── #
            st.markdown('#### Drill Down by Sector')
            _selected_sector = st.selectbox(
                'Filter by Sector', _sector_options,
                key='cal_season_sector',
                help='Choose a sector to see its monthly seasonality and best-timing distribution.'
            )

            _sector_analysis_cache = _result_payload.setdefault('sector_analysis', {})
            if _selected_sector not in _sector_analysis_cache:
                with st.spinner(f'Calculating {_selected_sector} analysis...'):
                    _sector_analysis_cache[_selected_sector] = _seasonality_for_sector(
                        _price_results, _div_cal_df, _sector_map, _selected_sector
                    )
            _agg_df, _best_days_df, _recovery_df, _best_raw_df, _recovery_raw_df = (
                _sector_analysis_cache[_selected_sector]
            )

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
                st.markdown('#### Distribution: Days After Ex-Date to Recover')
                st.caption(
                    f'For historical dividend events across {_scope_label}, the chart shows observed '
                    'recovery times; summary quantiles also account for events still unrecovered.'
                )

                if not _recovery_df.empty:
                    _recovered_only = _recovery_df[_recovery_df['recovered']].copy()
                    _n_rec = len(_recovered_only)
                    _n_censored = len(_recovery_df) - _n_rec
                    _trap_rate = _n_censored / len(_recovery_df) * 100
                    _recovery_metrics = st.columns(3)
                    _recovery_metrics[0].metric('Recovery Events', f'{len(_recovery_df):,}')
                    _recovery_metrics[1].metric('Recovered', f'{_n_rec:,}')
                    _recovery_metrics[2].metric(
                        'Dividend Traps',
                        f'{_n_censored:,}',
                        help=(
                            'Dividend events whose price has not returned to the cum-date level '
                            'within the observed follow-up window.'
                        ),
                    )
                    st.caption(
                        f'**{_trap_rate:.1f}%** of observed dividend events are currently classified '
                        'as dividend traps. Recent events may have less than 365 calendar days of follow-up.'
                    )
                    _median_rec = _kaplan_meier_quantile(_recovery_df, 0.50)
                    _r_p25 = _kaplan_meier_quantile(_recovery_df, 0.25)
                    _r_p75 = _kaplan_meier_quantile(_recovery_df, 0.75)
                    _r_p90 = _kaplan_meier_quantile(_recovery_df, 0.90)

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

                # ── Scatter — Yield on cum date vs. days to recover ── #
                if not _recovery_raw_df.empty:
                    st.markdown('#### Yield vs. Recovery or Dividend Trap Follow-Up')
                    st.caption(
                        f'Each point is one historical dividend event across {_scope_label}. Dividend traps '
                        'can be shown as red diamonds and remain excluded from the regression.'
                    )
                    _scatter_src = _recovery_raw_df.copy()
                    _scatter_src['ex_date'] = pd.to_datetime(_scatter_src['ex_date'])

                    _all_divs = []
                    for _pr in _price_results:
                        _divs = _pr.get('div_dates')
                        if _divs is not None and not _divs.empty:
                            _tmp = _divs.copy()
                            _tmp['symbol'] = _pr['symbol']
                            _tmp.rename(columns={'date': 'ex_date'}, inplace=True)
                            _all_divs.append(_tmp)
                    if _all_divs:
                        _div_lookup = pd.concat(_all_divs, ignore_index=True).drop_duplicates(
                            subset=['symbol', 'ex_date']
                        )
                        _scatter_df = _scatter_src.merge(_div_lookup, on=['symbol', 'ex_date'], how='inner')
                    else:
                        _scatter_df = pd.DataFrame()
                    if not _scatter_df.empty and 'dividend_amount' in _scatter_df.columns:
                        _scatter_df = _scatter_df.copy()
                        _scatter_df['recovered'] = _scatter_df['recovered'].fillna(False).astype(bool)
                        _scatter_df['recovery_status'] = np.where(
                            _scatter_df['recovered'], 'Recovered', 'Dividend Trap'
                        )
                        for _numeric_col in ['dividend_amount', 'cum_price', 'days_after']:
                            _scatter_df[_numeric_col] = pd.to_numeric(
                                _scatter_df[_numeric_col], errors='coerce'
                            )
                        _scatter_df['cum_yield'] = (
                            _scatter_df['dividend_amount'] / _scatter_df['cum_price'] * 100
                        )
                        _scatter_clean = _scatter_df.replace([np.inf, -np.inf], np.nan).dropna(
                            subset=['cum_yield', 'days_after']
                        )
                        _scatter_clean = _scatter_clean[
                            (_scatter_clean['cum_yield'] > 0) & (_scatter_clean['cum_price'] > 0)
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
                            ['Recovery days on X', 'Dividend yield on X'],
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
                                'cum_yield:Q', title='Dividend Yield on Cum Date (%)',
                                scale=alt.Scale(zero=False),
                            )
                            _regression_x = 'days_after:Q'
                            _regression_y = 'cum_yield:Q'
                        else:
                            _point_x = alt.X(
                                'cum_yield:Q', title='Dividend Yield on Cum Date (%)',
                                scale=alt.Scale(zero=False),
                            )
                            _point_y = alt.Y(
                                'days_after:Q', title=_duration_title,
                                scale=alt.Scale(zero=False),
                            )
                            _regression_x = 'cum_yield:Q'
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
                                alt.Tooltip('cum_yield:Q', format='.2f', title='Cum Yield %'),
                                alt.Tooltip('days_after:Q', format='.0f', title='Days Observed'),
                            ]
                        )

                        _scatter_layers = _scatter_chart
                        _regression_source = _scatter_clean[_scatter_clean['recovered']]
                        if len(_regression_source) >= 2 and _regression_source['cum_yield'].nunique() >= 2:
                            _slope, _intercept = np.polyfit(
                                _regression_source['cum_yield'], _regression_source['days_after'], 1
                            )
                            _x_min = float(_regression_source['cum_yield'].min())
                            _x_max = float(_regression_source['cum_yield'].max())
                            _regression_df = pd.DataFrame({
                                'cum_yield': [_x_min, _x_max],
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
                        'ex-date and either its recovery time or its current censored follow-up.'
                    )

                    _best_view = _best_raw_df[['symbol', 'ex_date', 'ex_price', 'low_date', 'days_before', 'low_price']].copy() \
                        if not _best_raw_df.empty else pd.DataFrame(
                            columns=['symbol', 'ex_date', 'ex_price', 'low_date', 'days_before', 'low_price'])
                    _rec_view = _recovery_raw_df[
                        ['symbol', 'ex_date', 'cum_date', 'cum_price', 'recover_date',
                         'recover_price', 'days_after', 'recovered']
                    ].copy() if not _recovery_raw_df.empty else pd.DataFrame(
                        columns=['symbol', 'ex_date', 'cum_date', 'cum_price', 'recover_date',
                                 'recover_price', 'days_after', 'recovered'])

                    _merged_raw = _best_view.merge(_rec_view, on=['symbol', 'ex_date'], how='outer')

                    # Merge dividend amount from the fetched dividend histories
                    _all_divs = []
                    for _pr in _price_results:
                        _divs = _pr.get('div_dates')
                        if _divs is not None and not _divs.empty and 'dividend_amount' in _divs.columns:
                            _tmp = _divs[['date', 'dividend_amount']].copy()
                            _tmp['symbol'] = _pr['symbol']
                            _tmp.rename(columns={'date': 'ex_date'}, inplace=True)
                            _tmp['ex_date'] = pd.to_datetime(_tmp['ex_date']).dt.strftime('%Y-%m-%d')
                            _all_divs.append(_tmp)
                    if _all_divs:
                        _div_lookup = pd.concat(_all_divs, ignore_index=True).drop_duplicates(subset=['symbol', 'ex_date'])
                        _merged_raw = _merged_raw.merge(_div_lookup, on=['symbol', 'ex_date'], how='left')
                    else:
                        _merged_raw['dividend_amount'] = None

                    # Price difference between cum-date and ex-date
                    _merged_raw['cum_ex_diff'] = _merged_raw['ex_price'] - _merged_raw['cum_price']
                    _merged_raw['recovery_status'] = np.select(
                        [
                            _merged_raw['recovered'].eq(True).fillna(False),
                            _merged_raw['recovered'].eq(False).fillna(False),
                        ],
                        ['Recovered', 'Dividend Trap'],
                        default='No Recovery Record',
                    )

                    _col_order = [
                        'symbol', 'ex_date', 'recovery_status', 'dividend_amount',
                        'ex_price', 'low_date', 'low_price', 'days_before',
                        'cum_date', 'cum_price', 'cum_ex_diff',
                        'recover_date', 'recover_price', 'days_after', 'recovered',
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
                                'recovery_status': st.column_config.TextColumn('Recovery Status'),
                                'dividend_amount': st.column_config.NumberColumn('Dividend', format='%.2f'),
                                'ex_price': st.column_config.NumberColumn('Ex Price', format='%.2f'),
                                'low_date': st.column_config.TextColumn('Low Date'),
                                'low_price': st.column_config.NumberColumn('Low Price', format='%.2f'),
                                'days_before': st.column_config.NumberColumn('Days Before', format='%d'),
                                'cum_date': st.column_config.TextColumn('Cum Date'),
                                'cum_price': st.column_config.NumberColumn('Cum Price', format='%.2f'),
                                'cum_ex_diff': st.column_config.NumberColumn('Cum→Ex Diff', format='%.2f'),
                                'recover_date': st.column_config.TextColumn('Recover Date'),
                                'recover_price': st.column_config.NumberColumn('Recover Price', format='%.2f'),
                                'days_after': st.column_config.NumberColumn(
                                    'Days Observed After Ex-Date', format='%d'
                                ),
                                'recovered': st.column_config.CheckboxColumn('Recovered'),
                            },
                        )
                        st.caption(f'{len(_merged_raw)} events')
