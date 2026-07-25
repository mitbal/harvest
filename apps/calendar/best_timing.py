import os
import io
import json
import time
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


sl = st.sidebar.radio('Stock List', ['JKSE', 'S&P500'], index=0, horizontal=True)

if sl is None:
    st.stop()

exch = 'jkse' if sl == 'JKSE' else 'sp500'

div_years_key = f'div_cal_years_{exch}'
div_score_key = f'div_score_{exch}'

url = os.environ['REDIS_URL']


def get_logger(name, level=logging.INFO):
    return setup_logging(name, level)

logger = get_logger('best_timing')


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url, socket_connect_timeout=10, socket_timeout=30, socket_keepalive=True, retry_on_timeout=True)
    return r

r = connect_redis(url)


@st.cache_data(max_entries=4, ttl=60 * 60, show_spinner='Downloading dividend data')
def get_data_from_redis(key):
    start = time.time()
    j = r.get(key)
    if j is None:
        logger.warning(f'Missing redis key: {key}')
        return None

    if isinstance(j, bytes) and j.startswith(b'PAR1'):
        return pd.read_parquet(io.BytesIO(j))

    rjson = json.loads(j)
    if 'date' in rjson and 'content' in rjson:
        content = rjson['content']
        return pd.DataFrame(json.loads(content))
    else:
        return pd.DataFrame(rjson)


# ── Year selection ───────────────────────────────────────────────────────────
years_df = get_data_from_redis(div_years_key)
if years_df is not None and 'year' in years_df.columns:
    available_years = sorted(years_df['year'].astype(int).unique().tolist())
else:
    available_years = [current_year]

if len(available_years) == 0:
    available_years = [current_year]
default_year = current_year if current_year in available_years else max(available_years)

selected_year = st.sidebar.selectbox(
    'Calendar Year', available_years,
    index=available_years.index(default_year),
    help='Dividend calendar year whose stocks are analysed for seasonality.'
)

# ── Load calendar & sector data ──────────────────────────────────────────────
div_cal_key = f'div_cal_{exch}_{selected_year}'
df = get_data_from_redis(div_cal_key)
if df is None:
    df = pd.DataFrame(columns=['symbol', 'date'])
else:
    df['date'] = pd.to_datetime(df['date'])

_div_score_df = get_data_from_redis(div_score_key)

_sector_map = {}
if _div_score_df is not None and 'sector' in _div_score_df.columns and 'symbol' in _div_score_df.columns:
    _sector_map = dict(zip(_div_score_df['symbol'], _div_score_df['sector'].fillna('Unknown')))

_all_symbols = sorted(df['symbol'].unique().tolist()) if not df.empty else []

if not _all_symbols:
    with st.spinner('Downloading stock list...'):
        try:
            if exch == 'jkse':
                _stock_df = hd.get_all_idx_stocks()
            else:
                _stock_df = hd.get_all_sp500_stocks()
            _all_symbols = sorted(_stock_df['symbol'].dropna().unique().tolist())
            st.info(
                'Using downloaded stock list. For richer data (sector mapping, historical calendar), '
                'run the pipeline first.'
            )
        except Exception as e:
            logger.error(f'Failed to download stock list: {e}')
            st.error(f'Cannot load stock list: {e}')
            st.stop()

st.title('Best Timing')
st.write(
    f'Pools historical price data across all **{len(_all_symbols)}** dividend-paying stocks '
    f'in the **{selected_year}** calendar to reveal which months are consistently cheaper '
    f'and how many days before / after ex-date prices dip and recover.'
)

# =========================================================================== #
# Aggregate Seasonality — Best Month to Buy Across All Dividend Stocks         #
# =========================================================================== #

MONTH_ORDER = list(calendar.month_abbr[1:])


def _iqr_filter(df, col, multiplier=1.5):
    """Return rows within [Q1 - m×IQR, Q3 + m×IQR] for column *col*."""
    if df is None or df.empty or col not in df.columns:
        return df
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
    return df[(df[col] >= lo) & (df[col] <= hi)]


@st.cache_data(max_entries=512, ttl=60 * 60 * 6, show_spinner=False)
def _fetch_price_for_stock(symbol, start_from='2015-01-01', end_year=None):
    rec = None
    try:
        pdf = hd.get_daily_stock_price(symbol, start_from=start_from)
        if pdf is not None and not pdf.empty and 'close' in pdf.columns:
            pdf = pdf[['date', 'close']].copy()
            pdf['date'] = pd.to_datetime(pdf['date'])
            if end_year is not None:
                pdf = pdf[pdf['date'].dt.year <= end_year]
            rec = {'symbol': symbol, 'price_df': pdf}
    except Exception:
        pass
    if rec is None:
        return None

    try:
        ddf = hd.get_dividend_history_single_stock(symbol)
        if ddf is not None and not ddf.empty and 'date' in ddf.columns:
            _div_cols = ['date']
            if 'adjDividend' in ddf.columns:
                _div_cols.append('adjDividend')
            ddf = ddf[_div_cols].copy()
            ddf['date'] = pd.to_datetime(ddf['date'], errors='coerce')
            ddf = ddf.dropna()
            ddf = ddf[ddf['date'] >= pd.Timestamp(start_from)]
            if end_year is not None:
                ddf = ddf[ddf['date'].dt.year <= end_year]
            rec['div_dates'] = ddf.reset_index(drop=True)
        else:
            rec['div_dates'] = pd.DataFrame(columns=['date'])
    except Exception:
        rec['div_dates'] = pd.DataFrame(columns=['date'])
    return rec


@st.cache_data(max_entries=16, ttl=60 * 60 * 6, show_spinner=False)
def _fetch_all_prices(symbols_tuple, start_from='2015-01-01', end_year=None):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_price_for_stock, s, start_from=start_from, end_year=end_year): s for s in symbols_tuple}
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            if rec is not None:
                results.append(rec)
    return results


def _seasonality_for_sector(price_results, div_cal_df, sector_map, sector='All', pre_ex_days=180):
    sector = sector or 'All'
    if sector != 'All':
        sector_symbols = {s for s, sec in sector_map.items() if sec == sector}
        results = [r for r in price_results if r['symbol'] in sector_symbols]
    else:
        results = list(price_results)

    if not results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    agg_df = hd.calc_aggregate_seasonality(results)

    all_best_days = []
    all_recovery_days = []
    best_raw = []
    recovery_raw = []
    for rec in results:
        sym = rec['symbol']
        if 'div_dates' in rec and not rec['div_dates'].empty:
            sdf_sym = rec['div_dates'].copy()
            sdf_sym['date'] = pd.to_datetime(sdf_sym['date']).dt.strftime('%Y-%m-%d')
        else:
            sdf_sym = div_cal_df[div_cal_df['symbol'] == sym][['date']].copy()
            sdf_sym['date'] = sdf_sym['date'].dt.strftime('%Y-%m-%d')

        best_detail = hd.calc_pre_ex_best_days(rec['price_df'], sdf_sym, pre_ex_days=pre_ex_days, detail=True)
        for ev in best_detail:
            ev['symbol'] = sym
            best_raw.append(ev)
            all_best_days.append(ev['days_before'])

        rec_detail = hd.calc_post_ex_recovery_days(rec['price_df'], sdf_sym, max_lookforward=365, detail=True)
        for ev in rec_detail:
            ev['symbol'] = sym
            recovery_raw.append(ev)
            all_recovery_days.append(ev['days_after'])

    best_days_df = pd.DataFrame({'days_before': all_best_days}) if all_best_days else pd.DataFrame()
    recovery_days_df = pd.DataFrame({'days_after': all_recovery_days}) if all_recovery_days else pd.DataFrame()
    best_raw_df = pd.DataFrame(best_raw) if best_raw else pd.DataFrame()
    recovery_raw_df = pd.DataFrame(recovery_raw) if recovery_raw else pd.DataFrame()
    return agg_df, best_days_df, recovery_days_df, best_raw_df, recovery_raw_df


def _season_sector_table(price_results, div_cal_df, sector_map, sectors, all_symbols):
    rows = []
    for sec in sectors:
        sec_symbols = {s for s, v in sector_map.items() if v == sec and s in all_symbols}
        n_stocks = len(sec_symbols)
        if n_stocks == 0:
            continue
        agg_df, best_df, rec_df, _, rec_raw_df = _seasonality_for_sector(price_results, div_cal_df, sector_map, sec)
        if agg_df.empty:
            rows.append({
                'sector': sec, 'n_stocks': n_stocks, 'best_month': '—',
                'avg_rel_price': None, 'median_days': None, 'p25_days': None, 'p75_days': None, 'n_events': 0,
                'median_recovery': None, 'p25_recovery': None, 'p75_recovery': None, 'p90_recovery': None, 'n_recovery': 0,
            })
            continue
        best_row = agg_df.loc[agg_df['mean'].idxmin()]
        if not best_df.empty:
            best_df_clean = _iqr_filter(best_df, 'days_before')
            median_days = int(best_df_clean['days_before'].median())
            p25_days = int(best_df_clean['days_before'].quantile(0.25))
            p75_days = int(best_df_clean['days_before'].quantile(0.75))
            n_events = len(best_df_clean)
        else:
            median_days = p25_days = p75_days = None
            n_events = 0
        if not rec_df.empty:
            rec_df_clean = _iqr_filter(rec_df, 'days_after')
            median_recovery = int(rec_df_clean['days_after'].median())
            p25_recovery = int(rec_df_clean['days_after'].quantile(0.25))
            p75_recovery = int(rec_df_clean['days_after'].quantile(0.75))
            p90_recovery = int(rec_df_clean['days_after'].quantile(0.90))
            n_recovery = len(rec_df_clean)
        else:
            median_recovery = p25_recovery = p75_recovery = p90_recovery = None
            n_recovery = 0
        rows.append({
            'sector': sec,
            'n_stocks': n_stocks,
            'best_month': best_row['month_name'],
            'avg_rel_price': best_row['mean'],
            'median_days': median_days,
            'p25_days': p25_days,
            'p75_days': p75_days,
            'n_events': n_events,
            'median_recovery': median_recovery,
            'p25_recovery': p25_recovery,
            'p75_recovery': p75_recovery,
            'p90_recovery': p90_recovery,
            'n_recovery': n_recovery,
        })
    return pd.DataFrame(rows, columns=[
        'sector', 'n_stocks', 'best_month', 'avg_rel_price',
        'median_days', 'p25_days', 'p75_days', 'n_events',
        'median_recovery', 'p25_recovery', 'p75_recovery', 'p90_recovery', 'n_recovery'
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

    _col_info, _col_btn = st.columns([3, 1])
    _n_sectors = len(_sector_options) - 1 if _sector_map else 0
    _col_info.caption(
        f'Will fetch price history for up to **{_n_stocks} stocks**'
        + (f' across **{_n_sectors} sectors**.' if _sector_map else '.')
        + ' This may take 30–60 seconds on first load; results are cached for 6 hours.'
    )

    _yr_col1, _yr_col2 = st.columns(2)
    _year_range_start = list(range(2015, current_year + 1))
    _year_range_end = list(range(2015, current_year + 1))
    _start_year = _yr_col1.selectbox(
        'Start Year', _year_range_start,
        index=_year_range_start.index(2015),
        help='First year of price history to include in the distribution.'
    )
    _end_year = _yr_col2.selectbox(
        'End Year', _year_range_end,
        index=_year_range_end.index(current_year),
        help='Last year of price history to include in the distribution.'
    )
    if _end_year < _start_year:
        st.warning('End year must be >= start year. Please adjust the range.')
        _end_year = _start_year
    _start_from_str = f'{_start_year}-01-01'

    if _col_btn.button('🔍 Calculate Seasonality', width='stretch'):
        st.session_state['cal_show_seasonality'] = True

    if st.session_state.get('cal_show_seasonality'):
        with st.spinner(f'Fetching price data for {_n_stocks} stocks ({_start_year}–{_end_year})…'):
            _symbols_tuple = tuple(_all_symbols)
            _price_results = _fetch_all_prices(_symbols_tuple, start_from=_start_from_str, end_year=_end_year)
            _div_cal_df = df[['symbol', 'date']].copy()
            _div_cal_df['date'] = pd.to_datetime(_div_cal_df['date'])

        if not _price_results:
            st.warning('Could not compute seasonality — price data unavailable.')
        else:
            # ── Sector breakdown table — best time to buy distribution ── #
            with st.expander(f'🏆 Best Time to Buy by Sector ({len(_sector_options) - 1} sectors)', expanded=True):
                _sector_table = _season_sector_table(
                    _price_results, _div_cal_df, _sector_map,
                    _sector_options[1:], _all_symbols
                )
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
                                'Avg Rel Price (%)', help='<100 means cheaper than the annual average',
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
                        },
                    )
                    st.caption(
                        'Cheapest-by-relative-price ranks which sector tends to dip most below its annual mean. '
                        '"Median Days Before" pools pre-ex timing across each sector\'s dividend events.'
                    )

            # ── Drill into a single sector with the interactive charts ── #
            st.markdown('#### Drill Down by Sector')
            _selected_sector = st.selectbox(
                'Filter by Sector', _sector_options,
                key='cal_season_sector',
                help='Choose a sector to see its monthly seasonality and best-timing distribution.'
            )

            _agg_df, _best_days_df, _recovery_df, _best_raw_df, _recovery_raw_df = _seasonality_for_sector(
                _price_results, _div_cal_df, _sector_map, _selected_sector
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

                    _best_row = _agg_df.loc[_agg_df['mean'].idxmin()]
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

                    _best_val = _best_row['mean']
                    st.success(
                        f'🏆 **{_best_month_name}** is historically the cheapest month for {_scope_label} '
                        f'(avg relative price: **{_best_val:.1f}%** of annual mean)'
                    )

                # ── Chart B — KDE of best days before ex-date ── #
                with _chart_cols[1]:
                    st.markdown('#### Distribution: Days Before Ex-Date to Buy')
                    st.caption(
                        f'For each historical dividend event across {_scope_label}, shows how many '
                        'calendar days before ex-date the price hit its lowest within a 180-day window.'
                    )

                    if not _best_days_df.empty:
                        _best_days_clean = _iqr_filter(_best_days_df, 'days_before')
                        _outliers_best = len(_best_days_df) - len(_best_days_clean)

                        _median_days = int(_best_days_clean['days_before'].median())

                        _kde_chart = alt.Chart(_best_days_clean).transform_density(
                            'days_before',
                            as_=['Days Before Ex-Date', 'Density'],
                            bandwidth=3,
                        ).mark_area(
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
                            x=alt.X('Days Before Ex-Date:Q', title='Calendar Days Before Ex-Date',
                                    scale=alt.Scale(domain=[0, 180])),
                            y=alt.Y('Density:Q', title='',
                                    axis=alt.Axis(tickSize=0, domain=False, labelFontSize=0)),
                            tooltip=[alt.Tooltip('Days Before Ex-Date:Q', format='.0f', title='Days Before')]
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

                        _n_events = len(_best_days_clean)
                        _p25 = int(_best_days_clean['days_before'].quantile(0.25))
                        _p75 = int(_best_days_clean['days_before'].quantile(0.75))
                        _outlier_note = f' ({_outliers_best} outliers removed via IQR)' if _outliers_best else ''
                        st.success(
                            f'🎯 Buy **{_median_days} days** before ex-date (median across {_n_events} events). '
                            f'Middle 50% range: **{_p25}–{_p75} days** before.{_outlier_note}'
                        )
                    else:
                        st.info('No pre-ex best-day data available for these stocks.')

                # ── Chart C — KDE of days after ex-date to recover ── #
                st.markdown('#### Distribution: Days After Ex-Date to Recover')
                st.caption(
                    f'For each historical dividend event across {_scope_label}, shows how many '
                    'calendar days after ex-date the price took to recover to the cum-date level.'
                )

                if not _recovery_df.empty:
                    _recovery_clean = _iqr_filter(_recovery_df, 'days_after')
                    _outliers_rec = len(_recovery_df) - len(_recovery_clean)

                    _median_rec = int(_recovery_clean['days_after'].median())
                    _r_p25 = int(_recovery_clean['days_after'].quantile(0.25))
                    _r_p75 = int(_recovery_clean['days_after'].quantile(0.75))
                    _r_p90 = int(_recovery_clean['days_after'].quantile(0.90))
                    _n_rec = len(_recovery_clean)

                    if _n_rec > 1:
                        _rec_std = float(_recovery_clean['days_after'].std())
                        _rec_bw = max(1.06 * _rec_std * (_n_rec ** -0.2), 1)
                    else:
                        _rec_bw = 3
                    _rec_p99 = float(_recovery_clean['days_after'].quantile(0.99)) if _n_rec > 0 else 180
                    _rec_domain_max = max(120, int(_rec_p99) + 10)

                    _rec_kde = alt.Chart(_recovery_clean).transform_density(
                        'days_after',
                        as_=['Days After Ex-Date', 'Density'],
                        bandwidth=_rec_bw,
                    ).mark_area(
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
                        x=alt.X('Days After Ex-Date:Q', title='Calendar Days After Ex-Date',
                                scale=alt.Scale(domain=[0, _rec_domain_max])),
                        y=alt.Y('Density:Q', title='',
                                axis=alt.Axis(tickSize=0, domain=False, labelFontSize=0)),
                        tooltip=[alt.Tooltip('Days After Ex-Date:Q', format='.0f', title='Days After')]
                    )

                    _rec_median_rule = alt.Chart(
                        pd.DataFrame({'x': [_median_rec], 'label': [f'Median: {_median_rec}d']})
                    ).mark_rule(color='#f39c12', strokeWidth=2, strokeDash=[5, 3]).encode(x='x:Q')
                    _rec_median_text = alt.Chart(
                        pd.DataFrame({'x': [_median_rec + 1.5], 'y': [0],
                                      'label': [f'Median: {_median_rec}d']})
                    ).mark_text(
                        align='left', color='#f39c12', fontSize=11, fontWeight='bold', dy=-8
                    ).encode(x='x:Q', y=alt.Y('y:Q', impute=alt.ImputeParams(value=0)), text='label:N')

                    st.altair_chart((_rec_kde + _rec_median_rule + _rec_median_text).properties(height=320),
                                    width='stretch')
                    _outlier_note = f' ({_outliers_rec} outliers removed via IQR)' if _outliers_rec else ''
                    st.success(
                        f'📈 Prices recover to the cum-date level a median of **{_median_rec} days** '
                        f'after ex-date (across {_n_rec} events). '
                        f'Middle 50% range: **{_r_p25}–{_r_p75} days**, 90th percentile: **{_r_p90} days** after.{_outlier_note}'
                    )
                else:
                    st.info('No post-ex recovery data available for these stocks.')

                # ── Scatter — Yield on cum date vs. days to recover ── #
                if not _recovery_raw_df.empty:
                    st.markdown('#### Yield on Cum Date vs. Recovery Days')
                    st.caption(
                        f'Each point is one historical dividend event across {_scope_label}. '
                        'Yield = adjDividend / cum_price × 100.'
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
                        _div_lookup = pd.concat(_all_divs, ignore_index=True)
                        _scatter_df = _scatter_src.merge(_div_lookup, on=['symbol', 'ex_date'], how='inner')
                    else:
                        _scatter_df = pd.DataFrame()
                    if not _scatter_df.empty and 'adjDividend' in _scatter_df.columns:
                        _scatter_df['cum_yield'] = _scatter_df['adjDividend'] / _scatter_df['cum_price'] * 100

                        _scatter_clean = _scatter_df.copy()
                        _scatter_clean = _iqr_filter(_scatter_clean, 'cum_yield')
                        _scatter_clean = _iqr_filter(_scatter_clean, 'days_after')
                        _outliers_scatter = len(_scatter_df) - len(_scatter_clean)

                        _scatter_chart = alt.Chart(_scatter_clean).mark_circle(
                            opacity=0.6, size=60
                        ).encode(
                            x=alt.X('cum_yield:Q', title='Dividend Yield on Cum Date (%)',
                                    scale=alt.Scale(zero=False)),
                            y=alt.Y('days_after:Q', title='Days to Recover',
                                    scale=alt.Scale(zero=False)),
                            tooltip=[
                                alt.Tooltip('symbol:N', title='Symbol'),
                                alt.Tooltip('ex_date:T', title='Ex Date', format='%Y-%m-%d'),
                                alt.Tooltip('cum_yield:Q', format='.2f', title='Cum Yield %'),
                                alt.Tooltip('days_after:Q', format='.0f', title='Days to Recover'),
                            ]
                        )

                        _regression = _scatter_chart.transform_regression(
                            'cum_yield', 'days_after'
                        ).mark_line(
                            color='#e74c3c', strokeWidth=2.5, strokeDash=[4, 3]
                        )

                        st.altair_chart(_scatter_chart + _regression, width='stretch')

                        _slope, _intercept = np.polyfit(
                            _scatter_clean['cum_yield'], _scatter_clean['days_after'], 1
                        )
                        st.success(
                            f'Linear regression: on average, each **+1%** in dividend yield '
                            f'corresponds to **{_slope:+.1f}** days to recover. '
                            f'At 0% yield, the model predicts **{_intercept:.0f}** recovery days.'
                        )

                        _outlier_note = f' ({_outliers_scatter} outliers removed via IQR)' if _outliers_scatter else ''
                        st.caption(
                            f'{len(_scatter_clean)} recovery events with matched dividend yield.{_outlier_note}'
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
                        'Per-event records pooled across the 10-year price & dividend history. '
                        'Each row shows, for one dividend event, the best time to buy before '
                        'ex-date and the time it took to recover to the cum-date price afterwards.'
                    )

                    _best_view = _best_raw_df[['symbol', 'ex_date', 'ex_price', 'low_date', 'days_before', 'low_price']].copy() \
                        if not _best_raw_df.empty else pd.DataFrame(
                            columns=['symbol', 'ex_date', 'ex_price', 'low_date', 'days_before', 'low_price'])
                    _rec_view = _recovery_raw_df[
                        ['symbol', 'ex_date', 'cum_date', 'cum_price', 'recover_date', 'recover_price', 'days_after']
                    ].copy() if not _recovery_raw_df.empty else pd.DataFrame(
                        columns=['symbol', 'ex_date', 'cum_date', 'cum_price', 'recover_date', 'recover_price', 'days_after'])

                    _merged_raw = _best_view.merge(_rec_view, on=['symbol', 'ex_date'], how='outer')

                    # Merge dividend amount from the fetched dividend histories
                    _all_divs = []
                    for _pr in _price_results:
                        _divs = _pr.get('div_dates')
                        if _divs is not None and not _divs.empty and 'adjDividend' in _divs.columns:
                            _tmp = _divs[['date', 'adjDividend']].copy()
                            _tmp['symbol'] = _pr['symbol']
                            _tmp.rename(columns={'date': 'ex_date'}, inplace=True)
                            _tmp['ex_date'] = pd.to_datetime(_tmp['ex_date']).dt.strftime('%Y-%m-%d')
                            _all_divs.append(_tmp)
                    if _all_divs:
                        _div_lookup = pd.concat(_all_divs, ignore_index=True).drop_duplicates(subset=['symbol', 'ex_date'])
                        _merged_raw = _merged_raw.merge(_div_lookup, on=['symbol', 'ex_date'], how='left')
                    else:
                        _merged_raw['adjDividend'] = None

                    # Price difference between cum-date and ex-date
                    _merged_raw['cum_ex_diff'] = _merged_raw['ex_price'] - _merged_raw['cum_price']

                    _col_order = [
                        'symbol', 'ex_date', 'adjDividend',
                        'ex_price', 'low_date', 'low_price', 'days_before',
                        'cum_date', 'cum_price', 'cum_ex_diff',
                        'recover_date', 'recover_price', 'days_after',
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
                                'adjDividend': st.column_config.NumberColumn('Dividend', format='%.2f'),
                                'ex_price': st.column_config.NumberColumn('Ex Price', format='%.2f'),
                                'low_date': st.column_config.TextColumn('Low Date'),
                                'low_price': st.column_config.NumberColumn('Low Price', format='%.2f'),
                                'days_before': st.column_config.NumberColumn('Days Before', format='%d'),
                                'cum_date': st.column_config.TextColumn('Cum Date'),
                                'cum_price': st.column_config.NumberColumn('Cum Price', format='%.2f'),
                                'cum_ex_diff': st.column_config.NumberColumn('Cum→Ex Diff', format='%.2f'),
                                'recover_date': st.column_config.TextColumn('Recover Date'),
                                'recover_price': st.column_config.NumberColumn('Recover Price', format='%.2f'),
                                'days_after': st.column_config.NumberColumn('Days After', format='%d'),
                            },
                        )
                        st.caption(f'{len(_merged_raw)} events')
