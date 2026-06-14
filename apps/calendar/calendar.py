import os
import io
import json
import time
import logging
import calendar
import concurrent.futures
from datetime import datetime

import redis
import lesley
import pandas as pd
import altair as alt
import streamlit as st

import harvest.data as hd
import harvest.plot as hp
from harvest.utils import setup_logging


current_year = datetime.today().year
current_month = datetime.today().month


sl = st.sidebar.radio('Stock List', ['JKSE', 'S&P500'], index=0, horizontal=True)


if sl is None:
    print('Please select one of the options above')
    st.stop()

if sl == 'JKSE':
    exch = 'jkse'
else:
    exch = 'sp500'

div_years_key = f'div_cal_years_{exch}'
div_score_key = f'div_score_{exch}'

url = os.environ['REDIS_URL']

#### Function definition


def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger

logger = get_logger('calendar')


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url, socket_connect_timeout=10, socket_timeout=30, socket_keepalive=True, retry_on_timeout=True)
    return r

r = connect_redis(url)


@st.cache_data(ttl=60*60, show_spinner='Downloading dividend data')
def get_data_from_redis(key):

    start = time.time()
    j = r.get(key)
    if j is None:
        logger.error(f'Missing redis key: {key}')
        st.error('Dividend calendar data is not available. Please refresh after the pipeline finishes.')
        st.stop()
    end = time.time()

    if isinstance(j, bytes) and j.startswith(b'PAR1'):
        # logger.info(f'get redis key: {key}, total time: {end-start:.4f} seconds (parquet)')
        return pd.read_parquet(io.BytesIO(j))

    rjson = json.loads(j)
    
    if 'date' in rjson and 'content' in rjson:
        last_updated = rjson['date']
        # logger.info(f'get redis key: {key}, total time: {end-start:.4f} seconds, last updated: {last_updated}')
        content = rjson['content']
        return pd.DataFrame(json.loads(content))
    else:
        # logger.info(f'get redis key: {key}, total time: {end-start:.4f} seconds')
        return pd.DataFrame(rjson)


def prep_div_month(df, month_idx=1):
    
    month_df = df[df.date.dt.month == month_idx].copy().reset_index(drop=True)
    month_df.sort_values(by='yield', ascending=False, inplace=True)

    month_df['rank'] = list(range(1, len(month_df)+1))
    month_df['div_yield'] = month_df['yield'].apply(lambda x: f'{x:2.2f}%')
    month_df['ex_date'] = month_df['date'].dt.strftime('%d %b')
    month_df['url_link'] = month_df['symbol'].apply(lambda x: f'https://panendividen.com/stock_picker?stock={x}')
    
    month_df.rename(columns={'symbol': 'stock',
                            'adjDividend': 'dividend'}, inplace=True)
    
    return month_df[['rank', 'url_link', 'ex_date', 'div_yield', 'dividend', 'price']]


#### End of Function definition

column_config = {
    'url_link': st.column_config.LinkColumn(
        'Stock',
        help='The code of the stock',
        max_chars=10,

        display_text=r"https://panendividen\.com/stock_picker\?stock=([A-Z]+)(?=\.JK|$)"
    ),
    'rank': st.column_config.NumberColumn(
        'Rank',
        help='Rank of the stock based on yield for this month',
        format='%d',
    ),
    'ex_date': st.column_config.TextColumn(
        'Ex Date',
        help='Ex-dividend date',
    ),
    'div_yield': st.column_config.TextColumn(
        'Yield',
        help='Dividend yield',
    ),
    'dividend': st.column_config.NumberColumn(
        'Dividend',
        help='Dividend amount',
        format='%.01f',
    ),
    'price': st.column_config.NumberColumn(
        'Price',
        help='Stock price',
        format='localized'
    ),
}


years_df = get_data_from_redis(div_years_key)
available_years = sorted(years_df['year'].astype(int).unique().tolist())
if len(available_years) == 0:
    available_years = [current_year]
default_year = current_year if current_year in available_years else max(available_years)

# ── URL query-param defaults ──────────────────────────────────────────────────
# Supported params:
#   ?view=yearly|monthly   (default: yearly)
#   ?year=2025             (default: current / latest available year)
#   ?month=3               (1-12, only relevant when view=monthly)
# Example: ?view=monthly&year=2025&month=6
qp = st.query_params

_qp_view = qp.get('view', 'yearly').lower()
_default_view_idx = 1 if _qp_view == 'monthly' else 0

_qp_year = qp.get('year', str(default_year))
try:
    _qp_year_int = int(_qp_year)
    _default_year_idx = available_years.index(_qp_year_int) if _qp_year_int in available_years else available_years.index(default_year)
except (ValueError, IndexError):
    _default_year_idx = available_years.index(default_year)

months = list(calendar.month_name)  # ['', 'January', ..., 'December']
_qp_month = qp.get('month', str(current_month))
try:
    _qp_month_int = int(_qp_month)
    _default_month_idx = (_qp_month_int - 1) if 1 <= _qp_month_int <= 12 else (current_month - 1)
except ValueError:
    _default_month_idx = current_month - 1
# ─────────────────────────────────────────────────────────────────────────────

view_control = st.sidebar.radio(
    'Calendar View', ['Full Year', 'Single Month'],
    index=_default_view_idx, horizontal=True
)

time_param_cols = st.columns(2)
selected_year = time_param_cols[0].selectbox('Year', available_years, index=_default_year_idx)

month_index = None
if view_control == 'Single Month':
    select_month = time_param_cols[1].selectbox('Month', months[1:], index=_default_month_idx)
    month_index = months.index(select_month)

# Keep URL params in sync with the current widget state
_new_view = 'monthly' if view_control == 'Single Month' else 'yearly'
if _new_view == 'monthly' and month_index is not None:
    st.query_params.update({'view': _new_view, 'year': str(selected_year), 'month': str(month_index)})
else:
    st.query_params.update({'view': _new_view, 'year': str(selected_year)})
    qp_keys = list(st.query_params.keys())
    if 'month' in qp_keys:
        del st.query_params['month']

div_cal_key = f'div_cal_{exch}_{selected_year}'
df = get_data_from_redis(div_cal_key)
df['date'] = pd.to_datetime(df['date'])
df['yield'] = df['adjDividend'] / df['price'] * 100

st.title(f'Dividend Calendar {selected_year+1}')
st.write(f'*based on {selected_year} data*')

div_score_df = get_data_from_redis(div_score_key)

if sl == 'JKSE':
    show_next_year = True
    div_score_df = div_score_df[['symbol', 'is_syariah']]
    is_syariah = st.sidebar.toggle('Syariah Only?')
    if is_syariah:
        df = df.merge(div_score_df, on='symbol', how='left')
        df = df[df['is_syariah'] == True]
else:
    show_next_year = False

if view_control == 'Full Year':
    cal = hp.plot_dividend_calendar(df, show_next_year=False, sl=sl)
    st.altair_chart(cal)

    idx = 1
    for i in range(3):

        row = st.container()
        row_cols = row.columns(4)

        for j in range(4):
            row_cols[j].write(f'{calendar.month_name[idx]}')

            month_df = prep_div_month(df, month_idx=idx)
            row_cols[j].dataframe(
                hide_index=True,
                column_config=column_config,
                data=month_df, height=210)
            idx += 1

else:
    st.write(select_month)
    cols = st.columns(2)

    labels = df.apply(lambda x: f"{x['date'].strftime('%d %b')}: {x['symbol']} ({x['yield']:2.2f}%)", axis=1)
    month_cal = lesley.month_plot(df['date'], df['yield'], labels=labels, month=month_index, show_date=True, width=500)
    cols[0].altair_chart(month_cal)

    month_df = prep_div_month(df, month_idx=month_index)
    cols[1].dataframe(
        hide_index=True,
        column_config=column_config,
        data=month_df
    )

stats_df = df if view_control == 'Full Year' else df[df['date'].dt.month == month_index]
if stats_df.empty:
    st.info('No dividend statistics available for this selection.')
else:
    st.divider()
    st.subheader('Dividend Statistics')

    c1, c2, c3 = st.columns(3)
    c1.metric('Events', len(stats_df))
    c2.metric('Avg Yield', f"{stats_df['yield'].mean():.2f}%")
    c3.metric('Median Yield', f"{stats_df['yield'].median():.2f}%")

    top_row = stats_df.loc[stats_df['yield'].idxmax()]
    st.caption(f"Highest yield: {top_row['symbol']} on {top_row['date'].strftime('%d %b')} at {top_row['yield']:.2f}% (dividend {top_row['adjDividend']:.2f}).")

    charts = st.columns(2)

    if view_control == 'Full Year':
        monthly_summary = (
            stats_df
            .assign(month=stats_df['date'].dt.month)
            .groupby('month')['yield']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_yield', 'count': 'events'})
        )
        month_order = list(calendar.month_abbr[1:])
        monthly_summary['month_name'] = monthly_summary['month'].apply(lambda m: calendar.month_abbr[m])
        monthly_summary['month_name'] = pd.Categorical(monthly_summary['month_name'], categories=month_order, ordered=True)
        monthly_summary = monthly_summary.sort_values('month_name')

        combo_base = alt.Chart(monthly_summary)
        events_bar = combo_base.mark_bar(color='#1f77b4').encode(
            x=alt.X('month_name:N', sort=month_order, title='Month'),
            y=alt.Y('events:Q', title='Dividend count'),
            tooltip=[alt.Tooltip('month_name:N', title='Month'), alt.Tooltip('events:Q', title='Count')]
        )
        yield_line = combo_base.mark_line(color='#ff7f0e', point=True).encode(
            x=alt.X('month_name:N', sort=month_order),
            y=alt.Y('avg_yield:Q', title='Avg yield (%)', axis=alt.Axis(titleColor='#ff7f0e'), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('month_name:N', title='Month'), alt.Tooltip('avg_yield:Q', format='.2f', title='Avg yield %')]
        )
        charts[0].altair_chart((events_bar + yield_line).resolve_scale(y='independent').properties(height=260), width='stretch')
    else:
        day_summary = (
            stats_df
            .assign(day=stats_df['date'].dt.day)
            .groupby('day')['yield']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'avg_yield', 'count': 'events'})
        )
        combo_base = alt.Chart(day_summary)
        events_bar = combo_base.mark_bar(color='#1f77b4').encode(
            x=alt.X('day:O', title='Ex-Date day'),
            y=alt.Y('events:Q', title='Dividend count'),
            tooltip=[alt.Tooltip('day:O', title='Day'), alt.Tooltip('events:Q', title='Count')]
        )
        yield_line = combo_base.mark_line(color='#ff7f0e', point=True).encode(
            x='day:O',
            y=alt.Y('avg_yield:Q', title='Avg yield (%)', axis=alt.Axis(titleColor='#ff7f0e'), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip('day:O', title='Day'), alt.Tooltip('avg_yield:Q', format='.2f', title='Avg yield %')]
        )
        charts[0].altair_chart((events_bar + yield_line).resolve_scale(y='independent').properties(height=260), width='stretch')

    kde = alt.Chart(stats_df[['yield']]).transform_density(
        'yield',
        as_=['Yield', 'Density']
    ).mark_area(
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#008631', offset=1)],
            x1=1, x2=1, y1=1, y2=0
        ),
        line={'color': '#008631'}
    ).encode(
        x=alt.X('Yield:Q', title='Dividend Yield (%)'),
        y=alt.Y('Density:Q', title='', axis=alt.Axis(tickSize=0, domain=False, labelAngle=0, labelFontSize=0)),
        tooltip=[alt.Tooltip('Yield:Q', format='.2f', title='Yield %')]
    )
    mean_rule = alt.Chart(pd.DataFrame({'mean_yield': [stats_df['yield'].mean()]})).mark_rule(color='red').encode(
        x='mean_yield:Q',
        tooltip=[alt.Tooltip('mean_yield:Q', format='.2f', title='Average yield')]
    )
    charts[1].altair_chart((kde + mean_rule).properties(height=260), width='stretch')

    freq_df = (
        df.groupby('symbol')['date']
        .nunique()
        .reset_index(name='payments')
    )
    freq_df['payments'] = freq_df['payments'].clip(upper=4)
    category_order = ['Once per year', 'Twice per year', 'Thrice per year', '4 times per year']
    freq_labels = {1: category_order[0], 2: category_order[1], 3: category_order[2], 4: category_order[3]}
    freq_df['category'] = freq_df['payments'].map(freq_labels)

    freq_summary = (
        freq_df.groupby('category')
        .agg(
            count=('symbol', 'size'),
            stocks=('symbol', lambda s: ', '.join(sorted(s)))
        )
        .reindex(category_order)
        .reset_index()
        .rename(columns={'category': 'frequency'})
    )
    freq_summary['count'] = freq_summary['count'].fillna(0).astype(int)
    freq_summary['stocks'] = freq_summary['stocks'].fillna('')

    st.subheader('Payout Frequency (per year)')
    st.dataframe(
        freq_summary,
        hide_index=True,
        width='stretch',
        column_config={
            'frequency': st.column_config.TextColumn('Frequency'),
            'count': st.column_config.NumberColumn('Companies', format='%d'),
            'stocks': st.column_config.TextColumn('Stocks')
        }
    )

# =========================================================================== #
# Aggregate Seasonality — Best Month to Buy Across All Dividend Stocks         #
# =========================================================================== #

st.divider()
st.subheader('📅 Best Month to Buy — Aggregate Seasonality')
st.write(
    'Pools historical price data across all dividend stocks in this calendar year '
    'to reveal which calendar months are consistently cheaper than the annual average. '
    'A relative price **< 100** means stocks are trading below their yearly average during that month.'
)

MONTH_ORDER = list(calendar.month_abbr[1:])


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _fetch_price_for_stock(symbol, start_from='2015-01-01'):
    """Download 10 years of daily closes for a single stock. Cached for 6 h."""
    try:
        pdf = hd.get_daily_stock_price(symbol, start_from=start_from)
        if pdf is not None and not pdf.empty and 'close' in pdf.columns:
            return {'symbol': symbol, 'price_df': pdf[['date', 'close']]}
    except Exception:
        pass
    return None


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _compute_agg_seasonality(symbols_tuple, div_cal_df_json):
    """Parallel-fetch prices then compute aggregate monthly seasonality + best-days KDE. Cached 6 h."""
    symbols = list(symbols_tuple)
    div_cal_df = pd.read_json(io.StringIO(div_cal_df_json))
    div_cal_df['date'] = pd.to_datetime(div_cal_df['date'])

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_price_for_stock, s): s for s in symbols}
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            if rec is not None:
                results.append(rec)

    if not results:
        return pd.DataFrame(), pd.DataFrame()

    agg_df = hd.calc_aggregate_seasonality(results)

    # Best-days KDE — for each stock × dividend event, find how many days before
    # ex-date the price was cheapest, then pool across all stocks and events.
    all_best_days = []
    for rec in results:
        sym = rec['symbol']
        sdf_sym = div_cal_df[div_cal_df['symbol'] == sym][['date']].copy()
        sdf_sym['date'] = sdf_sym['date'].dt.strftime('%Y-%m-%d')
        best = hd.calc_pre_ex_best_days(rec['price_df'], sdf_sym, pre_ex_days=180)
        all_best_days.extend(best)

    best_days_df = pd.DataFrame({'days_before': all_best_days}) if all_best_days else pd.DataFrame()
    return agg_df, best_days_df


# Only compute when the user explicitly asks — price fetch can be slow for many stocks
_all_symbols = sorted(stats_df['symbol'].unique().tolist()) if not stats_df.empty else []
_n_stocks = len(_all_symbols)

if _n_stocks == 0:
    st.info('No stocks available for seasonality analysis.')
else:
    _col_info, _col_btn = st.columns([3, 1])
    _col_info.caption(
        f'Will fetch 10+ years of price history for up to **{_n_stocks} stocks**. '
        'This may take 30–60 seconds on first load; results are cached for 6 hours.'
    )

    if _col_btn.button('🔍 Calculate Seasonality', width='stretch'):
        st.session_state['cal_show_seasonality'] = True

    if st.session_state.get('cal_show_seasonality'):
        with st.spinner(f'Fetching price data for {_n_stocks} stocks…'):
            _symbols_tuple = tuple(_all_symbols)
            # Pass all available dividend events so we can match each stock's sdf
            _div_cal_json = df[['symbol', 'date']].to_json()
            _agg_df, _best_days_df = _compute_agg_seasonality(_symbols_tuple, _div_cal_json)

        if _agg_df.empty:
            st.warning('Could not compute seasonality — price data unavailable.')
        else:
            _chart_cols = st.columns(2)

            # ---------------------------------------------------------------- #
            # Chart A — Aggregate monthly bar + IQR band                       #
            # ---------------------------------------------------------------- #
            with _chart_cols[0]:
                st.markdown('#### Aggregate Monthly Relative Price')
                st.caption('Median ± IQR across all stocks. Green highlight = historically cheapest month.')

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
                    f'🏆 **{_best_month_name}** is historically the cheapest month across these dividend stocks '
                    f'(avg relative price: **{_best_val:.1f}%** of annual mean)'
                )

            # ---------------------------------------------------------------- #
            # Chart B — KDE of best days before ex-date                        #
            # ---------------------------------------------------------------- #
            with _chart_cols[1]:
                st.markdown('#### Distribution: Days Before Ex-Date to Buy')
                st.caption(
                    'For each historical dividend event across all stocks, shows how many '
                    'calendar days before ex-date the price hit its lowest within a 180-day window.'
                )

                if not _best_days_df.empty:
                    _median_days = int(_best_days_df['days_before'].median())
                    _mean_days = int(_best_days_df['days_before'].mean())

                    _kde_chart = alt.Chart(_best_days_df).transform_density(
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

                    _n_events = len(_best_days_df)
                    _p25 = int(_best_days_df['days_before'].quantile(0.25))
                    _p75 = int(_best_days_df['days_before'].quantile(0.75))
                    st.success(
                        f'🎯 Buy **{_median_days} days** before ex-date (median across {_n_events} events). '
                        f'Middle 50% range: **{_p25}–{_p75} days** before.'
                    )
                else:
                    st.info('No pre-ex timing data available for these stocks.')

