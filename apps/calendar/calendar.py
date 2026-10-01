import os
import io
import json
import calendar
from datetime import datetime

import redis
import holidays
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


import harvest.plot as hp
from harvest import chart_style as cs
from harvest.utils import setup_logging


current_year = datetime.today().year
current_month = datetime.today().month

st.set_page_config(page_title='Dividend Calendar - Panen Dividen', page_icon=':material/calendar_month:')
hp.enable_chart_theme()


# Title and warning go first; they depend on the year picked in the controls below them
header = st.container()
# All options on one row: stock list, calendar view, year, month and the Syariah filter
controls = st.container(horizontal=True, vertical_alignment='bottom', gap='medium')

sl = controls.segmented_control('Stock list', ['JKSE', 'S&P500'], default='JKSE', required=True)

if sl == 'JKSE':
    exch = 'jkse'
    market = 'ID'
else:
    exch = 'sp500'
    market = 'US'

div_years_key = f'div_cal_years_{exch}'
div_score_key = f'div_score_{exch}'

url = os.environ['REDIS_URL']
SITE_URL = os.getenv('SITE_URL', 'https://panendividen.com').rstrip('/')

logger = setup_logging('calendar')


@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url, socket_connect_timeout=10, socket_timeout=30, socket_keepalive=True, retry_on_timeout=True)
    return r

r = connect_redis(url)


class MissingDataError(Exception):
    """A Redis key the page needs is not there yet (e.g. the pipeline has not run)."""


@st.cache_data(max_entries=4, ttl=60*60, show_spinner='Downloading dividend data')
def get_data_from_redis(key):
    # Raising instead of returning None keeps a missing key out of the cache,
    # so the page recovers as soon as the pipeline writes it
    j = r.get(key)
    if j is None:
        raise MissingDataError(key)

    if isinstance(j, bytes) and j.startswith(b'PAR1'):
        return pd.read_parquet(io.BytesIO(j))

    rjson = json.loads(j)
    if 'date' in rjson and 'content' in rjson:
        return pd.DataFrame(json.loads(rjson['content']))
    return pd.DataFrame(rjson)


def load_data(key):
    try:
        return get_data_from_redis(key)
    except MissingDataError:
        logger.error(f'Missing redis key: {key}')
        st.error('Dividend calendar data is not available. Please refresh after the pipeline finishes.')
        st.stop()


def prep_div_table(df, month, market='ID'):
    """One month's dividends ranked by yield.

    Yield and ex-date stay numeric/datetime so the table sorts them correctly.
    """
    table = df[df['date'].dt.month == month].sort_values('yield', ascending=False).reset_index(drop=True)
    table['rank'] = range(1, len(table) + 1)
    table['url_link'] = table['symbol'].map(lambda x: f'{SITE_URL}/stock_picker?market={market}&stock={x}')
    table = table.rename(columns={'date': 'ex_date', 'yield': 'div_yield', 'adjDividend': 'dividend'})
    return table[['rank', 'url_link', 'ex_date', 'div_yield', 'dividend', 'price']]


def format_dividend(value):
    """Dividend per share without trailing zeros: 350, 0.24, 0.2635."""
    return f'{value:,.4f}'.rstrip('0').rstrip('.')


def market_holidays(market, years):
    """{date: holiday name} for days the exchange is closed: IDX (market 'ID') or NYSE (market 'US')."""
    if market == 'US':
        closed = dict(holidays.financial_holidays('NYSE', years=years))
    else:
        # IDX closes on public holidays, cuti bersama (government category) and the 31 Dec bourse holiday
        closed = dict(holidays.Indonesia(years=years, categories=('public', 'government'), language='en_US'))
        for y in years:
            closed.setdefault(pd.Timestamp(year=y, month=12, day=31).date(), 'IDX Year-End Holiday')
    return dict(sorted(closed.items()))


def project_to_next_year(dates, market='ID'):
    """Shift ex-dates one year forward so the calendar grid uses the projected year's weekdays.

    DateOffset maps 29 Feb onto 28 Feb instead of raising like Timestamp.replace.
    Ex-dates are trading days, so a projection landing on a weekend or exchange holiday
    moves to the next trading day, or back to the previous one if that would leave the month.
    """
    projected = pd.Series(pd.to_datetime(dates)) + pd.DateOffset(years=1)
    if projected.empty:
        return projected

    years = sorted(projected.dt.year.unique().tolist())
    closed = np.array(list(market_holidays(market, years)), dtype='datetime64[D]')
    days = projected.values.astype('datetime64[D]')

    forward = np.busday_offset(days, 0, roll='forward', holidays=closed)
    backward = np.busday_offset(days, 0, roll='backward', holidays=closed)

    rolled = pd.Series(pd.to_datetime(forward), index=projected.index)
    leaves_month = rolled.dt.month != projected.dt.month
    rolled[leaves_month] = pd.to_datetime(backward[leaves_month.to_numpy()])
    return rolled


def add_yield(df):
    """Add yield (%) and drop rows that are not a real payout or have no usable price.

    A zero or missing price gives an inf/NaN yield that skews the averages and breaks idxmax;
    a zero dividend (e.g. a non-payer with a 0.0 row) is not a payout.
    """
    df = df.copy()
    adj = pd.to_numeric(df['adjDividend'], errors='coerce')
    price = pd.to_numeric(df['price'], errors='coerce')
    valid = (adj > 0) & (price > 0)

    df['yield'] = (adj / price.where(valid)) * 100
    valid &= np.isfinite(df['yield'])

    dropped = int((~valid).sum())
    if dropped:
        logger.warning(f'Dropped {dropped} dividend rows without a positive dividend and price: '
                       f'{sorted(df.loc[~valid, "symbol"].astype(str).unique().tolist())}')
    return df[valid].reset_index(drop=True)


def syariah_symbols(div_score_df):
    """Set of Syariah-compliant symbols, or None when the score data does not carry the flag."""
    if not {'symbol', 'is_syariah'}.issubset(div_score_df.columns):
        return None
    return set(div_score_df.loc[div_score_df['is_syariah'].eq(True), 'symbol'])


PAYOUT_FREQUENCIES = [
    # (label, min payments, max payments)
    ('Once per year', 1, 1),
    ('Twice per year', 2, 2),
    ('3 times per year', 3, 3),
    ('4 or more times per year', 4, None),
]


def payout_frequency_label(payments):
    for label, low, high in PAYOUT_FREQUENCIES:
        if payments >= low and (high is None or payments <= high):
            return label
    return None


def payout_frequency(df):
    """Companies per payout-frequency bucket, with their tickers; empty buckets are left out."""
    columns = ['frequency', 'count', 'stocks']
    if df.empty:
        return pd.DataFrame(columns=columns)

    # Count events, not unique dates: two ex-dates rolled onto the same trading day are still two payouts
    events = df.drop_duplicates(subset=['symbol', 'date', 'adjDividend'])
    freq_df = events.groupby('symbol').size().reset_index(name='payments')
    freq_df['frequency'] = freq_df['payments'].map(payout_frequency_label)

    order = [label for label, _, _ in PAYOUT_FREQUENCIES]
    summary = (
        freq_df.groupby('frequency')
        .agg(count=('symbol', 'size'), stocks=('symbol', lambda s: ', '.join(sorted(s))))
        .reindex(order)
        .dropna(subset=['count'])
        .reset_index()
    )
    summary['count'] = summary['count'].astype(int)
    return summary[columns]


def parse_query_params(qp, available_years, default_year, current_month):
    """Widget defaults from the URL: (view index, year index, month index), falling back on bad values.

    Supported params:
      ?view=yearly|monthly   (default: yearly)
      ?year=2025             (source year; defaults to latest completed year)
      ?month=3               (1-12, only relevant when view=monthly)
    Example: ?view=monthly&year=2025&month=6
    """
    view_idx = 1 if str(qp.get('view', 'yearly')).lower() == 'monthly' else 0

    try:
        year = int(qp.get('year', default_year))
    except ValueError:
        year = default_year
    year_idx = available_years.index(year if year in available_years else default_year)

    try:
        month = int(qp.get('month', current_month))
    except ValueError:
        month = current_month
    month_idx = (month if 1 <= month <= 12 else current_month) - 1

    return view_idx, year_idx, month_idx


def yield_summary(stats_df, by):
    """Events and average yield per month ('month') or per day of month ('day')."""
    key = stats_df['date'].dt.month if by == 'month' else stats_df['date'].dt.day
    summary = (
        stats_df
        .assign(**{by: key})
        .groupby(by)['yield']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'avg_yield', 'count': 'events'})
    )
    if by == 'month':
        summary['month_name'] = summary['month'].map(lambda m: calendar.month_abbr[m])
    return summary


def yield_combo_chart(summary, x_field, x_title, sort=None):
    """Dividend count bars with the average yield line on a second axis."""
    x = alt.X(x_field, sort=sort, title=x_title)
    x_tooltip = alt.Tooltip(x_field, title=x_title)
    base = alt.Chart(summary)
    events_bar = base.mark_bar(color=cs.CHART_PRIMARY).encode(
        x=x,
        y=alt.Y('events:Q', title='Dividend count'),
        tooltip=[x_tooltip, alt.Tooltip('events:Q', title='Count')]
    )
    yield_line = base.mark_line(color=cs.CHART_AMBER, point=True).encode(
        x=x,
        y=alt.Y('avg_yield:Q', title='Avg yield (%)', axis=alt.Axis(titleColor=cs.CHART_AMBER), scale=alt.Scale(zero=False)),
        tooltip=[x_tooltip, alt.Tooltip('avg_yield:Q', format='.2f', title='Avg yield %')]
    )
    return (events_bar + yield_line).resolve_scale(y='independent').properties(height=260)


def yield_histogram(stats_df):
    """How many dividends fall in each yield range, with the average marked."""
    bars = alt.Chart(stats_df[['yield']]).mark_bar(color=cs.CHART_PRIMARY, cornerRadiusEnd=2).encode(
        x=alt.X('yield:Q', bin=alt.Bin(maxbins=30), title='Dividend yield (%)'),
        y=alt.Y('count():Q', title='Dividend count'),
        tooltip=[
            alt.Tooltip('yield:Q', bin=alt.Bin(maxbins=30), title='Yield range (%)'),
            alt.Tooltip('count():Q', title='Count'),
        ],
    )
    mean_rule = alt.Chart(pd.DataFrame({'mean_yield': [stats_df['yield'].mean()]})).mark_rule(
        color=cs.CHART_AMBER, strokeWidth=2, strokeDash=[5, 3]
    ).encode(
        x='mean_yield:Q',
        tooltip=[alt.Tooltip('mean_yield:Q', format='.2f', title='Average yield (%)')]
    )
    return (bars + mean_rule).properties(height=260)


# Full-year calendar month size limits in pixels, the same as the live site before the redesign.
# lesley 0.4.0 draws square date cells of min(width / 7, height / 6) = 33px (30px squares after
# padding). On a 13" MacBook (1440x800 window, sidebar open) the image is ~975px wide, inside the
# 980px content area, and once scrolled the 4x3 grid fits below the 60px Streamlit header;
# the title and color key row sits just under it
CALENDAR_MONTH_WIDTH = 250
CALENDAR_MONTH_HEIGHT = 200

# Single Month view: a 466px-wide calendar (7 square cells of 466 / 7 = 66.6px, ~474px with padding)
# and the month's dividend table beside it at the same width
SINGLE_MONTH_WIDTH = 466
SINGLE_MONTH_HEIGHT = 400
SINGLE_MONTH_COLUMN_WIDTH = 474

# Ticker after stock=, without the .JK suffix; allows class shares like BRK.B or BF-B
TICKER_DISPLAY_PATTERN = r"[?&]stock=([A-Z0-9.\-]+?)(?:\.JK)?$"

column_config = {
    'url_link': st.column_config.LinkColumn(
        'Stock',
        help='The code of the stock',
        max_chars=10,

        display_text=TICKER_DISPLAY_PATTERN
    ),
    'rank': st.column_config.NumberColumn(
        'Rank',
        help='Rank of the stock based on yield for this month',
        format='%d',
    ),
    'ex_date': st.column_config.DateColumn(
        'Ex date',
        help='Projected ex-dividend date',
        format='DD MMM',
    ),
    'div_yield': st.column_config.NumberColumn(
        'Yield',
        help='Dividend as a percentage of the price',
        format='%.2f%%',
    ),
    # No fixed format: Streamlit's default shows up to 4 decimals with trailing zeros trimmed,
    # so US cents (0.24, 0.2635) and split-adjusted IDR amounts (0.009) keep their precision
    'dividend': st.column_config.NumberColumn(
        'Dividend',
        help='Dividend amount per share',
    ),
    'price': st.column_config.NumberColumn(
        'Price',
        help='Stock price',
        format='localized'
    ),
}


years_df = load_data(div_years_key)
available_years = sorted(years_df['year'].astype(int).unique().tolist())
if len(available_years) == 0:
    available_years = [current_year]
completed_years = [year for year in available_years if year < current_year]
default_year = max(completed_years) if completed_years else max(available_years)

_default_view_idx, _default_year_idx, _default_month_idx = parse_query_params(
    st.query_params, available_years, default_year, current_month
)
months = list(calendar.month_name)  # ['', 'January', ..., 'December']

view_options = ['Full Year', 'Single Month']
view_control = controls.segmented_control(
    'Calendar view', view_options, default=view_options[_default_view_idx], required=True
)

selected_year = controls.selectbox('Based on year', available_years, index=_default_year_idx, width=140)

month_index = None
if view_control == 'Single Month':
    select_month = controls.selectbox('Month', months[1:], index=_default_month_idx, width=160)
    month_index = months.index(select_month)

# Keep URL params in sync with the current widget state
if month_index is not None:
    st.query_params.update({'view': 'monthly', 'year': str(selected_year), 'month': str(month_index)})
else:
    st.query_params.update({'view': 'yearly', 'year': str(selected_year)})
    st.query_params.pop('month', None)

projected_year = selected_year + 1
projection_note = (
    f'Projected from {selected_year} ex-dividend dates: each date is moved to the same day in {projected_year}, '
    'or to the next trading day when that is a weekend or exchange holiday.'
)
with header:
    st.title(f'Dividend Calendar {projected_year}')
    if selected_year >= current_year:
        st.warning(f'{selected_year} is not over yet, so this projection only includes ex-dividend dates up to today.')

div_cal_key = f'div_cal_{exch}_{selected_year}'
df = load_data(div_cal_key)
# The page projects next year's calendar from this year's ex-dates
df['date'] = project_to_next_year(df['date'], market=market)
# Exchange holidays in the projected year, marked red on the calendar like weekends with the name on hover
closed_dates = market_holidays(market, [projected_year])
df = add_yield(df)

div_score_df = load_data(div_score_key)

if sl == 'JKSE':
    _syariah = syariah_symbols(div_score_df)
    if _syariah is None:
        logger.warning(f'{div_score_key} has no symbol/is_syariah columns; Syariah filter hidden')
    elif controls.toggle('Syariah only'):
        # Filter by membership rather than merging, so duplicate score rows cannot duplicate dividend events
        df = df[df['symbol'].isin(_syariah)]

if df.empty:
    # The calendar charts cannot be drawn without at least one dividend date
    st.info('No dividend data available for this selection.')
    st.stop()


COLOR_NOTE = ('Darker green means a higher dividend yield. '
              ':red[Red dates] are weekends and exchange holidays; hover a holiday for its name.')


if view_control == 'Full Year':
    # One fixed 4x3 image, color key and title underneath, so it can be screenshotted and shared as is.
    # Sized to fill the content width beside the open sidebar on a 1440px-wide (13" MacBook) window
    # while the whole grid still fits in one screen height
    st.altair_chart(
        hp.plot_dividend_calendar(df, sl=sl, closed_dates=closed_dates, width=CALENDAR_MONTH_WIDTH,
                                  height=CALENDAR_MONTH_HEIGHT,
                                  title=f'Dividend Calendar {projected_year} · {sl}', legend=True),
        width='content', theme=None,
    )
    st.caption(f'{COLOR_NOTE} {projection_note}')

    # Month tables in the same 4x3 order as the calendar, each ranked by yield
    st.subheader('Dividends per month')
    for first_month in (1, 5, 9):
        row = st.columns(4)
        for col, month in zip(row, range(first_month, first_month + 4)):
            with col:
                st.markdown(f'**{calendar.month_name[month]}**')
                month_table = prep_div_table(df, month, market=market)
                if month_table.empty:
                    st.caption('No ex-dividend dates.')
                else:
                    st.dataframe(month_table, hide_index=True, width='stretch', height=213, column_config=column_config)

else:
    # Calendar at its own width with the table right next to it (small gap); on narrow screens the
    # table wraps below the calendar
    month_row = st.container(horizontal=True, gap='small')
    with month_row.container(width=SINGLE_MONTH_COLUMN_WIDTH):
        st.altair_chart(
            hp.plot_dividend_month(df, month_index, sl=sl, closed_dates=closed_dates, title=select_month,
                                   width=SINGLE_MONTH_WIDTH, height=SINGLE_MONTH_HEIGHT),
            width='content', theme=None,
        )
        st.altair_chart(hp.plot_dividend_calendar_legend(sl), width='content', theme=None)
        st.caption(f'{COLOR_NOTE} {projection_note}')
    with month_row.container(width=SINGLE_MONTH_COLUMN_WIDTH):
        month_table = prep_div_table(df, month_index, market=market)
        # Default height shows ten rows; the rest of the month scrolls inside the table
        st.dataframe(month_table, hide_index=True, width='stretch', column_config=column_config)
        if len(month_table) > 10:
            st.caption(f'{len(month_table)} dividends this month, ranked by yield. Scroll the table to see them all.')

stats_df = df if view_control == 'Full Year' else df[df['date'].dt.month == month_index]
if stats_df.empty:
    # Only a single month can be empty here; the page stops earlier when there is no data at all
    st.info(f'No ex-dividend dates in {select_month}, so there are no statistics for this month.')
else:
    st.subheader('Dividend statistics')

    c1, c2, c3 = st.columns(3)
    c1.metric('Dividends', len(stats_df), border=True)
    c2.metric('Average yield', f"{stats_df['yield'].mean():.2f}%", border=True)
    c3.metric('Median yield', f"{stats_df['yield'].median():.2f}%", border=True)

    top_row = stats_df.loc[stats_df['yield'].idxmax()]
    st.caption(
        f"Highest yield: {top_row['symbol']} on {top_row['date'].strftime('%d %b')} at {top_row['yield']:.2f}% "
        f"(dividend {format_dividend(top_row['adjDividend'])} per share)."
    )

    charts = st.columns(2, gap='medium')

    if view_control == 'Full Year':
        combo = yield_combo_chart(yield_summary(stats_df, 'month'), 'month_name:N', 'Month', sort=list(calendar.month_abbr[1:]))
        combo_title = 'Dividends and average yield per month'
    else:
        combo = yield_combo_chart(yield_summary(stats_df, 'day'), 'day:O', 'Ex-date day')
        combo_title = 'Dividends and average yield per day'
    with charts[0]:
        st.markdown(f'**{combo_title}**')
        st.altair_chart(combo, width='stretch', theme=None)
    with charts[1]:
        st.markdown('**Yield distribution**')
        st.altair_chart(yield_histogram(stats_df), width='stretch', theme=None)
        st.caption(':orange[Dashed line]: average yield.')

# Frequency covers the whole year, so it shows even when the selected month has no dividends
freq_summary = payout_frequency(df)

st.subheader('Payout frequency per year')
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
