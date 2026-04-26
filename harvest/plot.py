import calendar
from datetime import datetime

import lesley
import numpy as np
import pandas as pd
import altair as alt
from streamlit_echarts5 import JsCode


def format_currency():
    return  "datum.value >= 1000000000000 ? format(datum.value/1000000000000, ',.0f') + ' T' : " +\
            "datum.value <= -1000000000000 ? '-' + format(-datum.value/1000000000000, ',.0f') + ' T' : " +\
            "datum.value >= 1000000000 ? format(datum.value/1000000000, ',.0f') + ' B' : " +\
            "datum.value <= -1000000000 ? '-' + format(-datum.value/1000000000, ',.0f') + ' B' : " +\
            "datum.value >= 0 ? format(datum.value/1000000, ',.0f') + ' M' : " +\
            "'-' + format(-datum.value/1000000, ',.0f') + ' M'"


def format_tooltip_currency(val, currency):
        
        is_negative = val < 0
        abs_val = abs(val)
        
        if abs_val >= 1_000_000_000_000:
            formatted = f"{abs_val/1_000_000_000_000:.2f} T {currency.upper()}"
        elif abs_val >= 1_000_000_000:
            formatted = f"{abs_val/1_000_000_000:.2f} B {currency.upper()}"
        elif abs_val >= 1_000_000:
            formatted = f"{abs_val/1_000_000:.2f} M {currency.upper()}"
        else:
            formatted = f"{abs_val/1_000:.2f} K {currency.upper()}"
        
        return f"-{formatted}" if is_negative else formatted

def format_tooltip_currency_expr(currency):
    return  "datum.value >= 1000000000000 ? format(datum.value/1000000000000, ',.2f') + ' T " + currency.upper() + "' : " +\
            "datum.value <= -1000000000000 ? '-' + format(-datum.value/1000000000000, ',.2f') + ' T " + currency.upper() + "' : " +\
            "datum.value >= 1000000000 ? format(datum.value/1000000000, ',.2f') + ' B " + currency.upper() + "' : " +\
            "datum.value <= -1000000000 ? '-' + format(-datum.value/1000000000, ',.2f') + ' B " + currency.upper() + "' : " +\
            "datum.value >= 1000000 ? format(datum.value/1000000, ',.2f') + ' M " + currency.upper() + "' : " +\
            "datum.value <= -1000000 ? '-' + format(-datum.value/1000000, ',.2f') + ' M " + currency.upper() + "' : " +\
            "datum.value >= 0 ? format(datum.value/1000, ',.2f') + ' K " + currency.upper() + "' : " +\
            "'-' + format(-datum.value/1000, ',.2f') + ' K " + currency.upper() + "'"


def plot_fin_chart(fin_df, currency='idr'):

    fin_df = fin_df.groupby('calendarYear').sum().reset_index()
    fin_df['netProfitMargin'] = fin_df['netIncome'] / fin_df['revenue']
    combined_chart = alt.Chart(fin_df[['calendarYear', 'revenue', 'netIncome']]).transform_fold(
        ['revenue', 'netIncome']
    ).transform_calculate(
        value_fmt=format_tooltip_currency_expr(currency)
    ).mark_bar().encode(
        x='calendarYear:N',
        y=alt.Y('value:Q', axis=alt.Axis(
            labelExpr=format_currency()
        )),
        color='key:N',
        xOffset='key:N',
        tooltip=['calendarYear', alt.Tooltip('value_fmt:N', title='value')]
    )
    margin_chart = alt.Chart(fin_df).mark_line(point=True).encode(
        x='calendarYear:N',
        y=alt.Y('netProfitMargin:Q', axis=alt.Axis(format='%')),
        tooltip=['calendarYear', alt.Tooltip('netProfitMargin:Q', format='.2%')]
    )
    return (combined_chart+margin_chart).resolve_scale(y='independent')


def plot_fin_chart_enhanced(fin_df, currency='idr', height=320):
    """
    Polished annual financial chart:
      - Grouped bars: Revenue (blue) + Net Income (teal), side-by-side, shared y scale
      - Overlay line: Net Profit Margin % on an independent right axis
    """
    df = fin_df.groupby('calendarYear').agg(
        revenue=('revenue', 'sum'),
        netIncome=('netIncome', 'sum'),
    ).reset_index()
    df['netProfitMargin'] = (df['netIncome'] / df['revenue'] * 100).round(2)
    df['year_str']        = df['calendarYear'].astype(str)

    # ── Grouped bars (shared y scale, no overlap) ────────────────────────── #
    bars = alt.Chart(df).transform_fold(
        ['revenue', 'netIncome'],
        as_=['metric', 'value'],
    ).transform_calculate(
        value_fmt=format_tooltip_currency_expr(currency)
    ).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
    ).encode(
        x=alt.X('year_str:O', title='', axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset('metric:N'),
        y=alt.Y('value:Q', title='Value', axis=alt.Axis(labelExpr=format_currency())),
        color=alt.Color('metric:N', scale=alt.Scale(
            domain=['revenue', 'netIncome'],
            range=['#42a5f5', '#26a69a'],
        ), legend=alt.Legend(
            title='',
            labelExpr="datum.label === 'revenue' ? 'Revenue' : 'Net Income'",
        )),
        tooltip=[
            alt.Tooltip('year_str:O',  title='Year'),
            alt.Tooltip('metric:N',    title='Metric'),
            alt.Tooltip('value_fmt:N', title='Value'),
        ]
    )

    # ── Net profit margin overlay line (independent right axis) ──────────── #
    margin_line = alt.Chart(df).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=60),
        color='#ffa726',
        strokeWidth=2.5,
    ).encode(
        x=alt.X('year_str:O'),
        y=alt.Y('netProfitMargin:Q', title='Net Profit Margin (%)',
                axis=alt.Axis(format='.0f', titleColor='#ffa726', labelColor='#ffa726')),
        tooltip=[
            alt.Tooltip('year_str:O',        title='Year'),
            alt.Tooltip('netProfitMargin:Q', title='Net Margin %', format='.1f'),
        ]
    )

    return (
        alt.layer(bars, margin_line)
        .resolve_scale(y='independent')
        .properties(height=height)
    )



def plot_profit_margin_trend(fin_df, currency='idr', height=220):
    """
    Rolling 4-quarter (TTM) net profit margin trend line with coloured area.
    Useful for spotting structural margin expansion or compression.
    """
    df = fin_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    rolling = df[['date', 'revenue', 'netIncome']].rolling(window=4, on='date').sum()
    rolling['date']             = df['date'].values
    rolling['netProfitMargin']  = (rolling['netIncome'] / rolling['revenue'] * 100).clip(-50, 100)
    rolling = rolling.dropna(subset=['netProfitMargin'])

    # Zero reference line
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        strokeDash=[4, 3], color='#aaaaaa', strokeWidth=1
    ).encode(y='y:Q')

    # Median reference
    med = float(rolling['netProfitMargin'].median())
    med_line = alt.Chart(pd.DataFrame({'y': [med]})).mark_rule(
        strokeDash=[5, 3], color='#ffa726', strokeWidth=1.5, opacity=0.8
    ).encode(y='y:Q')

    # Area fill — green above 0, red below
    area = alt.Chart(rolling).mark_area(
        interpolate='monotone',
        opacity=0.25,
        color='#26a69a',
    ).encode(
        x=alt.X('date:T', title=''),
        y=alt.Y('netProfitMargin:Q', title='TTM Net Margin (%)', scale=alt.Scale(zero=False)),
    )

    line = alt.Chart(rolling).mark_line(
        interpolate='monotone',
        color='#26a69a',
        strokeWidth=2.5,
    ).encode(
        x=alt.X('date:T'),
        y=alt.Y('netProfitMargin:Q', scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('date:T',              title='Date'),
            alt.Tooltip('netProfitMargin:Q',   title='TTM Net Margin %', format='.1f'),
        ]
    )

    return alt.layer(area, zero_line, med_line, line).properties(height=height)


def plot_quarterly_breakdown(fin_df, metric='netIncome', currency='idr', height=280):
    """
    Clean quarterly bar chart for a single financial metric.
    Each bar is coloured by QoQ growth (green = improvement, red = decline).
    A thin YoY comparison line is overlaid to show structural trends.
    """
    df = fin_df.copy()
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
    df['period_label'] = df['calendarYear'].astype(str) + ' ' + df['period']
    df['value_fmt']    = df[metric].apply(lambda x: format_tooltip_currency(x, currency))
    df['qoq_growth']   = df[metric].pct_change() * 100
    df['yoy_val']      = df[metric].shift(4)   # same quarter last year

    label_map = {'revenue': 'Revenue', 'netIncome': 'Net Income'}
    title = label_map.get(metric, metric)

    bars = alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=3, cornerRadiusTopRight=3
    ).encode(
        x=alt.X('calendarYear:O', title='Year', axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset('period:O', sort=['Q1', 'Q2', 'Q3', 'Q4']),
        y=alt.Y(f'{metric}:Q', title=title, axis=alt.Axis(labelExpr=format_currency())),
        color=alt.condition(
            alt.datum['qoq_growth'] < 0,
            alt.value('#ef5350'),
            alt.value('#42a5f5') if metric == 'revenue' else alt.value('#26a69a'),
        ),
        tooltip=[
            alt.Tooltip('calendarYear:O', title='Year'),
            alt.Tooltip('period:O',        title='Quarter'),
            alt.Tooltip('value_fmt:N',    title=title),
            alt.Tooltip('qoq_growth:Q',   title='QoQ Growth %', format='.1f'),
        ]
    )

    yoy_line = alt.Chart(df).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=30, opacity=0.6),
        color='#ffa726',
        strokeDash=[4, 2],
        strokeWidth=1.8,
        opacity=0.8,
    ).encode(
        x=alt.X('calendarYear:O'),
        xOffset=alt.XOffset('period:O', sort=['Q1', 'Q2', 'Q3', 'Q4']),
        y=alt.Y('yoy_val:Q'),
        tooltip=[
            alt.Tooltip('calendarYear:O', title='Year'),
            alt.Tooltip('period:O',        title='Quarter'),
            alt.Tooltip('yoy_val:Q',       title='Same Q Last Year', format=',.0f'),
        ]
    )

    return (bars + yoy_line).resolve_scale(y='shared').properties(height=height)


def plot_financial(fin_df, period='quarter', metric='netIncome', currency='idr'):
    
    if period == 'quarter':
        return plot_quarter_income(fin_df, metric, currency)
    else:
        return plot_yearly_income(fin_df, metric, currency)

def plot_yearly_income(fin_df, metric, currency='idr'):

    fin_df = fin_df.groupby('calendarYear').sum().reset_index()
    fin_df['value'] = fin_df[metric].apply(lambda x: format_tooltip_currency(x, currency))
    fin_df['growth'] = fin_df[metric].pct_change() * 100

    chart = alt.Chart(fin_df).mark_bar(
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X('calendarYear'),
        y=alt.Y(f'{metric}:Q', axis=alt.Axis(
            labelExpr=format_currency()
        )),
        color=alt.condition(alt.datum['growth'] < 0, alt.value('#ff796c'), alt.value('#008631')),
        tooltip=['calendarYear', 'value', alt.Tooltip('growth', format='.2f')]
    ).properties(
        height=300
    )
    return chart


def plot_quarter_income(fin_df, metric, currency='idr'):
    
    fin_df['value'] = fin_df[metric].apply(lambda x: format_tooltip_currency(x, currency))

    chart = alt.Chart(fin_df).mark_bar().encode(
        x=alt.X('period'),
        y=alt.Y(metric, axis=alt.Axis(
            labelExpr=format_currency()
        )),
        color='period',
        column='calendarYear',
        tooltip='value'
    ).properties(
        height=300
    )

    return chart


# MA palette: window → (line colour, dash pattern)
_MA_STYLES = {
    20:  ('#f39c12', []),         # amber  – solid
    50:  ('#3498db', []),         # blue   – solid
    200: ('#9b59b6', [4, 3]),     # purple – dashed
}


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, float('nan'))
    return 100 - 100 / (1 + rs)


def plot_candlestick(
    price_df,
    width: int = 1000,
    height: int = 300,
    ma_windows: list | None = None,   # e.g. [20, 50, 200]
    show_rsi: bool = False,
):
    """
    Interactive candlestick chart with optional MA overlays and RSI panel.

    Parameters
    ----------
    price_df   : DataFrame with columns [date, open, high, low, close, volume]
    ma_windows : list of integers – moving-average windows to overlay (e.g. [20, 50, 200])
    show_rsi   : bool – append an RSI(14) panel below the volume bar
    """
    df = price_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # ── Compute MAs & RSI on the full series so brush filtering shows correct lines ── #
    if ma_windows:
        for w in ma_windows:
            df[f'ma{w}'] = df['close'].rolling(w, min_periods=max(1, w // 2)).mean()

    if show_rsi:
        df['rsi'] = _compute_rsi(df['close'])

    open_close_color = alt.condition(
        'datum.open <= datum.close',
        alt.value('#06982d'),
        alt.value('#ae1325'),
    )

    today        = datetime.today()
    one_year_ago = today.replace(year=today.year - 1)
    x_init = (
        pd.to_datetime([today.strftime('%Y-%m-%d'), one_year_ago.strftime('%Y-%m-%d')])
        .astype(int) / 1e6
    )
    interval = alt.selection_interval(encodings=['x'], value={'x': list(x_init)})

    # ── Candlestick layers ──────────────────────────────────────────────── #
    base = alt.Chart(df).encode(
        alt.X('date:T', title='', scale=alt.Scale(domain=interval)),
        color=open_close_color,
    )

    rule = base.mark_rule().encode(
        alt.Y('low:Q',  title='Price', scale=alt.Scale(zero=False)),
        alt.Y2('high:Q'),
        tooltip=[
            alt.Tooltip('date:T',  title='Date'),
            alt.Tooltip('open:Q',  title='Open',  format=',.2f'),
            alt.Tooltip('high:Q',  title='High',  format=',.2f'),
            alt.Tooltip('low:Q',   title='Low',   format=',.2f'),
            alt.Tooltip('close:Q', title='Close', format=',.2f'),
        ],
    )

    bar = base.mark_bar().encode(
        alt.Y('open:Q',  scale=alt.Scale(zero=False, padding=10)),
        alt.Y2('close:Q'),
    )

    layers = [rule, bar]

    # ── Moving-average overlays ─────────────────────────────────────────── #
    for w in (ma_windows or []):
        style = _MA_STYLES.get(w, ('#aaaaaa', []))
        col, dash = style
        ma_line = alt.Chart(df).mark_line(
            color=col,
            strokeWidth=1.5,
            strokeDash=dash,
            opacity=0.85,
        ).encode(
            x=alt.X('date:T', scale=alt.Scale(domain=interval)),
            y=alt.Y(f'ma{w}:Q', scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip('date:T',       title='Date'),
                alt.Tooltip(f'ma{w}:Q',     title=f'MA{w}', format=',.2f'),
            ],
        ).transform_filter(interval)
        layers.append(ma_line)

    candlestick = alt.layer(*layers).properties(width=width, height=height).transform_filter(interval)

    # ── Volume navigator bar ────────────────────────────────────────────── #
    volume_bar = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X('date:T'),
            y=alt.Y('volume:Q', title='Volume', axis=alt.Axis(labelFontSize=9, format='~s')),
            color=alt.condition(interval, alt.value('#007FFF'), alt.value('lightgrey')),
            tooltip=[
                alt.Tooltip('date:T',   title='Date'),
                alt.Tooltip('volume:Q', title='Volume', format=',.0f'),
            ],
        )
        .add_params(interval)
        .properties(width=width, height=50)
    )

    chart = candlestick & volume_bar

    # ── Optional RSI panel ──────────────────────────────────────────────── #
    if show_rsi:
        # Overbought / oversold reference bands
        ob_band = alt.Chart(pd.DataFrame({'y1': [70], 'y2': [100]})).mark_rect(
            color='#e74c3c', opacity=0.08
        ).encode(y='y1:Q', y2='y2:Q')
        os_band = alt.Chart(pd.DataFrame({'y1': [0], 'y2': [30]})).mark_rect(
            color='#27ae60', opacity=0.08
        ).encode(y='y1:Q', y2='y2:Q')
        ob_rule = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(
            strokeDash=[4, 3], color='#e74c3c', strokeWidth=1, opacity=0.7
        ).encode(y='y:Q')
        os_rule = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(
            strokeDash=[4, 3], color='#27ae60', strokeWidth=1, opacity=0.7
        ).encode(y='y:Q')
        mid_rule = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(
            strokeDash=[2, 4], color='#aaaaaa', strokeWidth=1, opacity=0.5
        ).encode(y='y:Q')

        rsi_line = (
            alt.Chart(df)
            .mark_line(color='#8e44ad', strokeWidth=2)
            .encode(
                x=alt.X('date:T', scale=alt.Scale(domain=interval)),
                y=alt.Y('rsi:Q', title='RSI(14)', scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip('date:T', title='Date'),
                    alt.Tooltip('rsi:Q',  title='RSI',  format='.1f'),
                ],
            )
            .transform_filter(interval)
        )

        rsi_panel = (
            alt.layer(ob_band, os_band, ob_rule, os_rule, mid_rule, rsi_line)
            .properties(width=width, height=90, title='')
        )
        chart = chart & rsi_panel

    return chart


def plot_pe_distribution(df, pe, axis_label=None):

    kde = alt.Chart(df).transform_density('pe', as_=['PE', 'DENSITY'])
    pes_dist = kde.mark_area(
        line={'color': 'darkgreen'},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                alt.GradientStop(color='darkgreen', offset=1)],
            x1=1,
            x2=1,
            y1=1,
            y2=0
        ),
    ).encode(
        x=alt.X('PE:Q', title=axis_label),
        y=alt.Y('DENSITY:Q', title='', axis=alt.Axis(tickSize=0, domain=False, labelAngle=0, labelFontSize=0)),
        tooltip=(
            alt.Tooltip('PE:Q', format='.2f', title=axis_label),
        )
    )
    x_zero = kde.mark_rule().encode(
        x=alt.datum(pe),
        color=alt.value('red'),
        size=alt.value(2),
        tooltip=alt.Tooltip(format='.2f', title='Current val')
    )
   
    return pes_dist+x_zero


def plot_pe_timeseries(pe_df, axis_label=None):

    # Compute median & band reference lines
    valid = pe_df['pe'].replace([float('inf'), -float('inf')], float('nan')).dropna()
    median_pe = valid.median()
    p10 = valid.quantile(0.10)
    p90 = valid.quantile(0.90)

    base = alt.Chart(pe_df)

    # Shaded overvalued / undervalued regions
    p90_rule = alt.Chart(pd.DataFrame({'y': [p90]})).mark_rule(
        strokeDash=[4, 3], color='#e74c3c', strokeWidth=1, opacity=0.7
    ).encode(y='y:Q')
    p10_rule = alt.Chart(pd.DataFrame({'y': [p10]})).mark_rule(
        strokeDash=[4, 3], color='#27ae60', strokeWidth=1, opacity=0.7
    ).encode(y='y:Q')
    med_rule = alt.Chart(pd.DataFrame({'y': [median_pe]})).mark_rule(
        color='#f39c12', strokeWidth=1.5, opacity=0.9
    ).encode(y='y:Q')

    line = base.mark_line(color='#3498db', strokeWidth=2).encode(
        x='date:T',
        y=alt.Y('pe', title=axis_label).scale(zero=False),
        tooltip=(
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('pe:Q', format='.2f', title=axis_label)
        )
    )
    return (line + med_rule + p10_rule + p90_rule).properties(height=300)


def plot_price_vs_fair_value(pe_df, ratio_label='P/E'):
    """
    Overlay actual price with median-multiple-implied fair value and a 10th–90th
    percentile confidence band.  Areas where price < fair value are "cheap zones".

    Parameters
    ----------
    pe_df : DataFrame with columns ['date', 'close', 'pe'] plus the per-share metric column.
    ratio_label : str  – axis / tooltip label ('P/E' or 'P/S').
    """
    df = pe_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    valid_pe = df['pe'].replace([float('inf'), -float('inf')], float('nan')).dropna()
    if len(valid_pe) < 4:
        return None

    median_pe = float(valid_pe.median())
    p10_pe    = float(valid_pe.quantile(0.10))
    p90_pe    = float(valid_pe.quantile(0.90))

    # Detect per-share metric column (everything except date, close, pe)
    metric_col = [c for c in df.columns if c not in ('date', 'close', 'pe')]
    if not metric_col:
        return None
    metric_col = metric_col[0]

    df = df.dropna(subset=[metric_col])
    df['fair_value_median'] = median_pe * df[metric_col]
    df['fair_value_p10']    = p10_pe    * df[metric_col]
    df['fair_value_p90']    = p90_pe    * df[metric_col]

    base = alt.Chart(df)

    # Confidence band (p10–p90 multiple applied to current metric)
    band = base.mark_area(opacity=0.15, color='#f39c12').encode(
        x=alt.X('date:T', title=''),
        y=alt.Y('fair_value_p10:Q', title='Price', scale=alt.Scale(zero=False)),
        y2=alt.Y2('fair_value_p90:Q'),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('fair_value_p10:Q', format=',.0f', title=f'Fair Value ({ratio_label}=p10)'),
            alt.Tooltip('fair_value_p90:Q', format=',.0f', title=f'Fair Value ({ratio_label}=p90)'),
        ]
    )

    # Median fair-value line
    fair_line = base.mark_line(strokeDash=[4, 3], color='#f39c12', strokeWidth=2).encode(
        x='date:T',
        y=alt.Y('fair_value_median:Q', scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('fair_value_median:Q', format=',.0f', title='Fair Value (median)'),
        ]
    )

    # Actual price line
    price_line = base.mark_line(color='#2980b9', strokeWidth=2.5).encode(
        x='date:T',
        y=alt.Y('close:Q', scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('date:T', title='Date'),
            alt.Tooltip('close:Q', format=',.0f', title='Actual Price'),
        ]
    )

    return (band + fair_line + price_line).resolve_scale(y='shared').properties(height=260)


def plot_dividend_history(div_df, extrapolote=False, n_future_years=0, last_val=0, inc_val=0, eps_df=None):

    # aggregate to yearly basis for stock that paid interim during the year
    dividend_year_df = div_df.copy()
    dividend_year_df['year'] = div_df['date'].apply(lambda x: int(x.split('-')[0]))
    yearly = dividend_year_df.groupby('year')['adjDividend'].sum().to_frame().reset_index()

    # fill in the blank for the year when they do not pay dividend
    start_year = yearly.loc[0, 'year']
    end_year = max(yearly.loc[len(yearly)-1, 'year'], datetime.today().year-1)

    years = list(range(start_year, end_year + 1))
    df_temp = pd.DataFrame({'year': years, 'value': [0]*len(years)})
    df_train = pd.merge(df_temp, yearly, on='year', how='left')
    df_train = df_train.fillna(0)
    
    # plot the dividend green if it is increased from last year, and red if it is decreased
    df_train['inc'] = df_train['adjDividend'].shift(1) - df_train['adjDividend']
    div_bar = alt.Chart(df_train).mark_bar(
        cornerRadiusTopLeft=5, 
        cornerRadiusTopRight=5
    ).encode(
        alt.X('year:N'),
        alt.Y('adjDividend'),
        color=alt.condition(alt.datum['inc'] > 0, alt.value('#ff796c'), alt.value('#008631')),
        tooltip=['year', alt.Tooltip('adjDividend', format='.2f')]
    ).properties(
        height=450,
        width=600
    )

    if extrapolote:
        ext_years = list(range(end_year+1, end_year+1+n_future_years))
        ext_values = [last_val+(i+1)*inc_val for i in range(n_future_years)]

        if inc_val > 0:
            ext_color = alt.value('#008631')
        else:
            ext_color = alt.value('#ff796c')
        ext_df = pd.DataFrame({'year': ext_years, 'adjDividend': ext_values})
        div_bar2 = alt.Chart(ext_df).mark_bar(
            cornerRadiusTopLeft=5, 
            cornerRadiusTopRight=5
        ).encode(
            alt.X('year:N'),
            alt.Y('adjDividend:Q'),
            tooltip=['year', alt.Tooltip('adjDividend', format='.2f')],
            opacity=alt.value(0.4), 
            color=ext_color
        ).properties(
            height=450,
            width=600
        )

        div_bar = div_bar + div_bar2

    return div_bar


def plot_labels(price_df, label_df):
    
    price_chart = alt.Chart(price_df).mark_line().encode(
        x='date:T',
        y=alt.Y('close').scale(zero=False)
    )

    buy_chart = alt.Chart(label_df).mark_rule().encode(
        x='date:T',
        color=alt.value('green')
    )
    sell_chart = alt.Chart(label_df).mark_rule().encode(
        x='sell_date:T',
        color=alt.value('red')
    )

    return (price_chart + buy_chart + sell_chart).properties(
        height=450,
        width=1000
    ).interactive()


def plot_rsi(rsi_df):

    chart = alt.Chart(rsi_df).mark_line().encode(
        x='date:T',
        y='rsi'
    ).properties(
        height=100,
        width=1000
    ).interactive()

    return chart


def plot_bbands(bband_df):
    
    cols = ('BBL', 'BBM', 'BBU')
    bband_cols = [x for x in bband_df.columns.values.tolist() if x.startswith(cols)]

    m = bband_df[['date']+bband_cols].melt('date')
    c = alt.Chart(m).mark_line().encode(
        x='date:T',
        y='value',
        color='variable'
    ).properties(
        height=450,
        width=1000
    ).interactive()

    return c


def plot_supertrend(st_df):

    cols = ('SUPERTl', 'SUPERTs')
    bband_cols = [x for x in st_df.columns.values.tolist() if x.startswith(cols)]

    m = st_df[['date']+bband_cols].melt('date')
    c = alt.Chart(m).mark_line().encode(
        x='date:T',
        y='value',
        color=alt.Color('variable').scale(range=('green', 'red'))
    ).properties(
        height=450,
        width=1000
    ).interactive()

    return c


def plot_dividend_calendar(div_df, show_next_year=False, sl='JKSE'):

    if show_next_year:
        div_df['date'] = div_df['date'].apply(lambda x: x.replace(year=x.year+1))
    
    domain = [2, 3, 5, 10, 20, 50, 100]
    if sl != 'JKSE':
        domain = np.array(domain) / 10
    
    labels = div_df.apply(lambda x: f"{x['date'].strftime('%d %b')}: {x['symbol']} ({x['yield']:2.2f}%)", axis=1)

    full_chart = alt.hconcat()
    for i in range(4):
        column = alt.vconcat()
        for j in range(3):
            idx = (j*4)+i+1
            c = lesley.month_plot(div_df['date'], div_df['yield'], labels=labels, title=calendar.month_name[idx], 
                                        cmap='Greens', domain=domain, show_date=True, month=idx)
            column = column & c
        full_chart = full_chart | column
    return full_chart


def plot_treemap(tree_data, size_var='Market Cap', color_var='Dividend Yield', show_gradient=False, colormap='green_shade', group_secs=True):

    cmap_options = {
        'red_green': ['#A30000', '#9f6e73', '#aaa', '#79ab78', '#08701b'],
        'green_shade' : ["#79ab78", "#08701b"],
        'red_shade': ['#000000', '#9f6e73', '#A30000']
    }

    color = cmap_options[colormap]
    
    levels = [
            # Level 0 → Root ALL
            {
                "itemStyle": {
                    "borderWidth": 1,
                    "gapWidth": 1,
                    "borderColor": "#eee"
                },
                "upperLabel": {
                    "show": True,
                    "color": "#111",
                    "fontWeight": "bold",
                    "fontSize": 14
                },
                "label": {"show": False}
            },
            # Level 1 → Sector
            {
                "itemStyle": {
                    "borderWidth": 1,
                    "gapWidth": 1,
                    "borderColor": "#ddd",
                },
                "upperLabel": {
                    "show": True,
                    "color": "#333",
                    "fontWeight": "600",
                    "fontSize": 12
                },
                "label": {"show": False} # hide stock labels at this level
            },
            # level 2 -> Industry
            {
                "itemStyle": {
                    "gapWidth": 1,
                    "borderWidth": 1,
                    "borderColor": "#ccc",
                },
                "upperLabel": {
                    "show": True,
                    "color": "#333",
                    "fontWeight": "600",
                    "fontSize": 10
                },
            },
            # Level 3 → Individual stocks
            {
                "itemStyle": {
                    "borderWidth": 1,
                    "gapWidth": 1,
                    "borderColor": "#ccc",
                },
                "label": {
                    "show": True,
                    "color": "#fff",
                    "fontSize": 10,
                    'fontWeight': 'bold',
                    "overflow": "truncate"
                },
                "upperLabel": {"show": False}
            }
        ]
    
    if not group_secs:
         levels = [
            # Level 0 → Root ALL
            {
                "itemStyle": {
                    "borderWidth": 1,
                    "gapWidth": 1,
                    "borderColor": "#eee"
                },
                "upperLabel": {
                    "show": True,
                    "color": "#111",
                    "fontWeight": "bold",
                    "fontSize": 14
                },
                "label": {"show": False}
            },
            # Level 1 → Individual Stocks (Leaves)
            {
                "itemStyle": {
                    "borderWidth": 1,
                    "gapWidth": 1,
                    "borderColor": "#ccc",
                },
                "label": {
                    "show": True,
                    "color": "#fff",
                    "fontSize": 10,
                    'fontWeight': 'bold',
                    "overflow": "truncate"
                },
                "upperLabel": {"show": False}
            }
         ]

    base_series = [
    {
        "name": "ALL",
        "type": "treemap",
        "visibleMin": 200,
        "width": "95%",
        "height": "90%",
        "label": {
            "show": True,
            "formatter": "{b}",
            "position": "inside",
            "verticalAlign": "middle",
            "align": "center",
            "fontSize": 11,
            "overflow": "truncate"
        },
        "labelLayout": JsCode(
                "function(params){if(params.rect.width<5||params.rect.height<5)return{fontSize:0};return{fontSize:Math.min(Math.sqrt(params.rect.width*params.rect.height)/7,25)};}"
            ).js_code,
        "upperLabel": {
            "show": True,
            "formatter": "{b}",
            "color": "#111",
            "fontSize": 14,
            "fontWeight": "bold"
        },
        "itemStyle": {
            "borderColor": "#fff",
            "borderWidth": 1,
            "gapWidth": 1
        },
        "levels": levels,
        "colorMappingBy": "index",
        "data": tree_data
    }
    ]

    import copy
    gradient_series = copy.deepcopy(base_series)
    gradient_series[0]['visualMin'] = 0
    gradient_series[0]['visualMax'] = 100
    gradient_series[0]['visualDimension'] = 2
    gradient_series[0]['colorMappingBy'] = 'value'
    gradient_series[0]['color'] = color

    if show_gradient:
        series = gradient_series
    else:
        series = base_series


    title = f'{{boldText|{color_var}}} of Biggest stock for each sector based on {{boldText|{size_var}}}'
    option = {
        'title': {
        'text': title,
        'left': 'center',
        'textStyle': {
            'color': '#333',
            'fontSize': 20,
            'rich': {
                'boldText': {
                    'fontWeight': 'bold',
                    'fontSize': 24,
                    'color': 'green'
                }
            }
        }
    },
        'grid': {
            'left': '10%',
            'right': '10%',
            'top': 0,
            'bottom': 0
        },
        "tooltip": {
            "formatter": JsCode(
                f"function(info){{var value=info.value;var treePathInfo=info.treePathInfo;var treePath=[];for(var i=1;i<treePathInfo.length;i+=1){{treePath.push(treePathInfo[i].name)}}return['<div class=\"tooltip-title\">'+treePath.join('/')+'</div>','{size_var}: '+ value[0] +''].join('')}};"
            ).js_code,
        },
        "series": series,
    }

    return option


def plot_radar_chart(categories, data, title='Rating', color='rgba(0, 150, 0, 1)'):
    
    # if color is hex, convert to rgba
    if color.startswith('#'):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        line_color = f'rgba({r}, {g}, {b}, 1)'
        area_color = f'rgba({r}, {g}, {b}, 0.3)'
    else:
        line_color = color
        area_color = color.replace('1)', '0.3)').replace('1.0)', '0.3)')

    option = {
        'title': {
            'text': ''
        },
        'radar': {
            'indicator': [{'name': c, 'max': 100} for c in categories],
            'radius': '70%',
            'center': ['50%', '55%'],
            'axisName': {
                'color': '#333',
                'fontSize': 14,
                'fontWeight': 'bold'
            }
        },
        'series': [
            {
                'name': title,
                'type': 'radar',
                'data': [
                    {
                        'value': data,
                        'name': title,
                        'areaStyle': {
                            'color': area_color
                        },
                         'lineStyle': {
                            'color': line_color
                        },
                        'itemStyle': {
                            'color': line_color
                        }
                    }
                ]
            }
        ]
    }
    return option


def plot_card_distribution(df, column, current_val=None, color='green', height=180, show_axis=False, comparison_vals=None, x_range=None, fill_opacity=0.3):

    # Handle outliers for better visualization: remove top and bottom 5%
    # This ensures the distribution isn't squashed by extreme values
    q95 = df[column].quantile(0.95)
    q05 = df[column].quantile(0.05)
    plot_df = df[(df[column] <= q95) & (df[column] >= q05)]
    
    if color.startswith('#'):
        line_color = color
        fill_color_hex = color
    else:
        line_color = f'dark{color}'
        fill_color_hex = f'dark{color}'

    if x_range:
        x_scale = alt.Scale(domain=list(x_range), clamp=True)
    else:
        x_scale = alt.Undefined

    if show_axis:
        x_axis = alt.X(f'{column}:Q', scale=x_scale)
    else:
        x_axis = alt.X(f'{column}:Q', title=None, axis=None, scale=x_scale)

    if fill_opacity <= 0:
        # No fill: just show the outline line
        kde = alt.Chart(plot_df).transform_density(
            column,
            as_=[column, 'density'],
        ).mark_line(
            color=line_color
        ).encode(
            x=x_axis,
            y=alt.Y('density:Q', axis=None),
            tooltip=[alt.Tooltip(column, format='.2f')]
        )
    else:
        stops = [
            alt.GradientStop(color='white', offset=0),
            alt.GradientStop(color=fill_color_hex, offset=1)
        ]
        kde = alt.Chart(plot_df).transform_density(
            column,
            as_=[column, 'density'],
        ).mark_area(
            line={'color': line_color},
            opacity=fill_opacity,
            color=alt.Gradient(
                gradient='linear',
                stops=stops,
                x1=1,
                x2=1,
                y1=1,
                y2=0
            )
        ).encode(
            x=x_axis,
            y=alt.Y('density:Q', axis=None),
            tooltip=[alt.Tooltip(column, format='.2f')]
        )
    
    layers = [kde]

    if current_val is not None:
        # We create a dataframe for the rule. 
        # If the current value is outside the plot range, we clip it to the edge so the user sees it's extreme
        display_val = max(min(current_val, q95), q05)
        
        rule = alt.Chart(pd.DataFrame({column: [display_val]})).mark_rule(color=color, strokeWidth=3).encode(
            x=column
        ).encode(
            tooltip=[alt.Tooltip(column, format='.2f')]
        )
        layers.append(rule)

    if comparison_vals is not None:
        # Distinct color palette for comparison stocks
        _PALETTE = [
            '#e41a1c', '#377eb8', '#ff7f00', '#984ea3',
            '#a65628', '#f781bf', '#4daf4a', '#999999',
        ]

        # comparison_vals is a dict of {label: value}
        comp_data = []
        # Sort by value to handle proximity in sequence
        sorted_comp = sorted(comparison_vals.items(), key=lambda x: x[1])

        last_val = -float('inf')
        current_level = 0
        # Determine a proximity threshold based on the displayed range
        # If values are within 8% of the range, we stagger them
        threshold = (q95 - q05) * 0.08 if q95 > q05 else 1.0

        for i, (label, val) in enumerate(sorted_comp):
            display_val = max(min(val, q95), q05)

            if display_val - last_val < threshold:
                current_level = (current_level + 1) % 4  # Cycle through 4 vertical levels
            else:
                current_level = 0

            y_pos = 10 + (current_level * 20)  # 20px spacing between levels
            stock_color = _PALETTE[i % len(_PALETTE)]
            comp_data.append({column: display_val, 'label': label, 'y_pos': y_pos, 'color': stock_color})
            last_val = display_val

        comp_df = pd.DataFrame(comp_data)

        # Add one rule layer per stock so each can carry its own solid color
        # Note: use iterrows() instead of itertuples() because column names that are
        # Python reserved keywords (e.g. 'yield') get silently renamed by itertuples().
        for _, row in comp_df.iterrows():
            row_df = pd.DataFrame([{column: row[column], 'label': row['label'], 'y_pos': row['y_pos']}])
            rule = alt.Chart(row_df).mark_rule(
                color=row.color, strokeWidth=2
            ).encode(
                x=alt.X(f'{column}:Q'),
                tooltip=[alt.Tooltip('label:N'), alt.Tooltip(f'{column}:Q', format='.2f')]
            )
            text = alt.Chart(row_df).mark_text(
                align='left',
                baseline='top',
                dx=5,
                dy=5,
                angle=0,
                fontSize=13,
                fontWeight='bold',
                color=row.color
            ).encode(
                x=alt.X(f'{column}:Q'),
                text='label:N',
                y=alt.Y('y_pos:Q', axis=None).scale(domain=[0, height], range=[0, height])
            )
            layers.append(rule)
            layers.append(text)

    return alt.layer(*layers).resolve_scale(y='independent').properties(height=height)


def plot_card_histogram(df, column, current_val, color='green', height=180):
    
    if color.startswith('#'):
        bar_color = color
    else:
        bar_color = f'dark{color}'

    # Basic histogram with highlight
    base = alt.Chart(df).transform_bin(
        as_=['bin_min', 'bin_max'], field=column, bin=alt.Bin(maxbins=20, step=1)
    ).mark_bar(
        color=bar_color,
        stroke=bar_color,
        cursor='default'
    ).encode(
        x=alt.X('bin_min:Q', bin='binned', title=None, axis=None),
        x2=alt.X2('bin_max:Q'),
        y=alt.Y('count()', title=None, axis=None),
        opacity=alt.condition(
            f"datum.bin_min <= {current_val} && {current_val} < datum.bin_max",
            alt.value(1.0),
            alt.value(0.3)
        ),
        tooltip=[alt.Tooltip('bin_min:Q', title=column, format='.0f'), 'count()']
    )

    return base.properties(height=height)
