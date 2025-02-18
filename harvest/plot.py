import lesley
import numpy as np
import pandas as pd
import altair as alt


def format_currency():
    return  "datum.value >= 1000000000000 ? format(datum.value/1000000000000, ',.0f') + ' T' : " +\
            "datum.value >= 1000000000 ? format(datum.value/1000000000, ',.0f') + ' B' : " +\
            "format(datum.value/1000000, ',.0f') + ' M'"


def plot_financial(fin_df, period='quarter', metric='netIncome'):
    
    if period == 'quarter':
        return plot_quarter_income(fin_df, metric)
    else:
        return plot_yearly_income(fin_df, metric)

def plot_yearly_income(fin_df, metric):
    chart = alt.Chart(fin_df).mark_bar().encode(
        x=alt.X('calendarYear'),
        y=alt.Y(f'sum({metric}):Q'),
    )
    return chart

def plot_quarter_income(fin_df, metric):

    chart = alt.Chart(fin_df).mark_bar().encode(
        x=alt.X('period'),
        y=alt.Y(metric, axis=alt.Axis(
            labelExpr=format_currency()
        )),
        color='period',
        column='calendarYear'
    ).properties(
        height=300
    )

    return chart


def plot_candlestick(price_df, width=1000, height=300):
    open_close_color = alt.condition(
        'datum.open <= datum.close',
        alt.value("#06982d"),
        alt.value("#ae1325")
    )

    # brush = alt.selection_interval(value={'y': [20, 40]})
    interval = alt.selection_interval(encodings=['x'])

    base = alt.Chart(price_df).encode(
        alt.X('date:T', scale=alt.Scale(domain=interval)),
        color=open_close_color,
    )

    rule = base.mark_rule().encode(
        alt.Y(
            'low:Q',
            title='Price',
            scale=alt.Scale(zero=False),
        ),
        alt.Y2('high:Q')
    )

    bar = base.mark_bar().encode(
        alt.Y('open:Q'),
        alt.Y2('close:Q')
    )

    candlestick = alt.layer(rule, bar).properties(
        width=width,
        height=height
    )

    view = alt.Chart(price_df).mark_bar().encode(
        x=alt.X('date:T'),
        y='volume'
    ).properties(
        width=width,
        height=50
    )
    view = view.add_selection(interval)
    
    return candlestick & view


def plot_pe_distribution(df, pe):

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
        x='PE:Q',
        y=alt.Y('DENSITY:Q', title='', axis=alt.Axis(tickSize=0, domain=False, labelAngle=0, labelFontSize=0)),
        tooltip=(
            alt.Tooltip('PE:Q', format='.2f'),
        )
    )
    x_zero = kde.mark_rule().encode(
        x=alt.datum(pe),
        color=alt.value('red'),
        size=alt.value(1),
    )
   
    return pes_dist+x_zero


def plot_pe_timeseries(pe_df):

    chart = alt.Chart(pe_df).mark_line().encode(
        x = 'date:T',
        y = alt.Y('pe').scale(zero=False),
        tooltip=(
            alt.Tooltip('date:T'),
            alt.Tooltip('pe:Q', format='.2f')
        )
    )
    return chart


def plot_dividend_history(div_df, extrapolote=False, n_future_years=0, last_val=0, inc_val=0):

    # aggregate to yearly basis for stock that paid interim during the year
    dividend_year_df = div_df.copy()
    dividend_year_df['year'] = div_df['date'].apply(lambda x: int(x.split('-')[0]))
    yearly = dividend_year_df.groupby('year')['adjDividend'].sum().to_frame().reset_index()

    # fill in the blank for the year when they do not pay dividend
    start_year = yearly.loc[0, 'year']
    end_year = yearly.loc[len(yearly)-1, 'year']

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

        ext_df = pd.DataFrame({'year': ext_years, 'adjDividend': ext_values})
        div_bar2 = alt.Chart(ext_df).mark_bar(
            cornerRadiusTopLeft=5, 
            cornerRadiusTopRight=5
        ).encode(
            alt.X('year:N'),
            alt.Y('adjDividend'),
            tooltip=['year', alt.Tooltip('adjDividend', format='.2f')]
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
    
    cal = lesley.calendar_plot(div_df['date'], div_df['yield'], 
                               show_date=True, cmap='Greens', domain=domain, nrows=2)
    return cal
