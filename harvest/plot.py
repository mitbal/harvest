import pandas as pd
import altair as alt

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

    chart = alt.Chart(fin_df[fin_df['calendarYear'] > '2016']).mark_bar().encode(
        x=alt.X('period'),
        y=alt.Y(metric),
        color='period',
        column='calendarYear'
    ).properties(
        width=100
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

    view = alt.Chart(price_df).mark_line().encode(
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
        y='DENSITY:Q'
    )
    x_zero = kde.mark_rule().encode(
        x=alt.datum(pe),
        color=alt.value('red'),
        size=alt.value(1),
    )
   
    return pes_dist+x_zero


def plot_dividend_history(div_df):

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
        color=alt.condition(alt.datum['inc'] > 0, alt.value('#ff796c'), alt.value('#008631'))
    ).properties(
        height=450,
        width=600
    )

    return div_bar
