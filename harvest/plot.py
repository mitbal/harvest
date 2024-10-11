import pandas as pd
import altair as alt

def plot_quarter_income(fin_df):

    chart = alt.Chart(fin_df).mark_bar().encode(
        x=alt.X('period'),
        y=alt.Y('netIncome'),
        color='period',
        column='calendarYear'
    ).properties(
        width=100
    )

    return chart


def plot_candlestick(price_df):
    open_close_color = alt.condition(
        'datum.open <= datum.close',
        alt.value("#06982d"),
        alt.value("#ae1325")
    )

    base = alt.Chart(price_df).encode(
        alt.X('date:T',
            axis=alt.Axis(
                # format='%m/%Y',
                labelAngle=-45
            )
        ),
        color=open_close_color
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

    candlestick = (rule + bar).properties(
        width=1150,
        height=400
    )
    
    return candlestick


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
    div_bar = alt.Chart(df_train).mark_bar().encode(
        alt.X('year:N'),
        alt.Y('adjDividend'),
        color=alt.condition(alt.datum['inc'] > 0, alt.value('#ff796c'), alt.value('#008631'))
    ).properties(
        height=450,
        width=600
    )

    return div_bar
