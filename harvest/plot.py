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
