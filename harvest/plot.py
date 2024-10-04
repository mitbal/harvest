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
