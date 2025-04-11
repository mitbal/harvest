import streamlit as st

page_home = st.Page('home.py', title='Home', icon='🪙')
page_screener = st.Page('apps/screener/stock_picker.py', title='Stock Picker', icon='💸')
page_porto = st.Page('apps/porto/porto_overview.py', title='Portfolio Overview', icon='💰')
page_history = st.Page('apps/history/history_overview.py', title='Historical Overview', icon='🧭')
page_calendar = st.Page('apps/calendar/calendar.py', title='Dividend Calendar', icon='📅')
page_assistant = st.Page('apps/assistant/assistant.py', title='Financial Assistant', icon='🧑‍🏫')
page_article = st.Page('apps/article/article.py', title='Article', icon='📰')
page_simulator = st.Page('apps/simulator/simulator.py', title='Simulator', icon='🎮')

pages = st.navigation(
    {
        'Home': [page_home],
        'Apps': [page_screener, page_calendar, page_porto, page_history, page_assistant, page_article, page_simulator]
    }
)
pages.run()
