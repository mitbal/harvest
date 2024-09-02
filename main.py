import streamlit as st

page_home = st.Page('home.py', title='Home', icon='🪙')
page_porto = st.Page('apps/porto/porto_overview.py', title='Portfolio Overview', icon='💰')
page_history = st.Page('apps/history/history_overview.py', title='Historical Overview', icon='🧭')

pages = st.navigation(
    {
        'Home': [page_home],
        'Apps': [page_porto, page_history]
    }
)
pages.run()
