import streamlit as st

page_home = st.Page('home.py', title='Home', icon='🪙')
page_screener = st.Page('apps/screener/stock_picker.py', title='Stock Picker', icon='💸')
page_porto = st.Page('apps/porto/porto_overview.py', title='Portfolio Overview', icon='💰')
page_history = st.Page('apps/history/history_overview.py', title='Historical Breakdown', icon='🧭')
page_calendar = st.Page('apps/calendar/calendar.py', title='Dividend Calendar', icon='📅')
page_assistant = st.Page('apps/assistant/assistant.py', title='Financial Assistant', icon='🧑‍🏫')
page_article = st.Page('apps/article/article.py', title='Analysis Article', icon='📰')
page_simulator = st.Page('apps/simulator/simulator.py', title='Compounding Simulator', icon='🎮')
# page_trading = st.Page('apps/voc/copenhagen.py', title='Copenhagen Model', icon='🎮')

pages = st.navigation(
    {
        'Home': [page_home],
        'Apps': [page_screener,
                 page_calendar,
                 page_assistant,
                 page_article,
                 page_simulator,
                 page_porto,
                #  page_viz,
                #  page_history,
                #  page_trading
                 ]
    }
)

with st.sidebar:
    st.html(f'Support me on<br/>'+'<a href="https://trakteer.id/mitbal" target="_blank"><img id="wse-buttons-preview" src="https://cdn.trakteer.id/images/embed/trbtn-red-1.png?date=18-11-2023" height="40" style="border:0px;height:40px;" alt="Trakteer Saya"></a> ')
    st.markdown('[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/mitbal)')
    st.divider()
pages.run()

