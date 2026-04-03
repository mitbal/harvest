import uuid
import logging

import streamlit as st

from harvest.utils import setup_logging


# Suppress noisy asyncio/tornado logs from Railway/Docker disconnections
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('tornado.access').setLevel(logging.CRITICAL)
logging.getLogger('tornado.application').setLevel(logging.CRITICAL)
logging.getLogger('tornado.general').setLevel(logging.CRITICAL)


page_home = st.Page('home.py', title='Home', icon='🪙')
page_screener = st.Page('apps/screener/stock_picker.py', title='Stock Picker', icon='💸')
page_porto = st.Page('apps/porto/porto_overview.py', title='Portfolio Overview', icon='💰')
page_history = st.Page('apps/history/history_overview.py', title='Historical Breakdown', icon='🧭')
page_calendar = st.Page('apps/calendar/calendar.py', title='Dividend Calendar', icon='📅')
page_assistant = st.Page('apps/assistant/assistant.py', title='Financial Assistant', icon='🧑‍🏫')
page_article = st.Page('apps/article/article.py', title='Analysis Article', icon='📰')
page_simulator = st.Page('apps/simulator/simulator.py', title='Compounding Simulator', icon='🎮')
# page_trading = st.Page('apps/voc/copenhagen.py', title='Copenhagen Model', icon='🎮')
# page_viz = st.Page('apps/viz/viz.py', title='Vis')


pages = st.navigation(
    {
        'Home': [page_home],
        'Apps': [page_screener,
                 page_calendar,
                #  page_assistant,
                 page_article,
                 page_simulator,
                 page_porto,
                #  page_viz,
                #  page_history,
                #  page_trading
                 ]
    }
)


# --- UTM Tracking ---
# Runs on every page navigation; UTM params captured once per session
UTM_PARAMS = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term']

if 'visitor_id' not in st.session_state:
    st.session_state['visitor_id'] = str(uuid.uuid4())

for param in UTM_PARAMS:
    if param not in st.session_state:
        st.session_state[param] = st.query_params.get(param, None)

if 'visited_pages' not in st.session_state:
    st.session_state['visited_pages'] = set()

if pages.title not in st.session_state['visited_pages']:
    _logger = setup_logging('utm')
    _logger.info(
        f"VISIT | visitor={st.session_state['visitor_id']} "
        f"| page={pages.title} "
        f"| source={st.session_state['utm_source']} "
        f"| medium={st.session_state['utm_medium']} "
        f"| campaign={st.session_state['utm_campaign']}"
    )
    st.session_state['visited_pages'].add(pages.title)
# --- End UTM Tracking ---


st.html("""
<style>
    /* Apply primary color to all page titles (st.title → h1) */
    h1 {
        color: #064E3B !important;
    }

    /* Make sidebar page names bigger */
    [data-testid="stSidebarNav"] a span,
    [data-testid="stSidebarNavLink"] span {
        font-size: 1.1rem !important;
    }
</style>
""")

with st.sidebar:
    st.html(f'Support me on<br/>'+'<a href="https://trakteer.id/mitbal" target="_blank"><img id="wse-buttons-preview" src="https://cdn.trakteer.id/images/embed/trbtn-red-1.png?date=18-11-2023" height="40" style="border:0px;height:40px;" alt="Trakteer Saya"></a> ')
    st.markdown('[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/mitbal)')
    st.html(f'Join the Community!<br/>'+'<a href="https://reddit.com/r/panendividen" target="_blank"><img id="wse-buttons-preview" src="https://images.icon-icons.com/2530/PNG/512/reddit_button_icon_151844.png" height="30" style="border:0px;height:40px;" alt="Reddit r/panendividen"></a> ')
    st.divider()
pages.run()

