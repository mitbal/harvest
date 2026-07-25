import uuid
import logging

import streamlit as st

from harvest.utils import setup_logging


st.set_page_config(
    layout='wide',
    page_title='Panen Dividen'
)


# Suppress noisy asyncio/tornado logs from Railway/Docker disconnections
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('tornado.access').setLevel(logging.CRITICAL)
logging.getLogger('tornado.application').setLevel(logging.CRITICAL)
logging.getLogger('tornado.general').setLevel(logging.CRITICAL)
# Suppress bot-scanner noise (e.g. requests for system/js/core.js)
logging.getLogger('streamlit.web.server.media_file_handler').setLevel(logging.CRITICAL)


page_home = st.Page('home.py', title='Home', icon='🪙')
page_screener = st.Page('apps/screener/stock_picker.py', title='Ranking Screener', icon='💸')
page_comparison = st.Page('apps/screener/stock_comparison.py', title='Stock Comparison', icon='⚖️')
page_market_watch = st.Page('apps/screener/market_watch.py', title='Stock Heatmap', icon='📡')
# page_market_animation = st.Page('apps/screener/market_watch_animation.py', title='Market Animation', icon='📽️')
page_porto = st.Page('apps/porto/porto_overview.py', title='Portfolio Analytics', icon='💰')
page_history = st.Page('apps/history/history_overview.py', title='Historical Breakdown', icon='🧭')
page_calendar = st.Page('apps/calendar/calendar.py', title='Payout Calendar', icon='📅')
page_best_timing = st.Page('apps/calendar/best_timing.py', title='Best Timing', icon='🎯')
page_assistant = st.Page('apps/assistant/assistant.py', title='Financial Assistant', icon='🧑‍🏫')
page_article = st.Page('apps/article/article.py', title='Analysis Article', icon='📰')
page_simulator = st.Page('apps/simulator/simulator.py', title='Compounding Simulator', icon='🎮')
# page_backtester = st.Page('apps/trading/backtester.py', title='Position Trading - Long Term', icon='📈')
# page_day_trading = st.Page('apps/trading/day_trading.py', title='Day Trading Lab', icon='⚡')
# page_swing_trading = st.Page('apps/trading/swing_trading.py', title='Swing Trading Lab', icon='🏄')


pages = st.navigation(
    {
        'Home': [page_home],
        'Dividend': [page_screener,
                 page_calendar,
                 page_best_timing,
                #  page_market_watch,
                #  page_market_animation,
                #  page_comparison,
                 page_porto,
                 page_simulator,
                 ],
        'Market':
            [
                page_market_watch,
                page_comparison,
            ],
        # 'Trading': [
        #         page_backtester,
        #         page_day_trading,
        #         page_swing_trading,
        # ]
    }
)

# Redirect to portfolio only right after a fresh OAuth login.
# We detect this by checking the HTTP Referer header: after completing
# OAuth, the browser referrer will be from the OAuth provider domain
# (e.g. accounts.google.com). Direct opens / bookmarks have no referrer.
if st.user.is_logged_in and 'has_redirected_after_login' not in st.session_state:
    st.session_state['has_redirected_after_login'] = True
    if pages.title == 'Home':
        try:
            referrer = st.context.headers.get('Referer', '')
        except Exception:
            referrer = ''
        # Known OAuth / identity provider domains
        _OAUTH_DOMAINS = [
            'accounts.google.com',
            'github.com',
            'login.microsoftonline.com',
            'auth0.com',
            'okta.com',
        ]
        is_fresh_oauth_login = any(domain in referrer for domain in _OAUTH_DOMAINS)
        if is_fresh_oauth_login:
            st.switch_page(page_porto)


# --- URL Parameter Tracking ---
# Runs on every page navigation; URL params captured once per session
if 'tracked_url_params' not in st.session_state:
    st.session_state['tracked_url_params'] = set()

if 'visitor_id' not in st.session_state:
    st.session_state['visitor_id'] = str(uuid.uuid4())

# Capture any new url parameters present in this request
for param, value in st.query_params.items():
    if param not in st.session_state:
        st.session_state[param] = value
        st.session_state['tracked_url_params'].add(param)

if 'visited_pages' not in st.session_state:
    st.session_state['visited_pages'] = set()

if pages.title not in st.session_state['visited_pages']:
    _logger = setup_logging('tracking')
    
    # Build a string of all tracked URL parameters
    logged_params = [f"{p}={st.session_state[p]}" for p in st.session_state['tracked_url_params']]
    params_str = " | ".join(logged_params)
    
    log_msg = f"VISIT | visitor={st.session_state['visitor_id']} | page={pages.title}"
    if params_str:
        log_msg += f" | {params_str}"
        
    _logger.info(log_msg)
    st.session_state['visited_pages'].add(pages.title)
# --- End URL Parameter Tracking ---


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
    st.html(
        'Read the Blog!<br/>'
        '<a id="blog-link" href="https://blog.panendividen.com?utm_source=pd_web" target="_blank" '
        'style="display:inline-flex;align-items:center;gap:6px;background:#ffffff;'
        'color:#14532D;padding:6px 14px;border-radius:6px;text-decoration:none;'
        'font-weight:600;font-size:0.85rem;">✍️ blog.panendividen.com</a>'
        '<script>'
        'document.getElementById("blog-link").addEventListener("click", function() {'
        '  console.log("[tracking] blog_click");'
        '  var url = new URL(window.location.href);'
        '  url.searchParams.set("blog_click", "1");'
        '  fetch(url.toString(), {method: "GET", mode: "no-cors"}).catch(function(){});'
        '});'
        '</script>'
    )
    st.divider()

pages.run()

with open('footer.html', 'r') as f:
    footer = f.read()
st.markdown(footer, unsafe_allow_html=True)
