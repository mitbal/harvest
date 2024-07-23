import streamlit as st
from st_pages import Page, show_pages, add_page_title
from streamlit_lottie import st_lottie

st.title('Panen Dividen')
st_lottie('https://lottie.host/632869fc-4f0f-4707-84ff-00c73c591eed/QEtuVsuLs5.json')

show_pages(
    [
        Page('home.py', 'Home', '🏰'),
        Page('pages/hist_insight.py', 'Historical Insight', '🧾'),
        Page('pages/porto_analysis.py', 'Portfolio Analysis', '🧮')
    ]
)
