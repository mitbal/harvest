import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title='Panen Dividen',
    page_icon='🪙',
    layout='wide'
)

st.title('Panen Dividen')

cols = st.columns([0.35, 0.65])

with cols[0]:
    st_lottie(animation_source='https://lottie.host/632869fc-4f0f-4707-84ff-00c73c591eed/QEtuVsuLs5.json',
              height=380)

with open('README.md', 'r') as f:
    desc = f.read()

with cols[1]:
    st.markdown(desc, unsafe_allow_html=True)

if 'porto_file' not in st.session_state:
    st.session_state['porto_file'] = 'EMPTY'
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = 'EMPTY'
