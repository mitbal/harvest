import streamlit as st
from streamlit_lottie import st_lottie

st.title('Panen Dividen')
st_lottie(animation_source='https://lottie.host/632869fc-4f0f-4707-84ff-00c73c591eed/QEtuVsuLs5.json',
          height=200)

with open('README.md', 'r') as f:
    desc = f.read()
st.markdown(desc)

if 'porto_file' not in st.session_state:
    st.session_state['porto_file'] = 'EMPTY'
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = 'EMPTY'
