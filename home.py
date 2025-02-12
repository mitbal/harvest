import streamlit as st
from streamlit_lottie import st_lottie

#### Setup and configuration

st.set_page_config(
    page_title='Panen Dividen',
    page_icon='🪙',
    layout='wide'
)

st.title('Panen Dividen')

if 'porto_file' not in st.session_state:
    st.session_state['porto_file'] = 'EMPTY'
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = 'EMPTY'

#### End of setup and configuration

#### Main content of Landing Page
cols = st.columns([0.35, 0.65])

with cols[0]:
    st_lottie(animation_source='https://lottie.host/632869fc-4f0f-4707-84ff-00c73c591eed/QEtuVsuLs5.json',
              height=380)

with open('README.md', 'r') as f:
    desc = f.read()

with cols[1]:
    st.markdown(desc, unsafe_allow_html=True)

##### Features breakdown

st.divider()

with open('style.html', 'r') as f:
    style = f.read()
st.html(style)

with open('feature.html', 'r') as f:
    feature = f.read()
st.html(feature)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
# text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
# text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ♥️ and lots of ☕ by <a style='display: block; text-align: center;' href="https://github.com/mitbal" target="_blank">mitochondrion</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
