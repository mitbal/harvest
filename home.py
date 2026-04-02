import re
import logging

import streamlit as st
from streamlit_lottie import st_lottie


def md_to_html(text):
    """Lightweight markdown-to-HTML converter using stdlib only."""
    lines = text.split('\n')
    html_lines = []
    for line in lines:
        # Headings
        if line.startswith('#### '):
            content = re.sub(r'\*\*(.+?)\*\*', r'\1', line[5:])
            line = f'<h4>{content}</h4>'
        elif line.startswith('### '):
            content = re.sub(r'\*\*(.+?)\*\*', r'\1', line[4:])
            line = f'<h3>{content}</h3>'
        else:
            # Images: ![alt](url) — must come before links
            line = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', line)
            # Links with badges: [![alt](img_url)](link_url)
            line = re.sub(r'\[!\[([^\]]*)\]\(([^)]+)\)\]\(([^)]+)\)',
                          r'<a href="\3" target="_blank"><img src="\2" alt="\1"></a>', line)
            # Inline links: [text](url)
            line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', line)
            # Bold: **text**
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            # Wrap non-empty, non-heading lines in <p>
            if line.strip():
                line = f'<p>{line}</p>'
        html_lines.append(line)
    return '\n'.join(html_lines)

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

with open('readme_style.html', 'r') as f:
    readme_style = f.read()

with open('README.md', 'r') as f:
    desc_md = f.read()
desc_html = md_to_html(desc_md)

with cols[1]:
    st.markdown(readme_style, unsafe_allow_html=True)
    st.markdown(f'<div class="readme-wrapper">{desc_html}</div>', unsafe_allow_html=True)

##### Features breakdown

st.divider()

with open('style.html', 'r') as f:
    style = f.read()
st.html(style)

with open('feature.html', 'r') as f:
    feature = f.read()
st.html(feature)

with open('footer.html', 'r') as f:
    footer = f.read()
st.markdown(footer, unsafe_allow_html=True)

