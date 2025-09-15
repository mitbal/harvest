import logging

import streamlit as st

from harvest.utils import setup_logging

try:
    st.set_page_config(layout='wide')
except Exception as e:
    print('Set Page config has been called before')


@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger
logger = get_logger('article')


# 1. as sidebar menu
with st.sidebar:
    st.markdown("## Article List")
    st.html('<a href="/article?name=glossary">Glossarium</a>')
    st.html('<a href="/article?name=comparison">IHSG vs S&P500</a>')
    st.html('<a href="/article?name=sido">Better (Call) Buy SIDO</a>')
    st.html('<a href="/article?name=myor">MYOR != Pucuk Harum</a>')
    st.html('<a href="/article?name=bubble">Newton Bizarre (Financial) Adventure</a>')
    st.html('<a href="/article?name=investor">Investor Indonesia 2025</a>')
    # st.html('<a href="/article?name=bjtm">BJ * (TM + BR) = Cuan???</a>')

if len(st.query_params) == 0:
    st.title('Article')
    st.write('Please select an article from the sidebar')
    st.stop()

article_name = st.query_params['name']

with open(f'articles/{article_name}/{article_name}.md', 'r') as f:
    st.markdown(f.read())
    logger.info(f'opening article {article_name}')
