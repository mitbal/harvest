import logging
import streamlit as st
from harvest.utils import setup_logging

@st.cache_resource
def get_logger(name, level=logging.INFO):
    logger = setup_logging(name, level)
    return logger

logger = get_logger('article')

# Article Metadata
ARTICLES = [
    {"name": "glossary", "title": "Glossarium", "icon": "📚"},
    {"name": "comparison", "title": "IHSG vs S&P500", "icon": "📈"},
    {"name": "sido", "title": "Better (Call) Buy SIDO", "icon": "💊"},
    {"name": "myor", "title": "MYOR != Pucuk Harum", "icon": "☕"},
    {"name": "bubble", "title": "Newton Bizarre (Financial) Adventure", "icon": "🛁"},
    {"name": "investor", "title": "Investor Indonesia 2025", "icon": "🇮🇩"},
]

# State Management
if "selected_article" not in st.session_state:
    st.session_state.selected_article = st.query_params.get("name")

# If an article is selected, display it
if st.session_state.selected_article:
    article_name = st.session_state.selected_article
    
    # Back button
    if st.button("← Back to Articles"):
        st.session_state.selected_article = None
        st.query_params.clear()
        st.rerun()

    try:
        with open(f'articles/{article_name}/{article_name}.md', 'r') as f:
            st.markdown(f.read())
            logger.info(f'opening article {article_name}')
    except FileNotFoundError:
        st.error(f"Article '{article_name}' not found.")
        if st.button("Return Home"):
            st.session_state.selected_article = None
            st.query_params.clear()
            st.rerun()

else:
    # Main Page: Article Selection View
    st.title('Panen Dividen Articles')
    st.write('Select an article to read more about dividend investing and market analysis.')
    
    # Render articles as cards in a grid
    cols = st.columns(3)
    for idx, article in enumerate(ARTICLES):
        with cols[idx % 3]:
            st.divider()
            st.markdown(f"### {article['icon']} {article['title']}")
            if st.button(f"Read '{article['title']}'", key=article['name'], width='stretch'):
                st.session_state.selected_article = article['name']
                st.query_params["name"] = article['name']
                st.rerun()
