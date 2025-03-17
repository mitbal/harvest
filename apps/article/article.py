import streamlit as st

# 1. as sidebar menu
with st.sidebar:
    st.markdown("# Article List")
    st.html('<a href="/article?name=glossary">Glossary</a>')
    st.html('<a href="/article?name=comparison">IHSG vs S&P500</a>')

if len(st.query_params) == 0:
    st.title('Article')
    st.write('Please select an article from the sidebar')
    st.stop()

article_name = st.query_params['name']

with open(f'articles/{article_name}/{article_name}.md', 'r') as f:
    st.markdown(f.read())
