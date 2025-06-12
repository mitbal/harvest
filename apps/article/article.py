import streamlit as st

try:
    st.set_page_config(layout='wide')
except Exception as e:
    print('Set Page config has been called before')

# 1. as sidebar menu
with st.sidebar:
    st.markdown("## Article List")
    st.html('<a href="/article?name=glossary">Glossarium</a>')
    st.html('<a href="/article?name=comparison">IHSG vs S&P500</a>')
    st.html('<a href="/article?name=sido">Better (Call) Buy SIDO</a>')
    st.html('<a href="/article?name=myor">MYOR != Pucuk Harum</a>')
    st.html('<a href="/article?name=bubble">Newton Bizarre (Financial) Adventure</a>')

if len(st.query_params) == 0:
    st.title('Article')
    st.write('Please select an article from the sidebar')
    st.stop()

article_name = st.query_params['name']

with open(f'articles/{article_name}/{article_name}.md', 'r') as f:
    st.markdown(f.read())
