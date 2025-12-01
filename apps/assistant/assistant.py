import os
import json
import logging
from datetime import datetime

import redis
import streamlit as st
from openai import OpenAI

from harvest.utils import setup_logging


SEARCH_KEYWORD = "cari di web"  # maybe replace with "open sesame" :p


# Set page configuration
st.set_page_config(
    page_title='Om Jin the Financial Advisor',
    page_icon='🧞‍♂️',
    layout='wide'
)

@st.cache_resource
def get_logger(name, level=logging.INFO):

    logger = setup_logging(name, level)
    return logger
logger = get_logger('assistant')

# Get additional data from precomputed dividend table
@st.cache_resource
def connect_redis(redis_url):
    r = redis.from_url(redis_url)
    return r

redis_url = os.environ['REDIS_URL']
r = connect_redis(redis_url)
rjson = r.get('div_score_jkse')
div_score_json = json.loads(rjson)
content = div_score_json['content']


@st.cache_data
def get_system_prompt():
    with open('apps/assistant/system_prompt.txt', 'r') as f:
        system_prompt = f.read()
    return system_prompt
system_prompt = get_system_prompt()


# # Model selection
# model_options = {
#     'Gemini 2.0 Flash': 'google/gemini-2.0-flash-exp:free',
#     'Gemini 2.5 Pro': 'google/gemini-2.5-pro-exp-03-25:free'
# }

# model_id_default    = 'google/gemini-2.0-flash-exp:free'
model_id_default = 'google/gemma-3n-e2b-it:free'
model_id_web_search = 'openai/gpt-5-mini:online'

# Helper function to call OpenRouter API
def get_ai_response(prompt, chat_history, api_key, use_web_search=False):

    # Pick model_id
    model = model_id_default
    if use_web_search:
        model = model_id_web_search

    client = OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key,
    )

    extra_headers = {
        'HTTP-Referer': 'panendividen.com',
        'X-Title': 'Panen Dividen'
    }
    
    # Format messages for the API
    messages = []
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    if not use_web_search:
        # if use_web_search no need additional context, maybe a bit cheaper
        messages[0]['content'] += '\n' + content
    
    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add the current prompt
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=extra_headers,
            timeout=30,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}"


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize API key in session state
if "OPENROUTER_API_KEY" not in st.session_state:
    # Try to get from environment variables first
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        st.session_state.OPENROUTER_API_KEY = api_key

# Sidebar for configuration
st.sidebar.header('Sample Questions')

# add sample questions
question = st.sidebar.pills(
    '**Sample questions**',
    [
        'saham sawit apa yang dividennya paling tinggi',
        'jelaskan kenapa bjtm bagus untuk dividen',
        'bagusan mana itmg atau bssr',
        'mending ipcm atau ipcc'
    ],
    selection_mode='single',
    label_visibility='collapsed',
    default=None
)

# Reset chat button
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# Main chat interface
st.title("🧞‍♂️ Om Jin the Financial Advisor")
st.caption(f"Granting Financial Freedom wishes since 1825 BC")


avatars = {
    'assistant': '🧞‍♂️',
    'user': '🧑‍💼'
}

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatars[message['role']]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"{message['timestamp']}")


# Chat input
if prompt := st.chat_input("Message Om Jin...") or question:
    # Add user message to chat history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user", avatar=avatars['user']):
        st.markdown(prompt)
        st.caption(timestamp)

    # Get AI response
    with st.spinner(f"Om Jin is thinking..."):
        api_key = st.session_state.OPENROUTER_API_KEY
        response = get_ai_response(
            prompt, 
            st.session_state.messages[:-1],  # Exclude the just-added message
            api_key,
            use_web_search=(SEARCH_KEYWORD in prompt.lower())
        )
        
        # Add AI response to chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": timestamp
        })
        
        # Display AI response
        with st.chat_message("assistant", avatar=avatars['assistant']):
            st.markdown(response)
            st.caption(timestamp)

        logger.info(f'user: {prompt}')
        logger.info(f'ai: {response}')

disclaimer_text = """
Even with thousands of years of wisdom, Om Jin can still makes mistakes.
Please double check the answer given and always do your own research.
Remember, YOU are responsible for your own financial decisions
"""
st.sidebar.header('Disclaimer')
st.sidebar.write(disclaimer_text)
