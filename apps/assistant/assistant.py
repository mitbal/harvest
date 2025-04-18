import os
import json
import requests
from datetime import datetime

import redis
import streamlit as st


# Set page configuration
st.set_page_config(
    page_title='Om Jin the Financial Advisor',
    page_icon='🧞‍♂️',
    layout='wide'
)


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


# Helper function to call OpenRouter API
def get_ai_response(prompt, chat_history, api_key, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "localhost",
        "X-Title": "Panen Dividen",
        "Content-Type": "application/json"
    }

    system_prompt = """
    You are a financial advisor. 
    Answer the question of user based on the data below. 
    Don't make things up.
    Answer in user language.
    """
    
    # Format messages for the API
    messages = []
    messages.append({
        "role": "system",
        "content": system_prompt
    })
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
    
    data = {
        "model": model,
        "messages": messages
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(data),
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        st.error(f"Error: {response.status_code}")
        st.json(response.json())
        return f"Sorry, I encountered an error. Status code: {response.status_code}"

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
st.sidebar.title("Settings")

# API Key input
if "OPENROUTER_API_KEY" not in st.session_state:
    api_key = st.sidebar.text_input("Enter your OpenRouter API Key:", type="password")
    if api_key:
        st.session_state.OPENROUTER_API_KEY = api_key
else:
    st.sidebar.success("API Key is set! ✓")
    if st.sidebar.button("Change API Key"):
        del st.session_state.OPENROUTER_API_KEY
        st.rerun()
        
# Model selection
model_options = {
    'Gemini 2.0 Flash': 'google/gemini-2.0-flash-exp:free',
    'Gemini 2.5 Pro': 'google/gemini-2.5-pro-exp-03-25:free'
}

selected_model = st.sidebar.selectbox(
    "AI Model Selection:",
    list(model_options.keys()),
    index=0
)

model_id = model_options[selected_model]

# Reset chat button
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.rerun()
    
# Display information about selected model
st.sidebar.subheader("About the selected model")
st.sidebar.info(f"You're currently using **{selected_model}** (`{model_id}`)")

# Main chat interface
st.title("🧞‍♂️ Om Jin the Financial Advisor")
st.caption(f"Powered by {selected_model} via OpenRouter")

# Check if API Key is set
if "OPENROUTER_API_KEY" not in st.session_state:
    st.warning("Please enter your OpenRouter API Key in the sidebar to continue.")
    st.info("Don't have an API key? Get one at [openrouter.ai](https://openrouter.ai)")
    st.stop()


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

# st.write('api key', st.session_state.OPENROUTER_API_KEY)

# Chat input
if prompt := st.chat_input("Message the AI assistant..."):
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
    with st.spinner(f"Waiting for response from {selected_model}..."):
        api_key = st.session_state.OPENROUTER_API_KEY
        response = get_ai_response(
            prompt, 
            st.session_state.messages[:-1],  # Exclude the just-added message
            api_key,
            model_id
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
