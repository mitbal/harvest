import streamlit as st
import requests
import json
import os
from datetime import datetime
# from dotenv import load_dotenv

# Load environment variables (optional)
# load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Assistant with OpenRouter",
    page_icon="🤖",
    layout="wide"
)

# Helper function to call OpenRouter API
def get_ai_response(prompt, chat_history, api_key, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://your-app-hostname.com",  # Replace with your actual hostname in production
        "X-Title": "AI Chatbot App",  # Your app name
        "Content-Type": "application/json"
    }
    
    # Format messages for the API
    messages = []
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
        data=json.dumps(data)
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
        st.experimental_rerun()
        
# Model selection
model_options = {
    # "OpenAI GPT-4": "openai/gpt-4-turbo",
    # "OpenAI GPT-3.5 Turbo": "openai/gpt-3.5-turbo",
    # "Anthropic Claude 3 Opus": "anthropic/claude-3-opus",
    # "Anthropic Claude 3 Sonnet": "anthropic/claude-3-sonnet",
    # "Google Gemini Pro": "google/gemini-pro",
    # "Meta Llama 3 70B": "meta/llama-3-70b-instruct"
    'Gemini Free': 'google/gemini-2.0-flash-lite-preview-02-05:free'
}

selected_model = st.sidebar.selectbox(
    "Choose AI Model:",
    list(model_options.keys()),
    index=0  # Default to GPT-3.5 Turbo
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
st.title("🤖 AI Assistant")
st.caption(f"Powered by {selected_model} via OpenRouter")

# Check if API Key is set
if "OPENROUTER_API_KEY" not in st.session_state:
    st.warning("Please enter your OpenRouter API Key in the sidebar to continue.")
    st.info("Don't have an API key? Get one at [openrouter.ai](https://openrouter.ai)")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "timestamp" in message:
            st.caption(f"{message['timestamp']}")

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
    with st.chat_message("user"):
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
        with st.chat_message("assistant"):
            st.markdown(response)
            st.caption(timestamp)
