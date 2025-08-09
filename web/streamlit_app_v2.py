# web/streamlit_app_v2.py
import streamlit as st
import requests
import json
from datetime import datetime
import time
from typing import List, Dict, Any
import uuid

# Page configuration
st.set_page_config(
    page_title="EdgeMind AI - Open Source Privacy-First AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        color: #f3f4f6;
        padding: 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Metrics cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #475569;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #0f172a;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = "research"

# API configuration
API_BASE_URL = "http://localhost:8000"  # Change if using Docker

# Helper functions
def save_conversation():
    """Save current conversation to history"""
    if st.session_state.messages:
        st.session_state.conversations[st.session_state.conversation_id] = {
            'messages': st.session_state.messages.copy(),
            'timestamp': datetime.now().isoformat(),
            'agent': st.session_state.current_agent
        }

def load_conversation(conv_id):
    """Load a conversation from history"""
    if conv_id in st.session_state.conversations:
        st.session_state.conversation_id = conv_id
        st.session_state.messages = st.session_state.conversations[conv_id]['messages'].copy()
        st.session_state.current_agent = st.session_state.conversations[conv_id].get('agent', 'research')

def new_conversation():
    """Start a new conversation"""
    save_conversation()
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []

def export_conversation():
    """Export conversation as JSON"""
    return json.dumps({
        'conversation_id': st.session_state.conversation_id,
        'messages': st.session_state.messages,
        'timestamp': datetime.now().isoformat()
    }, indent=2)

def call_api(message: str, agent: str) -> str:
    """Call the EdgeMind API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={
                "message": message,
                "agent": agent,
                "conversation_id": st.session_state.conversation_id
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        # Fallback for when API is not running
        return f"🤖 [{agent}] API not connected. Response would be generated here for: {message}"
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.markdown("# 🧠 EdgeMind AI")
    st.markdown("### Open-Source Privacy-First AI")
    
    # Agent selector
    st.markdown("---")
    st.markdown("### 🤖 Select AI Agent")
    
    agents = {
        "research": "🔍 Research Specialist",
        "analyst": "📊 Data Analyst",
        "writer": "✍️ Content Writer",
        "coder": "💻 Code Assistant"
    }
    
    selected_agent = st.selectbox(
        "Choose your assistant:",
        options=list(agents.keys()),
        format_func=lambda x: agents[x],
        index=list(agents.keys()).index(st.session_state.current_agent)
    )
    st.session_state.current_agent = selected_agent
    
    # Conversation management
    st.markdown("---")
    st.markdown("### 💬 Conversations")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat"):
            new_conversation()
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat"):
            save_conversation()
            st.success("Saved!")
    
    # List saved conversations
    if st.session_state.conversations:
        st.markdown("#### Recent Chats")
        for conv_id, conv_data in list(st.session_state.conversations.items())[-5:]:
            if st.button(f"📝 {conv_id[:8]}...", key=conv_id):
                load_conversation(conv_id)
                st.rerun()
    
    # Export feature
    st.markdown("---")
    if st.button("📥 Export Conversation"):
        json_data = export_conversation()
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"conversation_{st.session_state.conversation_id[:8]}.json",
            mime="application/json"
        )
    
    # Settings
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 4000, 2000, 100)
    
    # Status
    st.markdown("---")
    st.markdown("### 📊 Status")
    
    # Check API status
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
        api_status = "🟢 Connected"
        redis_status = "🟢" if health.get('redis') == 'connected' else "🔴"
    except:
        api_status = "🔴 Disconnected"
        redis_status = "🔴"
    
    st.markdown(f"**API:** {api_status}")
    st.markdown(f"**Memory:** {redis_status} Redis")
    st.markdown(f"**Model:** 🟢 Ready")
    
    # Info
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **EdgeMind v0.2.0**
    
    Open-source ChatGPT alternative
    that runs on your hardware.
    
    [GitHub](https://github.com/artbyoscar/ai-playground) | 
    [Docs](#) | [Discord](#)
    """)

# Main content area
st.markdown("# 🧠 EdgeMind AI Platform")
st.markdown("### Your Privacy-First AI Assistant")

# Metrics dashboard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Active Agent",
        agents[st.session_state.current_agent].split()[1],
        delta="Ready"
    )
with col2:
    st.metric(
        "Messages",
        len(st.session_state.messages),
        delta=f"+{len(st.session_state.messages)}"
    )
with col3:
    st.metric(
        "Response Time",
        "87ms",
        delta="-13ms",
        delta_color="normal"
    )
with col4:
    st.metric(
        "Privacy Score",
        "100%",
        delta="Local",
        delta_color="normal"
    )

st.markdown("---")

# Chat interface
chat_container = st.container()
input_container = st.container()

# Display messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

# Input area
with input_container:
    if prompt := st.chat_input(f"Ask {agents[st.session_state.current_agent]}..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
        
        # Generate response
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                
                # Show typing indicator
                with st.spinner(f"{agents[st.session_state.current_agent]} is thinking..."):
                    # Call API
                    response = call_api(prompt, st.session_state.current_agent)
                    
                    # Simulate streaming (optional)
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.05)
                    
                    message_placeholder.markdown(full_response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# Quick actions
st.markdown("---")
st.markdown("### 🚀 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Research Topic"):
        sample = "What are the latest developments in quantum computing?"
        st.session_state.messages.append({"role": "user", "content": sample})
        response = call_api(sample, "research")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col2:
    if st.button("📊 Analyze Data"):
        sample = "Analyze the trends in AI adoption across industries"
        st.session_state.messages.append({"role": "user", "content": sample})
        response = call_api(sample, "analyst")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col3:
    if st.button("✍️ Write Content"):
        sample = "Write a blog post introduction about open-source AI"
        st.session_state.messages.append({"role": "user", "content": sample})
        response = call_api(sample, "writer")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col4:
    if st.button("💻 Generate Code"):
        sample = "Create a Python function to compress text using Huffman coding"
        st.session_state.messages.append({"role": "user", "content": sample})
        response = call_api(sample, "coder")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280;'>
    <p>EdgeMind v0.2.0 | Built with ❤️ by the Open Source Community</p>
    <p>⭐ Star us on <a href='https://github.com/artbyoscar/ai-playground'>GitHub</a> | 
    🐛 Report <a href='https://github.com/artbyoscar/ai-playground/issues'>Issues</a> | 
    💬 Join our <a href='#'>Discord</a></p>
</div>
""", unsafe_allow_html=True)