# web/streamlit_app.py
import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.working_ai_playground import AIPlayground

# Page config
st.set_page_config(
    page_title="🎮 AI Playground",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'ai_playground' not in st.session_state:
    st.session_state.ai_playground = AIPlayground()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main header
st.title("🎮 AI Playground")
st.markdown("**Powered by Together.ai + Mixtral-8x7B** • Your personal AI development environment")

# Sidebar
with st.sidebar:
    st.header("🛠️ AI Tools")
    
    tool_mode = st.selectbox(
        "Choose AI Assistant:",
        ["💬 General Chat", "💻 Code Assistant", "✍️ Content Writer", "📈 Business Advisor"]
    )
    
    st.divider()
    
    # Usage stats
    if st.button("📊 Show Stats"):
        stats = st.session_state.ai_playground.get_stats()
        for key, value in stats.items():
            st.metric(key.replace('_', ' ').title(), value)
    
    st.divider()
    
    # Save conversation
    if st.button("💾 Save Conversation"):
        if st.session_state.ai_playground.conversation_log:
            filename = st.session_state.ai_playground.save_conversation()
            st.success(f"Saved: {filename}")
        else:
            st.warning("No conversation to save yet!")
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.ai_playground.conversation_log = []
        st.success("Chat cleared!")

# Main chat interface
st.header(f"{tool_mode}")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        st.write(chat["ai"])

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message to chat
    st.session_state.chat_history.append({"user": user_input, "ai": ""})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get AI response based on selected tool
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            if tool_mode == "💬 General Chat":
                response = st.session_state.ai_playground.chat(user_input)
            elif tool_mode == "💻 Code Assistant":
                response = st.session_state.ai_playground.code_assistant(user_input)
            elif tool_mode == "✍️ Content Writer":
                response = st.session_state.ai_playground.content_writer(user_input)
            elif tool_mode == "📈 Business Advisor":
                response = st.session_state.ai_playground.business_advisor(user_input)
            
            st.write(response)
            
            # Update chat history
            st.session_state.chat_history[-1]["ai"] = response

# Quick examples section
st.divider()
st.header("🚀 Quick Examples")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💬 Say Hello", key="hello"):
        example_input = "Hello! I'm building an AI playground. What can you help me with?"
        st.session_state.chat_history.append({"user": example_input, "ai": ""})
        response = st.session_state.ai_playground.chat(example_input)
        st.session_state.chat_history[-1]["ai"] = response
        st.rerun()

with col2:
    if st.button("💻 Code Example", key="code"):
        example_input = "Write a Python function to fetch data from an API with error handling"
        st.session_state.chat_history.append({"user": example_input, "ai": ""})
        response = st.session_state.ai_playground.code_assistant(example_input)
        st.session_state.chat_history[-1]["ai"] = response
        st.rerun()

with col3:
    if st.button("✍️ Content Example", key="content"):
        example_input = "Write a compelling product description for an AI-powered productivity app"
        st.session_state.chat_history.append({"user": example_input, "ai": ""})
        response = st.session_state.ai_playground.content_writer(example_input)
        st.session_state.chat_history[-1]["ai"] = response
        st.rerun()

with col4:
    if st.button("📈 Business Example", key="business"):
        example_input = "I'm a solo developer. How can I monetize my AI skills quickly?"
        st.session_state.chat_history.append({"user": example_input, "ai": ""})
        response = st.session_state.ai_playground.business_advisor(example_input)
        st.session_state.chat_history[-1]["ai"] = response
        st.rerun()

# Footer with system status
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 AI Model", "Mixtral-8x7B")

with col2:
    st.metric("🔌 API Status", "🟢 Connected")

with col3:
    conversations = len(st.session_state.ai_playground.conversation_log)
    st.metric("💬 Conversations", conversations)

# Hidden debug info
with st.expander("🔧 Debug Info"):
    st.json({
        "model": st.session_state.ai_playground.model,
        "api_url": st.session_state.ai_playground.url,
        "total_conversations": len(st.session_state.ai_playground.conversation_log),
        "session_chats": len(st.session_state.chat_history)
    })