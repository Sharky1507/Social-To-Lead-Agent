"""
Streamlit UI for the AutoStream AI Agent.
"""
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import create_agent, run_conversation
from agent.state import ConversationState


# Load environment variables
load_dotenv()


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "agent" not in st.session_state:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.session_state.agent = create_agent(api_key)
        else:
            st.session_state.agent = None
    
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = {
            "messages": [],
            "intent": "unknown",
            "lead_info": {},
            "lead_captured": False,
            "response": ""
        }
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def main():
    # Page configuration
    st.set_page_config(
        page_title="AutoStream AI Assistant",
        page_icon="🎬",
        layout="centered"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .lead-captured {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🎬 AutoStream AI Assistant")
    st.markdown("*Your intelligent guide to automated video editing*")
    st.divider()
    
    # Sidebar with info
    with st.sidebar:
        st.header("📊 Conversation Status")
        
        # Show current intent
        intent = st.session_state.conversation_state.get("intent", "unknown")
        intent_emoji = {
            "greeting": "👋",
            "inquiry": "❓",
            "high_intent": "🎯",
            "unknown": "🔍"
        }
        st.metric("Current Intent", f"{intent_emoji.get(intent, '🔍')} {intent.title()}")
        
        # Show lead info if any
        lead_info = st.session_state.conversation_state.get("lead_info", {})
        if lead_info:
            st.subheader("📋 Lead Information")
            st.write(f"**Name:** {lead_info.get('name', 'Not provided')}")
            st.write(f"**Email:** {lead_info.get('email', 'Not provided')}")
            st.write(f"**Platform:** {lead_info.get('platform', 'Not provided')}")
        
        # Show lead captured status
        if st.session_state.conversation_state.get("lead_captured", False):
            st.success("✅ Lead Captured!")
        
        st.divider()
        
        # API Key input
        st.subheader("⚙️ Configuration")
        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Enter your Gemini API key"
        )
        
        if st.button("Update API Key"):
            if api_key_input:
                os.environ["GOOGLE_API_KEY"] = api_key_input
                st.session_state.agent = create_agent(api_key_input)
                st.success("API Key updated!")
                st.rerun()
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset Conversation"):
            st.session_state.conversation_state = {
                "messages": [],
                "intent": "unknown",
                "lead_info": {},
                "lead_captured": False,
                "response": ""
            }
            st.session_state.chat_history = []
            st.rerun()
    
    # Check for API key
    if not st.session_state.agent:
        st.warning("⚠️ Please configure your Google API Key in the sidebar to start chatting.")
        st.info("""
        **How to get an API key:**
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Paste it in the sidebar configuration
        """)
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    state, response = run_conversation(
                        st.session_state.agent,
                        st.session_state.conversation_state,
                        prompt
                    )
                    st.session_state.conversation_state = state
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Show lead captured notification
                    if state.get("lead_captured", False) and len([m for m in st.session_state.chat_history if "Lead captured" in m.get("content", "")]) == 0:
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.rerun()
    
    # Welcome message if no chat history
    if not st.session_state.chat_history:
        st.info("""
        👋 **Welcome to AutoStream!**
        
        I'm your AI assistant. I can help you with:
        - 💰 **Pricing information** - Learn about our Basic and Pro plans
        - 🎥 **Features** - Discover our video editing capabilities
        - 📋 **Policies** - Refund and support information
        - 🚀 **Get Started** - Sign up for AutoStream
        
        *Just type your question below to get started!*
        """)


if __name__ == "__main__":
    main()
