"""
Chatbot page for RAG assistant
"""
import streamlit as st
import json
import os
from datetime import datetime

# Constants
MODEL = 'llama3.1'  # Default model
LOGS_FOLDER = 'chat_logs'
LOG_FILE = os.path.join(LOGS_FOLDER, 'chat_log.txt')
CONTEXT_FILE = os.path.join(LOGS_FOLDER, 'context.json')


# Utils
def ensure_logs_folder_exists():
    """Ensure the chat logs folder exists, creating it if necessary."""
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)


def load_context():
    """Load conversation context from file if it exists."""
    ensure_logs_folder_exists()
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, 'r') as f:
            return json.load(f)
    return None


def save_context(context):
    """Save conversation context to file."""
    ensure_logs_folder_exists()
    with open(CONTEXT_FILE, 'w') as f:
        json.dump(context, f)


def log_chat(user, bot):
    """Log the conversation to a file."""
    ensure_logs_folder_exists()
    with open(LOG_FILE, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] You: {user}\n")
        f.write(f"[{timestamp}] Bot: {bot}\n\n")


def ask_llm(prompt, system_context="", model_context=None):
    """
    Send a prompt to the language model with optional system context and model context
    
    NOTE: THIS IS A PLACEHOLDER FUNCTION. You must replace it with your own LLM integration
    before using this application. See the README.md for examples of integrating with
    different LLM backends (OpenAI, Ollama, etc.).

    Args:
        prompt: The user's message
        system_context: System instructions to prepend to the message
        model_context: The context for conversation history

    Returns:
        Tuple of (response, new_context)
    """
    # This is a placeholder function that should be replaced with your LLM implementation
    response = (
        "⚠️ This is a placeholder response. You need to implement LLM integration.\n\n"
        f"You asked: {prompt}\n\n"
        "See README.md for examples of how to integrate with OpenAI, Ollama, or other LLM backends."
    )
    return response, model_context


def run():
    """Run the chatbot interface."""

    # Add custom CSS for a cleaner UI
    st.markdown("""
    <style>
        /* Reduce space between elements */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0;
            max-width: 95%;
        }

        /* Styling for chat messages */
        .chat-message {
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #f0f2f6;
        }
        .chat-message.bot {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
        }
        .chat-message.system {
            background-color: #f0f8ff;
            border: 1px dashed #b0c4de;
        }

        /* Chat container */
        .chat-container {
            display: flex;
            flex-direction: column;
            max-height: 60vh;
            height: auto;
            overflow-y: auto;
            padding: 0.5rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .message-header {
            font-weight: bold;
            margin-bottom: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Ensure the logs folder exists when the app starts
    ensure_logs_folder_exists()

    # Page header
    st.markdown("<h1 style='font-size:1.5rem;'>💬 RAG Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p>Ask questions about your data with this RAG-powered chatbot.</p>", unsafe_allow_html=True)

    # Initialize session state
    if "model_context" not in st.session_state:
        st.session_state.model_context = load_context()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check if system_prompt.txt exists in project root and load it
    default_system_context = "You are a helpful RAG assistant. Answer questions based on the provided knowledge base."
    
    # Try to find system_prompt.txt at the project root level
    try:
        # Get the root project directory (BraceFaceRag)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        system_prompt_file = os.path.join(project_root, "system_prompt.txt")
        
        if "system_context" not in st.session_state:
            if os.path.exists(system_prompt_file):
                try:
                    with open(system_prompt_file, 'r') as f:
                        st.session_state.system_context = f.read().strip()
                except Exception as e:
                    st.error(f"Error reading system prompt file: {e}")
                    st.session_state.system_context = default_system_context
            else:
                st.session_state.system_context = default_system_context
    except Exception as e:
        st.error(f"Error setting up system context: {e}")
        if "system_context" not in st.session_state:
            st.session_state.system_context = default_system_context

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    # Settings area
    with st.expander("⚙️ Settings", expanded=False):
        # System Context
        st.text_area(
            "System Context:",
            value=st.session_state.system_context,
            height=100,
            key="context_input",
            help="Instructions for the AI that apply to all messages."
        )

        if st.button("Update Context"):
            st.session_state.system_context = st.session_state.context_input
            st.rerun()

        # Temperature Control
        st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            format="%.1f",
            key="temperature",
            help="Higher values make responses more random/creative."
        )

    # Function to handle message submission
    def submit_message():
        user_input = st.session_state[input_key].strip()

        if user_input:
            # Add user message to history
            st.session_state.messages.append({"role": "You", "content": user_input})

            # Get the bot response
            with st.spinner("Thinking..."):
                response, new_context = ask_llm(
                    prompt=user_input,
                    system_context=st.session_state.system_context,
                    model_context=st.session_state.model_context
                )
                st.session_state.model_context = new_context
                save_context(new_context)
                log_chat(user_input, response)

            # Add bot response to history
            st.session_state.messages.append({"role": "Bot", "content": response.strip()})

            # Clear the input field
            st.session_state.input_counter += 1

            # Trigger UI refresh
            st.rerun()

    # Function to clear chat history
    def clear_chat():
        st.session_state.messages = []
        st.session_state.model_context = None
        save_context(None)
        st.rerun()

    # Chat display area
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown(
                "<div style='text-align:center;color:#808080;padding:10px;'>Start a conversation by typing a message below.</div>",
                unsafe_allow_html=True)

        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]

            if role == "You":
                st.markdown(f'''
                <div class="chat-message user">
                    <div class="message-header">👤 {role}</div>
                    <div>{content}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message bot">
                    <div class="message-header">🤖 {role}</div>
                    <div>{content}</div>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    input_container = st.container()
    with input_container:
        # Create a dynamic key
        input_key = f"user_input_{st.session_state.input_counter}"

        # Message form
        with st.form(key="message_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([6, 1, 1])

            with col1:
                st.text_input(
                    "Ask your data...",  # Use this as the label
                    placeholder="Ask anything about your data...",
                    key=input_key,
                    label_visibility="collapsed"  # This will hide the label
                )

            with col2:
                submit_button = st.form_submit_button("Send", use_container_width=True)

            with col3:
                clear_button = st.form_submit_button("Clear", use_container_width=True)

        # Process form submissions
        if submit_button:
            submit_message()

        if clear_button:
            clear_chat()