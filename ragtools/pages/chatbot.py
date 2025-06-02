"""
Chatbot page for RAG assistant
"""
import streamlit as st
import json
import os
from datetime import datetime
import time

# Constants
LOGS_FOLDER = 'chat_logs'
LOG_FILE = os.path.join(LOGS_FOLDER, 'chat_log.txt')
CONTEXT_FILE = os.path.join(LOGS_FOLDER, 'context.json')


# Utils
def ensure_logs_folder_exists():
    """Ensure the chat logs folder exists, creating it if necessary."""
    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)


def load_context():
    """Load conversation context from the file if it exists."""
    ensure_logs_folder_exists()
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, 'r') as f:
            print(f'Loading context from {CONTEXT_FILE}')
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
    Send a prompt to the language model with retrieved context from the vector store
    """
    print(f"Prompt: \n{prompt}\n\n")
    print(f"model_context: \n{model_context}\n\n")

    try:
        from ragtools.import_rag_data import get_retriever
        from langchain_community.llms import Ollama
        from langchain.chains.llm import LLMChain
        from langchain.prompts import PromptTemplate
        
        # Get the retriever from the vector store
        knowledge_dir = st.session_state.knowledge_dir
        db_dir = st.session_state.db_dir
        retriever = get_retriever(knowledge_dir, db_dir)
        
        if not retriever:
            return f"Error: Couldn't initialize retriever. Please check if your knowledge base is properly set up.", model_context
        
        # Create a custom prompt template that includes the system context
        template = """
        {system_instructions}
        
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Create the prompt with the system context
        prompt_template = PromptTemplate(
            input_variables=["system_instructions", "context", "question"],
            template=template
        )

        print(f"prompt_template: \n{prompt_template}\n\n")
        
        # Initialize the LLM
        llm = Ollama(model=st.session_state.llm, temperature=float(st.session_state.temperature))
        
        # Create a basic LLMChain
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_template,
            verbose=True
        )
        
        # Get relevant documents first
        docs = retriever.get_relevant_documents(prompt)
        print(f"docs: \n{docs}\n\n")
        context_text = "\n\n".join(doc.page_content for doc in docs)
        
        # Pass all required variables directly to the LLMChain
        response = llm_chain.invoke({
            "system_instructions": system_context,
            "context": context_text,
            "question": prompt
        })
        print(f"response: \n{response}\n\n")
        # Extract the answer from the response
        answer = response.get("text", "No answer generated")
        print(f"answer: \n{answer}\n\n")
        return answer, model_context
        
    except Exception as e:
        error_message = f"Error processing your request: {str(e)}"
        import traceback
        traceback.print_exc()
        return error_message, model_context


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

    # Check if system_prompt.txt exists in the project root and load it
    default_system_context = "You are a helpful RAG assistant. Answer questions based on the provided knowledge base."

    # Try to find system_prompt.txt at the project root level
    try:
        # Get the actual project root path
        system_prompt_file = os.path.join(st.session_state.project_dir, "system_prompt.txt")
        print(f"st.session_state.project_dir: {st.session_state.project_dir}")
        print(f"Checking for system prompt at: {system_prompt_file}")
        # print(f"File exists: {os.path.exists(system_prompt_file)}")
        
        if os.path.exists(system_prompt_file):
            try:
                with open(system_prompt_file, 'r') as f:
                    system_context = f.read().strip()
                    st.session_state.system_context = system_context
                    st.session_state.context_input = system_context
                    # print(f"Successfully loaded system prompt from: {system_prompt_file}")
            except Exception as e:
                st.error(f"Error reading system prompt file: {e}")
                st.session_state.system_context = default_system_context
                st.session_state.context_input = default_system_context
        else:
            print(f"System prompt file not found at: {system_prompt_file}")
            st.session_state.system_context = default_system_context
            st.session_state.context_input = default_system_context
    except Exception as e:
        st.error(f"Error setting up system context: {e}")
        if "system_context" not in st.session_state:
            st.session_state.system_context = default_system_context
            st.session_state.context_input = default_system_context

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    # Settings area
    with st.expander("⚙️ Settings", expanded=False):
        # # Debug information
        # st.write("Debug Info:")
        # st.write(f"Project dir: {st.session_state.get('project_dir', 'Not set')}")
        # st.write(f"Current system_context: {st.session_state.get('system_context', 'Not set')[:100]}...")
        # st.write(f"Current context_input: {st.session_state.get('context_input', 'Not set')[:100]}...")
        
        # Initialize context_input in the session state if it doesn't exist
        if "context_input" not in st.session_state:
            if "system_context" in st.session_state:
                st.session_state.context_input = st.session_state.system_context
            else:
                st.session_state.context_input = default_system_context

        # Use the text area - it will automatically sync with the session state
        context_text = st.text_area(
            "System Context:",
            value=st.session_state.context_input,
            height=300,
            help="Instructions for the AI that apply to all messages.",
            key="context_textarea"  # Use a different key to avoid conflicts
        )

        # Update the Context button to apply changes
        if st.button("Update Context", key="update_context_btn"):
            try:
                # Update session state
                st.session_state.context_input = context_text
                st.session_state.system_context = context_text
                
                # Save to file
                system_prompt_file = os.path.join(st.session_state.project_dir, "system_prompt.txt")
                print(f"Attempting to save to: {system_prompt_file}")
                print(f"Content to save: {context_text[:100]}...")
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(system_prompt_file), exist_ok=True)
                
                with open(system_prompt_file, 'w', encoding='utf-8') as f:
                    f.write(context_text)
                
                # Verify the file was written
                if os.path.exists(system_prompt_file):
                    with open(system_prompt_file, 'r', encoding='utf-8') as f:
                        saved_content = f.read()
                    if saved_content == context_text:
                        st.success("Context updated and saved to system_prompt.txt!")
                        print("File saved and verified successfully")
                    else:
                        st.warning("File saved but content verification failed")
                        print(f"Content mismatch - Expected: {context_text[:50]}..., Got: {saved_content[:50]}...")
                else:
                    st.error("File was not created")
                    print("File was not created")
                    
            except Exception as e:
                st.error(f"Error updating context: {str(e)}")
                print(f"Exception occurred: {e}")
                import traceback
                traceback.print_exc()
            
            # Force a rerun to update the UI
            time.sleep(0.1)  # Small delay to ensure file operations complete
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
            # Add the user message to the history
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