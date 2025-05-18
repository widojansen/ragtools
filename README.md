# RAG Tools

A toolkit for creating Retrieval-Augmented Generation (RAG) applications with a standardized Streamlit interface.

## Overview

RAG Tools provides a simple, consistent framework for building RAG-powered applications. It includes:

- A standardized Streamlit UI for RAG applications
- A modular page system for extending functionality
- A chatbot interface with built-in conversation logging
- Support for integration with any LLM backend

## Installation

### From PyPI (recommended)
```
bash
pip install ragtools
```
### From Source
```
bash
git clone https://github.com/widojansen/ragtools.git
cd ragtools
pip install -e .
```
## Quick Start

### Basic Usage

Create a simple RAG application with just a few lines of code:
```
python
from ragtools.streamlit_ui import launch_streamlit_ui

# Launch with default configuration
launch_streamlit_ui()
```
Run your application with:
```
bash
streamlit run your_app.py
```
### Customizing the UI

You can customize the UI by passing a configuration dictionary:
```
python
from ragtools.streamlit_ui import launch_streamlit_ui

# Custom configuration
config = {
    "project_name": "My RAG App",
    "page_title": "Document Assistant",
    "page_icon": "📚",
    "input_field_placeholder": "Ask about your documents..."
}

# Launch with custom configuration
launch_streamlit_ui(config)
```
## Building a Complete RAG Application

Here's a more comprehensive example of building a RAG application with RAG Tools and LangChain:
```
python
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from ragtools.streamlit_ui import launch_streamlit_ui

# Set up your embeddings and LLM
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Load documents (example: load PDF files from a directory)
loader = DirectoryLoader("./documents/", glob="**/*.pdf")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
doc_chunks = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(documents=doc_chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Custom implementation of ask_llm to replace the placeholder in chatbot.py
def custom_ask_llm(prompt, system_context="", model_context=None):
    """Custom LLM integration using LangChain."""
    # Combine system context with user prompt if provided
    if system_context:
        full_prompt = f"{system_context}\n\nQuestion: {prompt}"
    else:
        full_prompt = prompt
        
    # Query the RAG chain
    result = rag_chain({"query": full_prompt})
    
    # Format the response
    response = result["result"]
    
    # You could save the sources in the context for later reference
    if model_context is None:
        model_context = {}
    
    model_context["source_documents"] = [
        doc.page_content for doc in result.get("source_documents", [])
    ]
    
    return response, model_context

# Patch the ask_llm function in chatbot.py
import ragtools.pages.chatbot
ragtools.pages.chatbot.ask_llm = custom_ask_llm

# Configure and launch the UI
config = {
    "project_name": "Document Q&A",
    "page_title": "Document Assistant",
    "page_icon": "📚",
    "input_field_placeholder": "Ask questions about your documents..."
}

# Launch the RAG application
launch_streamlit_ui(config)
```
## Extending with Custom Pages

You can extend RAG Tools with custom pages:

1. Create a new Python file in the `pages` directory, for example `ragtools/pages/document_uploader.py`
2. Implement a `run()` function in your page module
3. Add a method to display the page in `RagStreamlitUI` class
4. Add a button in the sidebar to navigate to your new page

Example of a custom page:
```
python
# ragtools/pages/document_uploader.py
import streamlit as st

def run():
    st.title("Document Uploader")
    
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        # Process the uploaded file
        st.success(f"Successfully uploaded {uploaded_file.name}")
        
        # Add your document processing logic here
        # For example, extracting text, chunking, and adding to vector store
```
Then modify `streamlit_ui.py` to include your new page:
```
python
# Add to RagStreamlitUI.run() method
with st.sidebar:
    # ...existing code...
    if st.button("📄 Upload Documents", use_container_width=True):
        st.session_state.current_page = "document_uploader"
        st.rerun()

# Add a new display method
def _display_document_uploader_page(self):
    """Display the Document Uploader page"""
    try:
        from .pages import document_uploader
        document_uploader.run()
    except ImportError as e:
        st.error(f"Error loading document uploader: {str(e)}")
    except Exception as e:
        st.error(f"Error displaying document uploader: {str(e)}")

# Modify the page display logic
if st.session_state.current_page == "chatbot":
    self._display_chatbot_page()
elif st.session_state.current_page == "document_uploader":
    self._display_document_uploader_page()
```
## Integration with Different LLM Backends

RAG Tools is designed to work with any LLM backend. Here are examples of how to integrate with popular options:

### Using OpenAI
```
python
import openai

def custom_ask_llm(prompt, system_context="", model_context=None):
    messages = []
    
    # Add system message if provided
    if system_context:
        messages.append({"role": "system", "content": system_context})
    
    # Add user message
    messages.append({"role": "user", "content": prompt})
    
    # Make API call to OpenAI
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=st.session_state.get("temperature", 0.7)
    )
    
    # Extract the response text
    response_text = response.choices[0].message.content
    
    # You could store conversation history in model_context
    return response_text, model_context
```
### Using Ollama
```
python
import requests

def custom_ask_llm(prompt, system_context="", model_context=None):
    # Prepare the prompt with system context if provided
    full_prompt = prompt
    if system_context:
        full_prompt = f"### System:\n{system_context}\n\n### User:\n{prompt}"
    
    payload = {
        'model': 'llama3.1',
        'prompt': full_prompt,
        'stream': False,
        'options': {
            'temperature': st.session_state.get("temperature", 0.7)
        }
    }
    
    if model_context:
        payload['context'] = model_context
    
    # Make request to Ollama API
    response = requests.post('http://localhost:11434/api/generate', json=payload)
    data = response.json()
    
    return data['response'], data.get('context')
```
## Configuration Options

RAG Tools supports the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `project_name` | Name of the project displayed in the sidebar | "RAG Tools" |
| `page_title` | Title displayed in the browser tab | "RAG Assistant" |
| `page_icon` | Icon displayed in the browser tab | "🧠" |
| `input_field_label` | Label for the input field | "Enter your query" |
| `input_field_default` | Default text for the input field | "" |
| `input_field_placeholder` | Placeholder text for the input field | "Ask me anything..." |
| `input_field_help` | Help text for the input field | "Type your question here and press Enter" |

## Dependencies

RAG Tools requires the following Python packages:
- Python ≥ 3.12
- streamlit
- psutil

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
```
