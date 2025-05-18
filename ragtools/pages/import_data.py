"""
Import Data page for RAG applications
Provides a UI for importing, processing, and indexing documents for RAG
"""
import os
import time
import shutil
import tempfile
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import subprocess

# Import necessary LangChain components if available
try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    import chromadb
    from langchain_chroma import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


def ensure_directory(directory: str) -> None:
    """Ensure a directory exists, creating it if necessary"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_file_metadata(file_path: str, filename: str) -> Dict[str, str]:
    """
    Extract and create metadata for a file
    
    Args:
        file_path: Path to the file
        filename: Name of the file
        
    Returns:
        Dictionary containing metadata
    """
    # Get file stats for metadata
    file_stats = os.stat(file_path)
    last_modified = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d')

    # Extract a title from the filename (remove extension and replace underscores)
    title = os.path.splitext(filename)[0].replace('_', ' ').title()

    # Determine file type
    file_extension = os.path.splitext(filename)[1].lower()

    # Create metadata dictionary
    metadata = {
        "source": filename,
        "title": title,
        "last_modified": last_modified,
        "file_path": file_path,
        "file_type": file_extension,
    }

    return metadata


def check_ollama_availability() -> Tuple[bool, str]:
    """
    Check if Ollama is available and return available models
    
    Returns:
        Tuple of (is_available, model_info)
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return True, result.stdout
    except Exception as e:
        return False, str(e)


def init_ollama_embeddings(model: str, progress_callback=None) -> Optional[Any]:
    """
    Initialize Ollama embeddings with robust error handling
    
    Args:
        model: Name of the embeddings model to use
        progress_callback: Optional callback for progress updates
        
    Returns:
        Embeddings object or None if failed
    """
    if not LANGCHAIN_AVAILABLE:
        if progress_callback:
            progress_callback("LangChain not available. Please install required packages.")
        return None
        
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            if progress_callback:
                progress_callback(f"Initializing embeddings model (attempt {attempt + 1}/{max_retries})...")
            embeddings = OllamaEmbeddings(model=model)

            # Test the embeddings with a sample text
            test_text = "This is a test."
            test_embedding = embeddings.embed_query(test_text)
            if test_embedding and len(test_embedding) > 0:
                if progress_callback:
                    progress_callback(f"Successfully connected to Ollama with embeddings model")
                return embeddings
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error initializing embeddings (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                if progress_callback:
                    progress_callback(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    return None


def process_documents(
    files, 
    chunk_size: int, 
    chunk_overlap: int,
    embedding_model: str,
    db_path: str,
    progress_callback=None
) -> Optional[Any]:
    """
    Process documents for RAG
    
    Args:
        files: List of uploaded files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        embedding_model: Model to use for embeddings
        db_path: Path to store the vector database
        progress_callback: Optional callback for progress updates
        
    Returns:
        Vectorstore or None if processing failed
    """
    if not LANGCHAIN_AVAILABLE:
        if progress_callback:
            progress_callback("LangChain not available. Please install required packages.")
        return None
        
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files to temporary directory
        for uploaded_file in files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Create metadata for this file
            metadata = create_file_metadata(file_path, uploaded_file.name)
            
            # Process based on file type
            if uploaded_file.name.endswith(".txt"):
                try:
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    
                    # Apply metadata to each document
                    for doc in docs:
                        doc.metadata.update(metadata)
                        
                    documents.extend(docs)
                    if progress_callback:
                        progress_callback(f"Loaded text file: {uploaded_file.name}")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error processing text file {uploaded_file.name}: {str(e)}")
                    
            elif uploaded_file.name.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    # Apply metadata to each document (with page numbers)
                    for doc in docs:
                        doc.metadata.update(metadata)
                        
                    documents.extend(docs)
                    if progress_callback:
                        progress_callback(f"Loaded PDF file: {uploaded_file.name} ({len(docs)} pages)")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error processing PDF {uploaded_file.name}: {str(e)}")
        
        if not documents:
            if progress_callback:
                progress_callback("No documents were successfully loaded.")
            return None
            
        if progress_callback:
            progress_callback(f"Loaded {len(documents)} documents with metadata")
            
        # Split text into chunks while preserving metadata
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        if progress_callback:
            progress_callback(f"Created {len(chunks)} chunks with preserved metadata")
            
        # Initialize embeddings
        embeddings = init_ollama_embeddings(embedding_model, progress_callback)
        if not embeddings:
            if progress_callback:
                progress_callback("Failed to initialize embeddings. Please check Ollama is running.")
            return None
            
        # Create vector store
        ensure_directory(db_path)
        
        # If the database already exists, remove it to start fresh
        if os.path.exists(db_path):
            if progress_callback:
                progress_callback(f"Removing existing database at {db_path}")
            shutil.rmtree(db_path)
            time.sleep(1)  # Give OS time to complete deletion
            
        # Create a new ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        
        # Create a collection
        collection = client.create_collection("document_chunks")
        
        # Process in small batches
        batch_size = 5
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) - 1) // batch_size + 1
            
            if progress_callback:
                progress_callback(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                
            try:
                # Prepare data for ChromaDB
                ids = [f"doc_{i + j}" for j in range(len(batch))]
                texts = [doc.page_content for doc in batch]
                
                # Convert metadata to format compatible with ChromaDB
                metadatas = []
                for doc in batch:
                    # ChromaDB requires metadata to be simple types
                    meta = {k: str(v) for k, v in doc.metadata.items()}
                    metadatas.append(meta)
                    
                # Generate embeddings
                if progress_callback:
                    progress_callback(f"Generating embeddings for batch {batch_num}...")
                try:
                    embs = embeddings.embed_documents(texts)
                    
                    # Add to collection
                    collection.add(
                        ids=ids,
                        embeddings=embs,
                        documents=texts,
                        metadatas=metadatas
                    )
                    if progress_callback:
                        progress_callback(f"Successfully added batch {batch_num} to ChromaDB")
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error generating embeddings: {str(e)}")
                    # Try one at a time
                    if progress_callback:
                        progress_callback(f"Trying one document at a time...")
                    for j, doc in enumerate(batch):
                        try:
                            doc_id = f"doc_{i + j}"
                            doc_text = doc.page_content
                            doc_meta = {k: str(v) for k, v in doc.metadata.items()}
                            
                            emb = embeddings.embed_documents([doc_text])[0]
                            collection.add(
                                ids=[doc_id],
                                embeddings=[emb],
                                documents=[doc_text],
                                metadatas=[doc_meta]
                            )
                            if progress_callback:
                                progress_callback(f"Added document {doc_id}")
                        except Exception as e2:
                            if progress_callback:
                                progress_callback(f"Failed to add document {j}: {str(e2)}")
                            continue
                            
                # Pause to let Ollama recover
                time.sleep(1.0)
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error in batch {batch_num}: {str(e)}")
                    progress_callback("Pausing for 5 seconds to let Ollama recover...")
                time.sleep(5)
                continue
                
        # Create Langchain wrapper over the ChromaDB collection
        vectorstore = Chroma(
            client=client,
            collection_name="document_chunks",
            embedding_function=embeddings
        )
        
        if progress_callback:
            progress_callback("Vector store created successfully!")
            
        return vectorstore
        
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error processing documents: {str(e)}")
        return None
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def run():
    """Run the data import interface"""
    
    # Page header
    st.markdown("<h1 style='font-size:1.5rem;'>📥 Import RAG Data</h1>", unsafe_allow_html=True)
    st.markdown("<p>Import and process documents for your RAG application.</p>", unsafe_allow_html=True)
    
    # Check if LangChain is available
    if not LANGCHAIN_AVAILABLE:
        st.error(
            "LangChain is not available. Please install the required packages:\n\n"
            "```bash\npip install langchain langchain-community langchain-ollama chromadb\n```"
        )
        return
        
    # Initialize session state for settings
    if "knowledge_dir" not in st.session_state:
        st.session_state.knowledge_dir = "./knowledge/"
    
    if "db_path" not in st.session_state:
        st.session_state.db_path = "./chroma_db"
        
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
        
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200
        
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "granite-embedding:278m"
        
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "qwen3:8b"
        
    if "progress_messages" not in st.session_state:
        st.session_state.progress_messages = []
        
    # Check Ollama availability
    ollama_available, ollama_models = check_ollama_availability()
    
    # Settings section
    with st.expander("⚙️ RAG Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Knowledge Directory", 
                          value=st.session_state.knowledge_dir,
                          key="knowledge_dir_input",
                          help="Directory to store knowledge base files")
            
            st.text_input("Vector Database Path", 
                          value=st.session_state.db_path,
                          key="db_path_input",
                          help="Directory to store the ChromaDB vector database")
            
            st.selectbox("Embedding Model",
                         options=["granite-embedding:278m", "nomic-embed-text", "all-minilm-l6-v2"],
                         index=0,
                         key="embedding_model_input",
                         help="Model to use for generating embeddings")
                         
        with col2:
            st.slider("Chunk Size", 
                      min_value=100, 
                      max_value=2000, 
                      value=st.session_state.chunk_size,
                      step=50,
                      key="chunk_size_input",
                      help="Size of text chunks for processing")
                      
            st.slider("Chunk Overlap", 
                      min_value=0, 
                      max_value=500, 
                      value=st.session_state.chunk_overlap,
                      step=10,
                      key="chunk_overlap_input",
                      help="Overlap between text chunks")
                      
            st.selectbox("LLM Model",
                         options=["qwen3:8b", "llama3.1", "mixtral", "phi3:mini"],
                         index=0,
                         key="llm_model_input",
                         help="Language model for querying the knowledge base")
        
        # Save settings button
        if st.button("Save Settings"):
            st.session_state.knowledge_dir = st.session_state.knowledge_dir_input
            st.session_state.db_path = st.session_state.db_path_input
            st.session_state.chunk_size = st.session_state.chunk_size_input
            st.session_state.chunk_overlap = st.session_state.chunk_overlap_input
            st.session_state.embedding_model = st.session_state.embedding_model_input
            st.session_state.llm_model = st.session_state.llm_model_input
            st.success("Settings saved!")
    
    # Ollama status
    st.subheader("Ollama Status")
    if ollama_available:
        st.success("✅ Ollama is running")
        
        # Display available models
        with st.expander("Available Ollama Models", expanded=False):
            st.code(ollama_models)
            
        # Check if required models are available
        if st.session_state.embedding_model not in ollama_models:
            st.warning(f"⚠️ Embedding model '{st.session_state.embedding_model}' is not available.")
            st.code(f"ollama pull {st.session_state.embedding_model}")
            
        if st.session_state.llm_model not in ollama_models:
            st.warning(f"⚠️ LLM model '{st.session_state.llm_model}' is not available.")
            st.code(f"ollama pull {st.session_state.llm_model}")
    else:
        st.error(
            "❌ Ollama is not running or not installed. Please install and start Ollama:\n\n"
            "1. Install: https://ollama.com/download\n"
            "2. Start the Ollama service\n"
            "3. Pull required models\n"
        )
        st.code("ollama pull granite-embedding:278m\nollama pull qwen3:8b")
    
    # File uploader
    st.subheader("Import Documents")
    uploaded_files = st.file_uploader(
        "Upload documents", 
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    # Display uploaded files
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Show file details in a table
        file_details = []
        for file in uploaded_files:
            file_details.append({
                "Name": file.name,
                "Type": file.type,
                "Size": f"{round(file.size / 1024, 2)} KB"
            })
            
        st.table(file_details)
        
        # Process button
        if st.button("Process Documents", type="primary", disabled=not ollama_available):
            # Clear previous progress messages
            st.session_state.progress_messages = []
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a placeholder for progress messages
            progress_container = st.container()
            
            # Callback for progress updates
            def update_progress(message):
                st.session_state.progress_messages.append(message)
                status_text.text(message)
                # Update progress container
                with progress_container:
                    for msg in st.session_state.progress_messages:
                        st.text(msg)
            
            # Process documents
            with st.spinner("Processing documents..."):
                vectorstore = process_documents(
                    files=uploaded_files,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                    embedding_model=st.session_state.embedding_model,
                    db_path=st.session_state.db_path,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(100)
                
                if vectorstore:
                    st.success("✅ Documents processed and vector store created successfully!")
                    
                    # Show stats
                    st.subheader("Vector Store Statistics")
                    st.json({
                        "Documents": len(uploaded_files),
                        "Chunks": vectorstore._collection.count(),
                        "Database Path": st.session_state.db_path,
                        "Embedding Model": st.session_state.embedding_model
                    })
                else:
                    st.error("❌ Failed to process documents and create vector store.")
    
    # URL input
    st.subheader("Import from URL")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input("Enter URL to import content from:")
    
    with col2:
        if st.button("Import from URL", disabled=True):  # Currently disabled as not implemented
            st.info("URL import functionality is not implemented yet.")
    
    st.info("Note: URL import functionality will be available in a future update.")
    
    # Display information about the RAG system
    st.subheader("RAG System Information")
    
    # Check if vector store exists
    vector_store_exists = os.path.exists(st.session_state.db_path)
    
    if vector_store_exists:
        try:
            client = chromadb.PersistentClient(path=st.session_state.db_path)
            collection = client.get_collection("document_chunks")
            document_count = collection.count()
            
            st.success(f"✅ Vector store found at {st.session_state.db_path}")
            st.json({
                "Collection": "document_chunks",
                "Document Count": document_count,
                "Database Path": st.session_state.db_path
            })
        except Exception as e:
            st.warning(f"⚠️ Vector store exists but could not be accessed: {str(e)}")
    else:
        st.warning(f"⚠️ No vector store found at {st.session_state.db_path}")
        st.info("Upload and process documents to create a vector store.")

    # Add information about other functionalities
    st.markdown("---")
    st.markdown("""
    ### Next Steps
    
    After importing your documents:
    
    1. Navigate to the **Chatbot** page to query your knowledge base
    2. The RAG system will retrieve relevant context from your documents and generate answers
    
    For more information on RAG systems, see the [LangChain RAG documentation](https://python.langchain.com/docs/use_cases/question_answering/).
    """)


if __name__ == "__main__":
    run()