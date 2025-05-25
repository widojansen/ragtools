import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

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

# Updated imports - remove HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global variable to store vectorstore path
_db_path = "./chroma_db"

def create_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Create metadata for a file based on its path.
    """
    try:
        file_path = Path(file_path)
        filename = file_path.name
        
        return {
            "source": str(file_path),
            "filename": filename,
            "file_type": file_path.suffix.lower(),
            "created_at": time.time(),
        }
    except Exception as e:
        logger.error(f"Error creating metadata: {e}")
        return {"source": str(file_path)}

def get_files_hash(directory: str) -> str:
    """
    Generate a hash of all PDF files in the directory to detect changes.
    """
    hash_obj = hashlib.md5()
    
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return ""
            
        pdf_files = sorted([f for f in os.listdir(directory) if f.lower().endswith('.pdf')])
        for filename in pdf_files:
            file_path = os.path.join(directory, filename)
            file_stat = os.stat(file_path)
            file_info = f"{filename}:{file_stat.st_size}:{file_stat.st_mtime}"
            hash_obj.update(file_info.encode())
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error generating files hash: {e}")
        return ""

def load_pdf_file(file_path: str) -> List[Document]:
    """Load a single PDF file and add metadata."""
    try:
        logger.info(f"Loading PDF file: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Add metadata to each document
        for doc in docs:
            doc.metadata.update(create_file_metadata(file_path))
        
        logger.info(f"Loaded {len(docs)} pages with metadata")
        return docs
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return []

def init_ollama_embeddings(model: str = "granite-embedding:278m", progress_callback=None) -> Optional[Any]:
    """
    Initialize Ollama embeddings with robust error handling
    
    Args:
        model: Name of the embeddings model to use (defaults to granite-embedding:278m)
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
                    progress_callback(f"Successfully connected to Ollama with embeddings model {model}")
                return embeddings
            logger.info(f"Successfully initialized Ollama embeddings with model {model}")
            return embeddings
        except Exception as e:
            logger.error(f"Error initializing embeddings (attempt {attempt + 1}): {str(e)}")
            
            # Check for specific error messages
            error_str = str(e).lower()
            if "connection refused" in error_str:
                logger.error("Ollama server appears to be down. Please ensure it's running.")
            elif "internal server error" in error_str:
                logger.error("Ollama server returned an internal error. Consider restarting Ollama.")
                
            if progress_callback:
                progress_callback(f"Error initializing embeddings (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                if progress_callback:
                    progress_callback(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    logger.error("Failed to initialize Ollama embeddings after multiple attempts")
    return None

def create_or_load_vectorstore(content_directory: str, db_directory: str = _db_path, force_reload: bool = False):
    """
    Create or load a vector store with document embeddings from the content directory.
    
    Args:
        content_directory: Directory containing the documents to embed
        db_directory: Directory to store the vector database
        force_reload: Force reloading and re-embedding of all documents
        
    Returns:
        A vector store object
    """
    try:
        import os
        import hashlib
        import logging
        from langchain_community.vectorstores import Chroma
        
        # Check if content directory exists
        if not os.path.exists(content_directory):
            raise ValueError(f"Content directory does not exist: {content_directory}")
            
        # Create DB directory if it doesn't exist
        os.makedirs(db_directory, exist_ok=True)
            
        # Get a hash of all files in the content directory
        files_hash = get_files_hash(content_directory)
        hash_file = os.path.join(db_directory, "content_hash.txt")
        
        # Check if we need to reload the vector store
        needs_reload = force_reload
        if not needs_reload and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                stored_hash = f.read().strip()
                if stored_hash != files_hash:
                    logging.info(f"Content hash changed. Rebuilding vector store.")
                    needs_reload = True
        else:
            needs_reload = True
            
        # Initialize embeddings model - now using Ollama
        embed_model = init_ollama_embeddings(model="granite-embedding:278m")
        if not embed_model:
            raise ValueError("Failed to initialize Ollama embeddings. Is Ollama running?")
        
        # Load existing vector store if it exists and we don't need to reload
        if not needs_reload and os.path.exists(db_directory):
            logging.info(f"Loading existing vector store from {db_directory}")
            vector_store = Chroma(
                persist_directory=db_directory,
                embedding_function=embed_model
            )
            
            # Check if the vector store actually has embeddings
            if vector_store._collection.count() > 0:
                logging.info(f"Successfully loaded vector store with {vector_store._collection.count()} embeddings")
                return vector_store
            else:
                logging.warning("Vector store exists but is empty. Will rebuild.")
                needs_reload = True
                
        # Create new vector store if needed
        if needs_reload:
            logging.info(f"Creating new vector store from content in {content_directory}")
            
            # Process all files in the content directory
            documents = []
            for root, _, files in os.walk(content_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    # Process PDF files
                    if file_ext == '.pdf':
                        try:
                            pdf_docs = load_pdf_file(file_path)
                            documents.extend(pdf_docs)
                        except Exception as e:
                            logging.error(f"Error processing PDF file {file_path}: {str(e)}")
                    
                    # Add support for other file types as needed
                    # elif file_ext == '.docx':
                    #     # Process DOCX files
                    
            if not documents:
                logging.warning(f"No documents found in {content_directory}")
                # Create an empty vector store
                vector_store = Chroma(
                    persist_directory=db_directory,
                    embedding_function=embed_model
                )
            else:
                logging.info(f"Creating embeddings for {len(documents)} documents")
                
                # Create vector store with small batches
                # Create an empty ChromaDB instance first
                vector_store = Chroma.from_documents(
                    documents=[],  # Start with empty documents
                    embedding=embed_model,
                    persist_directory=db_directory
                )
                
                # Add documents in small batches
                batch_size = 1  # Process one document at a time
                for i in range(0, len(documents), batch_size):
                    batch_end = min(i + batch_size, len(documents))
                    batch = documents[i:batch_end]
                    
                    logging.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
                    
                    try:
                        # Add documents to the existing vector store
                        vector_store.add_documents(documents=batch)
                        
                        # Sleep to let Ollama recover
                        time.sleep(5.0)  # Longer pause between batches
                    except Exception as e:
                        logging.error(f"Error processing batch: {e}")
                        # Continue with next batch
                
            # Save the content hash
            with open(hash_file, "w") as f:
                f.write(files_hash)
                
            logging.info(f"Vector store created with {vector_store._collection.count()} embeddings")
            
            return vector_store
            
    except Exception as e:
        logging.error(f"Error creating vector store: {str(e)}")
        raise

def get_retriever(content_directory: str, db_directory: str = _db_path):
    """
    Get a retriever from an existing or newly created vector store.
    This is the main function to call from external applications.
    """
    vectorstore = create_or_load_vectorstore(content_directory, db_directory)
    if vectorstore:
        return vectorstore.as_retriever()
    return None

def import_rag_data(content_directory: str, db_directory: str = _db_path, force_reload: bool = False):
    """
    Legacy function for backward compatibility.
    """
    return create_or_load_vectorstore(content_directory, db_directory, force_reload)

# Direct script execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import documents for RAG processing")
    parser.add_argument("--content-dir", required=True, help="Directory containing documents to process")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory to store the vector database")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if cached")
    args = parser.parse_args()
    
    create_or_load_vectorstore(args.content_dir, args.db_dir, args.force)