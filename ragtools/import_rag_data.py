import os
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

# Import necessary LangChain components
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

def load_pdf_file(file_path: str) -> List[Any]:
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

def check_ollama_model(model_name: str) -> bool:
    """Check if an Ollama model is available"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return model_name in result.stdout
    except Exception as e:
        logger.error(f"Error checking Ollama models: {e}")
        return False

def list_available_ollama_models():
    """List all available Ollama models"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        logger.info(f"Available Ollama models:\n{result.stdout}")
        return result.stdout.strip().split("\n")
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []

def init_ollama_embeddings(model: str = "granite-embedding:278m", progress_callback=None) -> Optional[Any]:
    """Initialize embeddings with robust error handling"""
    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain not available. Please install required packages.")
        return None
    
    # List of currently available embedding models (as of May 2025)
    current_embedding_models = ["granite-embedding:278m", "nomic-embed-text", "mxbai-embed-large", "llama3"]
    
    # List available models for troubleshooting
    #list_available_ollama_models()
    
    # Check if model exists in Ollama
    if not check_ollama_model(model):
        logger.warning(f"Model {model} not found in Ollama. Trying to pull it...")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            
            # Try fallback models
            for fallback in current_embedding_models:
                if model != fallback:
                    logger.info(f"Trying fallback model: {fallback}")
                    try:
                        subprocess.run(["ollama", "pull", fallback], check=True)
                        if check_ollama_model(fallback):
                            model = fallback
                            logger.info(f"Using fallback model: {model}")
                            break
                    except Exception:
                        continue
            else:
                logger.error("All fallback models failed")
                return None
    
    # Initialize embeddings
    try:
        logger.info(f"Initializing embedding model: {model}")
        embeddings = OllamaEmbeddings(model=model)
        
        # Test the embeddings
        test_text = "This is a test document."
        embedding = embeddings.embed_query(test_text)
        
        if embedding and len(embedding) > 0:
            logger.info(f"Successfully initialized embeddings with dimension {len(embedding)}")
            return embeddings
        else:
            logger.error("Embedding model returned empty vector")
            return None
            
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
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

    print(f"Creating or loading vectorstore: {db_directory}")

    try:
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
            
        # Initialize embeddings model
        # default from Claude Sonnet 3.7 was nomic-embed-text
        embed_model = init_ollama_embeddings(model="granite-embedding:278m")
        if not embed_model:
            raise ValueError("Failed to initialize Ollama embeddings. Is Ollama running?")
        
        # Load existing vector store if it exists and we don't need to reload
        if not needs_reload and os.path.exists(db_directory):
            logging.info(f"Loading existing vector store from {db_directory}")
            try:
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
            except Exception as e:
                logging.error(f"Error loading existing vector store: {e}")
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
                            print(f"Loaded pdf file: {file_path}")
                        except Exception as e:
                            logging.error(f"Error processing PDF file {file_path}: {str(e)}")
                    
                    # Add support for other file types as needed
                    elif file_ext == '.txt':
                        try:
                            loader = TextLoader(file_path)
                            docs = loader.load()

                            # Apply metadata to each document
                            # for doc in docs:
                            #     doc.metadata.update(metadata)

                            documents.extend(docs)
                            print(f"Loaded text file: {file_path}")
                        except Exception as e:
                            logging.error(f"Error processing text file {file_path}: {str(e)}")

            print(f"Will create new vector store with {len(documents)} documents")

            if not documents:
                logging.warning(f"No documents found in {content_directory}")
                # Create an empty vector store
                vector_store = Chroma(
                    persist_directory=db_directory,
                    embedding_function=embed_model
                )
                vector_store.persist()
                return vector_store
            else:
                logging.info(f"Processing {len(documents)} documents")
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # Smaller chunks
                    chunk_overlap=50,
                    length_function=len
                )
                
                chunks = text_splitter.split_documents(documents)
                logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
                
                # Create vector store directly
                try:
                    logging.info("Creating vector store with all chunks")
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embed_model,
                        persist_directory=db_directory
                    )
                    
                    # Save the content hash
                    with open(hash_file, "w") as f:
                        f.write(files_hash)
                        
                    logging.info(f"Vector store created with {vector_store._collection.count()} embeddings")
                    return vector_store
                except Exception as e:
                    logging.error(f"Error creating vector store: {e}")
                    raise
            
    except Exception as e:
        logging.error(f"Error in create_or_load_vectorstore: {str(e)}")
        raise

def create_simple_vectorstore(content_directory: str, db_directory: str = _db_path, force_reload: bool = False):
    """
    Create a simple vector store without complex batching or error handling.
    """
    print(f"Create simple vector store with {db_directory}")

    try:
        # Initialize embeddings
        embed_model = init_ollama_embeddings(model="granite-embedding:278m")
        if not embed_model:
            raise ValueError("Failed to initialize embeddings")
            
        # Load documents
        documents = []
        for root, _, files in os.walk(content_directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    docs = load_pdf_file(file_path)
                    documents.extend(docs)
                    
        if not documents:
            logger.warning(f"No documents found in {content_directory}")
            return None
            
        logger.info(f"Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks
            chunk_overlap=50,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Create vector store with all chunks at once
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embed_model,
            persist_directory=db_directory
        )
        
        logger.info(f"Created vector store with {len(chunks)} chunks")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating simple vector store: {e}")
        return None

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
    Main function for importing RAG data.
    """
    try:
        # Try the original method first
        result = create_or_load_vectorstore(content_directory, db_directory, force_reload)
        if result:
            return result
            
        # If that fails, fall back to the simplified method
        logger.info("Falling back to simplified vector store creation")
        return create_simple_vectorstore(content_directory, db_directory, force_reload=True)
    
    except Exception as e:
        logger.error(f"All vector store creation methods failed: {e}")
        return None

# Direct script execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import documents for RAG processing")
    parser.add_argument("--content-dir", required=True, help="Directory containing documents to process")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory to store the vector database")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if cached")
    args = parser.parse_args()
    
    import_rag_data(args.content_dir, args.db_dir, args.force)