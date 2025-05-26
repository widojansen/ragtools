import os
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

import streamlit

# Import necessary LangChain components

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
    Generate a hash of all PDF and TXT files in the directory to detect changes.
    """
    hash_obj = hashlib.md5()
    
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return ""
            
        knowledge_files = sorted([f for f in os.listdir(directory) if (f.lower().endswith('.pdf') or f.lower().endswith('.txt'))])
        for filename in knowledge_files:
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


def load_txt_file(file_path: str) -> List[Any]:
    """Load a single TXT file and add metadata."""
    try:
        logger.info(f"Loading TXT file: {os.path.basename(file_path)}")
        loader = TextLoader(file_path)
        docs = loader.load()

        # Add metadata to each document
        for doc in docs:
            doc.metadata.update(create_file_metadata(file_path))

        logger.info(f"Loaded {len(docs)} pages with metadata")
        return docs
    except Exception as e:
        logger.error(f"Error loading TXT {file_path}: {e}")
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

def try_fallback_models(available_models: list[str], original_model: str) -> Optional[str]:
    """Try to pull and use fallback embedding models when the original model fails."""
    for fallback_model in available_models:
        if original_model != fallback_model:
            logger.info(f"Trying fallback model: {fallback_model}")
            try:
                subprocess.run(["ollama", "pull", fallback_model], check=True)
                if check_ollama_model(fallback_model):
                    logger.info(f"Using fallback model: {fallback_model}")
                    return fallback_model
            except Exception as e:
                logger.debug(f"Failed to pull fallback model {fallback_model}: {e}")
                continue
    
    logger.error("All fallback models failed")
    return None


def init_ollama_embeddings(model: str) -> Optional[Any]:
    # List of currently available embedding models (as of May 2025)
    current_embedding_models = ["granite-embedding:278m", "nomic-embed-text", "mxbai-embed-large", "llama3"]
    
    # Check if the model exists in Ollama
    if not check_ollama_model(model):
        logger.warning(f"Model {model} not found in Ollama. Trying to pull it...")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            
            # Try fallback models
            fallback_result = try_fallback_models(current_embedding_models, model)
            if fallback_result:
                model = fallback_result
            else:
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

def create_or_load_vectorstore(embeddings: str, content_directory: str, db_directory: str, force_reload: bool = False):
    """
    Create or load a vector store with document embeddings from the content directory.
    
    Args:
        embeddings : embeddings LLM
        content_directory: Directory containing the documents to embed
        db_directory: Directory to store the vector database
        force_reload: Force reloading and re-embedding of all documents
        
    Returns:
        A vector store object
    """

    print(f"Embeddings model: {embeddings}")
    print(f"Knowledge directory: {content_directory}")
    print(f"Creating or loading vectorstore: {db_directory}")

    try:
        # Check if the content directory exists
        if not os.path.exists(content_directory):
            raise ValueError(f"Content directory does not exist: {content_directory}")
            
        # Create the DB directory if it doesn't exist
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
        embed_model = init_ollama_embeddings(model=embeddings)
        if not embed_model:
            raise ValueError("Failed to initialize Ollama embeddings. Is Ollama running?")
        
        # Load the existing vector store if it exists and we don't need to reload
        if not needs_reload and os.path.exists(db_directory):
            logging.info(f"Loading existing vector store from {db_directory}")
            try:
                vector_store = Chroma(
                    persist_directory=db_directory,
                    embedding_function=embed_model
                )

                if vector_store._collection.count() <= 0:
                    logging.warning("Vector store exists but is empty. Will rebuild.")
                    needs_reload = True
                # Check if the vector store actually has embeddings
                else:
                    logging.info(f"Successfully loaded vector store with {vector_store._collection.count()} embeddings")
                    return vector_store
            except Exception as e:
                logging.error(f"Error loading existing vector store: {e}")
                needs_reload = True
                
        # Create the new vector store if needed
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
                            docs = load_txt_file(file_path)
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
                
                return vector_store
            else:
                logging.info(f"Processing {len(documents)} pages")
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # Smaller chunks
                    chunk_overlap=50,
                    length_function=len
                )
                
                chunks = text_splitter.split_documents(documents)
                logging.info(f"Split {len(documents)} pages into {len(chunks)} chunks")
                
                # Create the vector store directly
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
        return None

    except Exception as e:
        logging.error(f"Error in create_or_load_vectorstore: {str(e)}")
        raise


def get_retriever(content_directory: str, db_directory: str):
    """
    Get a retriever from an existing or newly created vector store.
    This is the main function to call from external applications.
    """
    vectorstore = create_or_load_vectorstore(streamlit.session_state.embeddings, content_directory, db_directory)
    if vectorstore:
        return vectorstore.as_retriever()
    return None

def import_rag_data(content_directory: str, db_directory: str, force_reload: bool = False):
    """
    Main function for importing RAG data.
    """
    try:
        result = create_or_load_vectorstore(streamlit.session_state.embeddings, content_directory, db_directory, force_reload)
        if result:
            return result
        # Failed without exception...
        logger.info("Failed to create vector store without exception")
        return None
    except Exception as e:
        logger.error(f"All vector store creation methods failed: {e}")
        return None

# Direct script execution
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import documents for RAG processing")
    parser.add_argument("--content-dir", required=True, help="Directory containing documents to process")
    parser.add_argument("--embeddings", required=True, help="Embeddings model")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory to store the vector database")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if cached")
    parser.add_argument("--force-reload", action="store_true", help="Force reprocessing even if cached")
    args = parser.parse_args()
    
    import_rag_data(args.content_dir, args.db_dir, args.force)