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

def init_ollama_embeddings(model: str = "all-minilm-l6-v2", progress_callback=None) -> Optional[Any]:
    if not check_ollama_model(model):
        logger.warning(f"Model {model} not found in Ollama. Trying to pull it...")
        try:
            subprocess.run(["ollama", "pull", model], check=True)
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return None
    
    # Rest of your initialization code...