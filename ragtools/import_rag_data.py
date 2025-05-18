import os
import time
import shutil
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import Chroma with direct client access
import chromadb
from langchain_chroma import Chroma

# Directory containing your content files
content_dir = "./knowledge/"
documents = []

print("Setting up RAG system with Granite embeddings and Qwen3...")


# Function to extract and create metadata for a file
def create_file_metadata(file_path, filename):
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


# Process all files in the directory
for filename in os.listdir(content_dir):
    file_path = os.path.join(content_dir, filename)

    # Skip directories
    if os.path.isdir(file_path):
        continue

    # Create metadata for this file
    metadata = create_file_metadata(file_path, filename)

    # Process based on file type
    if filename.endswith(".txt"):
        # Load text files
        try:
            loader = TextLoader(file_path)
            docs = loader.load()

            # Apply metadata to each document
            for doc in docs:
                doc.metadata.update(metadata)

            documents.extend(docs)
            print(f"Loaded text file: {filename}")
        except Exception as e:
            print(f"Error processing text file {filename}: {e}")

    elif filename.endswith(".pdf"):
        # Load PDF files
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Apply metadata to each document (with page numbers)
            for doc in docs:
                doc.metadata.update(metadata)

            documents.extend(docs)
            print(f"Loaded PDF file: {filename} ({len(docs)} pages)")
        except Exception as e:
            print(f"Error processing PDF {filename}: {e}")

print(f"Loaded {len(documents)} documents with metadata")

# Split text into chunks while preserving metadata
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks with preserved metadata")


# Initialize Granite embeddings with robust error handling
def init_granite_embeddings():
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            print(f"Initializing Granite embeddings (attempt {attempt + 1}/{max_retries})...")
            embeddings = OllamaEmbeddings(model="granite-embedding:278m")

            # Test the embeddings with a sample text
            test_text = "This is a test."
            test_embedding = embeddings.embed_query(test_text)
            if test_embedding and len(test_embedding) > 0:
                print(f"Successfully connected to Ollama with Granite embeddings")
                return embeddings
        except Exception as e:
            print(f"Error initializing Granite embeddings (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    # If we get here, all attempts failed
    print("Unable to initialize Granite embeddings after multiple attempts.")
    print("Checking if Ollama is running and if the model is downloaded:")

    # Try to run ollama list to check available models
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print("\nAvailable Ollama models:")
        print(result.stdout)

        if "granite-embedding:278m" not in result.stdout:
            print("\nGranite embedding model not found. Try downloading it with:")
            print("ollama pull granite-embedding:278m")
    except Exception as e:
        print(f"Error checking Ollama status: {e}")

    raise ValueError(
        "Failed to initialize Granite embeddings. Please ensure Ollama is running and the model is downloaded.")


# Get embeddings
try:
    embeddings = init_granite_embeddings()
except Exception as e:
    print(f"Fatal error: {e}")
    exit(1)


# Create vector store with direct ChromaDB interaction
def create_vector_store_direct(chunks, embeddings, db_path):
    # If the directory exists, remove it to start fresh
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        shutil.rmtree(db_path)
        time.sleep(1)  # Give the OS time to complete the deletion

    # Create a new ChromaDB client
    client = chromadb.PersistentClient(path=db_path)

    # Create a collection
    collection = client.create_collection("document_chunks")

    # Process in small batches
    batch_size = 5

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1} ({len(batch)} chunks)")

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

            # Generate embeddings manually
            print(f"  Generating embeddings for batch...")
            try:
                embs = embeddings.embed_documents(texts)

                # Add to collection
                collection.add(
                    ids=ids,
                    embeddings=embs,
                    documents=texts,
                    metadatas=metadatas
                )
                print(f"  Successfully added batch to ChromaDB")
            except Exception as e:
                print(f"  Error generating embeddings: {e}")
                # Try one at a time
                print(f"  Trying one document at a time...")
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
                        print(f"    Added document {doc_id}")
                    except Exception as e2:
                        print(f"    Failed to add document {j}: {e2}")
                        continue

            # Pause to let Ollama recover
            time.sleep(1.0)

        except Exception as e:
            print(f"Error in batch {i // batch_size + 1}: {e}")
            print("Pausing for 5 seconds to let Ollama recover...")
            time.sleep(5)
            continue

    # Create Langchain wrapper over the ChromaDB collection
    return Chroma(
        client=client,
        collection_name="document_chunks",
        embedding_function=embeddings
    )


# Create vector store
db_path = "./chroma_db"
try:
    vectorstore = create_vector_store_direct(chunks, embeddings, db_path)
    print("Vector store created successfully")
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit(1)

# Create a custom prompt template optimized for Qwen
template = """Answer the question based only on the provided context below. Be concise yet thorough.

Context:
{context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


# Initialize Qwen LLM with retry logic
def init_qwen_llm():
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            print(f"Initializing Qwen3 LLM (attempt {attempt + 1}/{max_retries})...")
            llm = Ollama(
                model="qwen3:8b",
                temperature=0.2,  # Lower temperature for more factual responses
                top_k=10,  # Optimized for knowledge retrieval
                num_ctx=4096  # Larger context window
            )

            # Test with a simple query
            test_response = llm.invoke("Hello")
            if test_response:
                print(f"Successfully connected to Ollama with Qwen3")
                return llm
        except Exception as e:
            print(f"Error initializing Qwen3 LLM (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    # If we get here, all attempts failed
    print("Unable to initialize Qwen3 LLM after multiple attempts.")
    print("Checking if Ollama is running and if the model is downloaded:")

    # Try to run ollama list to check available models
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print("\nAvailable Ollama models:")
        print(result.stdout)

        if "qwen3:8b" not in result.stdout:
            print("\nQwen3 model not found. Try downloading it with:")
            print("ollama pull qwen3:8b")
    except Exception as e:
        print(f"Error checking Ollama status: {e}")

    raise ValueError("Failed to initialize Qwen3 LLM. Please ensure Ollama is running and the model is downloaded.")


# Get LLM
try:
    llm = init_qwen_llm()
except Exception as e:
    print(f"Fatal error: {e}")
    exit(1)

# Create retriever with optimized parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,  # Retrieve 4 chunks for better context
        "include_metadata": True
    }
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("\n=== RAG System Ready ===")
print("Using Granite embeddings (278M) and Qwen3 (8B)")
print("Type 'exit' to quit the application")

# Interactive Q&A loop with metadata display
while True:
    query = input("\nAsk a question about your website content: ")
    if query.lower() == 'exit':
        break

    if not query.strip():
        continue

    try:
        print("\nSearching and generating response...")
        start_time = time.time()
        response = qa_chain({"query": query})
        end_time = time.time()

        print("\nAnswer: ", response["result"])
        print(f"\nResponse generated in {end_time - start_time:.2f} seconds")

        # Print sources with metadata
        print("\nSources:")
        for i, doc in enumerate(response["source_documents"]):
            print(f"\nSource {i + 1}:")
            print(f"  Title: {doc.metadata.get('title', 'Unknown')}")

            # Display PDF-specific metadata if available
            if doc.metadata.get('file_type') == '.pdf':
                print(f"  File Type: PDF")
                print(f"  Page: {doc.metadata.get('page', 'Unknown')}")
            else:
                print(f"  File Type: {doc.metadata.get('file_type', 'Unknown')[1:].upper()}")

            print(f"  Last Modified: {doc.metadata.get('last_modified', 'Unknown')}")
            print(f"  Source File: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Content Sample: {doc.page_content[:150]}...")

    except Exception as e:
        print(f"Error processing query: {e}")
        print("Please try a different question.")
