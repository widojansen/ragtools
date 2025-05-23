# Vector Database Structure in Your RAG Application

Your RAG application uses ChromaDB as the vector database, which organizes data in a specific structure. Here's how the data is structured in your vector database:

## ChromaDB Structure Overview

1. **Collection**: The database is organized into a single collection named "document_chunks"

2. **Each Entry in the Collection Contains**:
   - **ID**: A unique identifier for each chunk (e.g., "doc_123")
   - **Text**: The actual content of the document chunk
   - **Embeddings**: Vector representations of the text chunks (created by your embedding model)
   - **Metadata**: Additional information about each chunk

3. **Physical Storage**:
   - The database is stored in the `chroma_db` directory in your project
   - ChromaDB uses a combination of SQLite and parquet files to store the data

## Data Elements in Detail

### 1. Document Chunks
The text from your PDF documents is split into smaller chunks (approximately 1000 tokens each with 200 tokens overlap, based on your settings). Each chunk becomes a separate entry in the database.

### 2. Embeddings
Each chunk is transformed into a high-dimensional vector (embedding) using the HuggingFace model "BAAI/bge-small-en-v1.5". These embeddings are numerical representations that capture the semantic meaning of the text.

### 3. Metadata
For each chunk, metadata is stored that includes:
- **source**: The path to the original document
- **filename**: Name of the original file
- **file_type**: File extension (e.g., ".pdf")
- **created_at**: Timestamp of when the document was processed
- For PDF files, additional metadata like page numbers may be included

## Database Organization

When you examine the `chroma_db` directory, you'll find:
- SQLite database files storing relationships and metadata
- Parquet files containing the actual embeddings (which are large numerical arrays)
- A `content_hash.txt` file that stores a hash of the processed documents to avoid reprocessing

## How Retrieval Works

When a query is made to the database:
1. The query is converted to an embedding using the same model
2. ChromaDB performs a similarity search to find the most relevant chunks
3. The most similar chunks (based on vector distance) are returned
4. These chunks provide the context for generating an answer

## Example Database Query

For example, if someone asks "What are the regulations for dental implants?":
1. The question is converted to an embedding vector
2. The database finds chunks with similar vectors (semantically similar content)
3. These chunks provide the context about dental implant regulations
4. The LLM uses this context to generate an appropriate answer

This vector-based approach is what makes the RAG system powerful - it can find relevant information based on meaning rather than just keyword matching.


## Me
Is it with this approach possible to remove or update a specific document, e.g. `Regeling mondzorg - NRREG-2507`?

## AI Assistant
Yes, it is possible to remove or update a specific document like "Regeling mondzorg - NRREG-2507" from your vector database, but your current codebase doesn't have explicit functionality for this. Let me explain how you could implement this:
## Current Approach and Limitations
Currently, your system:
1. Processes all PDF files in the `knowledge` directory at once
2. Creates a single hash for all documents
3. Rebuilds the entire database when any document changes

This "all-or-nothing" approach doesn't allow for selective document updates or removals.
## Implementing Document-Level Management
Here's how you could implement selective document management:
### 1. Track Document-Specific Information
Create a metadata file (e.g., `document_index.json`) to track individual documents:
```
{
  "documents": {
    "Regeling mondzorg - NRREG-2507.pdf": {
      "hash": "a1b2c3d4e5f6...",
      "chunk_ids": ["doc_123", "doc_124", "..."],
      "last_updated": "2025-05-23T12:28:01",
      "embedding_model": "BAAI/bge-small-en-v1.5"
    },
    "Other document.pdf": {
      "hash": "f6e5d4c3b2a1...",
      "chunk_ids": ["doc_789", "doc_790", "..."],
      "last_updated": "2025-05-22T10:15:30",
      "embedding_model": "BAAI/bge-small-en-v1.5"
    }
  },
  "database_info": {
    "total_chunks": 1110,
    "last_updated": "2025-05-23T12:28:01"
  }
}
```

### 2. Add Document Removal Function
Add a function to that can remove a specific document: `import_rag_data.py`
```
def remove_document(document_name, db_directory=_db_path):
    """
    Remove a specific document from the vector store.

    Args:
        document_name: The filename of the document to remove
        db_directory: Directory of the vector database

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the document index
        index_path = os.path.join(db_directory, "document_index.json")
        if not os.path.exists(index_path):
            logger.error("Document index not found")
            return False

        with open(index_path, 'r') as f:
            document_index = json.load(f)

        # Check if document exists in the index
        if document_name not in document_index['documents']:
            logger.error(f"Document {document_name} not found in index")
            return False

        # Get chunk IDs for this document
        chunk_ids = document_index['documents'][document_name]['chunk_ids']

        # Initialize embeddings and connect to the DB
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Connect to the existing database
        vectorstore = Chroma(
            persist_directory=db_directory,
            embedding_function=embeddings,
            collection_name="document_chunks"
        )

        # Remove the chunks
        vectorstore._collection.delete(ids=chunk_ids)

        # Update the document index
        del document_index['documents'][document_name]
        document_index['database_info']['total_chunks'] -= len(chunk_ids)
        document_index['database_info']['last_updated'] = datetime.now().isoformat()

        # Save the updated index
        with open(index_path, 'w') as f:
            json.dump(document_index, f, indent=2)

        # Persist changes
        vectorstore.persist()

        logger.info(f"Successfully removed document: {document_name}")
        return True

    except Exception as e:
        logger.error(f"Error removing document {document_name}: {str(e)}")
        return False
```

### 3. Add Document Update Function
Similarly, add a function to update a specific document:

```
def update_document(file_path, db_directory=_db_path):
    """
    Update a specific document in the vector store.

    Args:
        file_path: Path to the updated document
        db_directory: Directory of the vector database

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        document_name = os.path.basename(file_path)

        # First remove the existing document
        removed = remove_document(document_name, db_directory)
        if not removed:
            # If document doesn't exist, we'll just add it as new
            logger.warning(f"Document {document_name} not found for update, will be added as new")

        # Now process and add the updated document
        # [Similar to your existing document processing code]
        # ...

        return True

    except Exception as e:
        logger.error(f"Error updating document {file_path}: {str(e)}")
        return False
```

### 4. Add UI Controls
Add a UI component in to allow document management: `pages/import_data.py`


```
# Document management section
st.subheader("Manage Existing Documents")

# Get list of documents in the knowledge directory
knowledge_dir = st.session_state.knowledge_dir
existing_documents = []
if os.path.exists(knowledge_dir):
    existing_documents = [f for f in os.listdir(knowledge_dir) if f.lower().endswith(('.pdf', '.txt'))]

if existing_documents:
    # Display existing documents with management options
    st.write(f"Found {len(existing_documents)} documents in knowledge base:")

    for doc in existing_documents:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.text(doc)
        with col2:
            if st.button("Update", key=f"update_{doc}"):
                # Process update logic
                st.info(f"Updating {doc}... (not implemented)")
        with col3:
            if st.button("Remove", key=f"remove_{doc}"):
                # Process removal logic
                st.info(f"Removing {doc}... (not implemented)")
else:
    st.info("No documents found in knowledge directory.")
```

## Alternative: Modify Your Hash Approach
Instead of implementing document-level tracking, you could also modify your current approach to:
1. Calculate individual hashes for each document
2. Compare individual hashes to detect changes
3. Only reprocess documents that have changed

This would be more efficient than rebuilding the entire database when a single document changes, but wouldn't allow for selective removal without additional tracking.


## Implementation Recommendation
I recommend implementing document-level tracking as described above, which would give you the flexibility to:
- Remove specific documents
- Update specific documents
- Track which chunks belong to which documents
- Manage your knowledge base more efficiently

This would involve adding the document index tracking to your existing codebase and modifying the import and processing functions to handle document-level operations.
