import base64
import io
import os
import uuid
import ollama
from PIL import Image
from typing import List, Dict, Any, Optional
import json
import shutil
import gc

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_processor import process_document

# Path and model configuration
STORE_PATH = "storage"
CHROMA_DB_PATH = os.path.join(STORE_PATH, "chroma_db")
DOCSTORE_PATH = os.path.join(STORE_PATH, "docstore")
EMBEDDING_MODEL = "qwen3-embedding:4b"
LLM_MODEL = "qwen2.5vl:7b"

# Lazily loaded singletons
_embeddings: Optional[OllamaEmbeddings] = None
_vectorstore: Optional[Chroma] = None
_docstore: Optional[LocalFileStore] = None
_retriever: Optional[MultiVectorRetriever] = None
_text_splitter: Optional[RecursiveCharacterTextSplitter] = None

def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"Initializing embedding model: {EMBEDDING_MODEL}")
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings

def get_docstore() -> LocalFileStore:
    global _docstore
    if _docstore is None:
        if not os.path.exists(DOCSTORE_PATH):
            os.makedirs(DOCSTORE_PATH)
        _docstore = LocalFileStore(DOCSTORE_PATH)
    return _docstore

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        print("Loading/Initializing Chroma vector store...")
        _vectorstore = Chroma(
            collection_name="multimodal_rag_persistent",
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DB_PATH
        )
    return _vectorstore

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return _text_splitter

def get_retriever() -> MultiVectorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MultiVectorRetriever(
            vectorstore=get_vectorstore(),
            docstore=get_docstore(),
            id_key="doc_id"
        )
    return _retriever

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_image_summary(image_b64: str) -> str:
    print("Generating summary for an image...")
    try:
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{
                'role': 'user',
                'content': 'Provide a detailed description of this image. If it contains charts, graphs, or tables, extract the key information and data. Describe the main subject and any important context.',
                'images': [image_b64]
            }],
            options={"temperature": 0.0}
        )
        print("Image summary generated successfully.")
        return response['message']['content']
    except Exception as e:
        print(f"ERROR: Failed to generate image summary: {e}")
        return "No summary could be generated for this image."

def generate_formula_summary(image_b64: str) -> str:
    print("Generating summary for a formula image...")
    try:
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{
                'role': 'user',
                'content': "This image contains a mathematical or chemical formula. Transcribe it into a standard textual representation like LaTeX or a simple plain text description. For example, for an image of x squared, return 'x^2'.",
                'images': [image_b64]
            }],
            options={"temperature": 0.0}
        )
        return response['message']['content']
    except Exception as e:
        return f"Error generating formula summary: {e}"

def document_exists(filename: str) -> bool:
    vectorstore = get_vectorstore()
    existing_docs = vectorstore.get(where={"source": filename}, limit=1)
    if existing_docs and existing_docs['ids']:
        print(f"Document '{filename}' already exists in the vector store.")
        return True
    return False

def add_to_knowledge_base(processed_data: List[Dict[str, Any]]):
    """
    Build or rebuild the vector knowledge base from processed document data.
    This function is idempotent: each call clears old data and fills with new data.
    """
    if not processed_data:
        print("No processed data to add. Skipping.")
        return

    source_filename = processed_data[0].get('source', 'unknown_source')
    if document_exists(source_filename):
        print(f"Skipping addition of '{source_filename}' as it's already in the database.")
        return "exists"
    print(f"\n--- Adding content from '{source_filename}' to the knowledge base ---")

    doc_ids = []
    docs_to_vectorize = []
    metadatas = []
    items_to_store_in_docstore = []
    text_splitter = get_text_splitter()
    for item in processed_data:
        unique_id = str(uuid.uuid4())
        chunk_metadata = {"source": source_filename, "page": item.get('page', 1)}
        if item['type'] == 'text':
            sub_chunks = text_splitter.split_text(item['content'])
            for chunk in sub_chunks:
                chunk_id = str(uuid.uuid4())
                chunk_metadata["doc_id"] = chunk_id
                items_to_store_in_docstore.append((chunk_id, {"type": "text", "content": chunk}))
                doc_ids.append(chunk_id)
                docs_to_vectorize.append(chunk)
                metadatas.append(chunk_metadata.copy())
        elif item['type'] == 'image' or item['type'] == 'image_formula':
            image_b64 = image_to_base64(item['content'])
            chunk_metadata["doc_id"] = unique_id
            if item['type'] == 'image':
                summary = generate_image_summary(image_b64)
                items_to_store_in_docstore.append((unique_id, {
                    "type": "image",
                    "content_b64": image_b64,
                    "summary": f"Summary of an image from page {item.get('page', 1)}: {summary}"
                }))
            else:
                summary = generate_formula_summary(image_b64)
                items_to_store_in_docstore.append((unique_id, {
                    "type": "image",
                    "content_b64": image_b64,
                    "summary": f"A formula from page {item['page']} is represented as: {summary}"
                }))
            doc_ids.append(unique_id)
            docs_to_vectorize.append(summary)
            metadatas.append(chunk_metadata)
    docstore = get_docstore()
    if items_to_store_in_docstore:
        encoded_items = [
            (key, json.dumps(value).encode('utf-8'))
            for key, value in items_to_store_in_docstore
        ]
        docstore.mset(encoded_items)
        print(f"Successfully stored {len(encoded_items)} items in the docstore.")
    retriever = get_retriever()
    if docs_to_vectorize:
        print(f"Embedding and adding {len(docs_to_vectorize)} new chunks to the vector store...")
        retriever.vectorstore.add_texts(texts=docs_to_vectorize, ids=doc_ids, metadatas=metadatas)
    print("--- Knowledge base built successfully! ---")

def clear_knowledge_base():
    """
    Completely clears the knowledge base, ensuring ChromaDB connections are closed before deleting files.
    """
    global _vectorstore, _docstore, _retriever
    print("\n--- Clearing the entire knowledge base ---")
    client_to_reset = None
    if _vectorstore is not None:
        try:
            client_to_reset = _vectorstore._client
        except AttributeError:
            pass
    print("Releasing Python object references...")
    _vectorstore = None
    _docstore = None
    _retriever = None
    gc.collect()
    if client_to_reset:
        print("Resetting ChromaDB client to close connections...")
        try:
            client_to_reset.reset()
        except Exception as e:
            print(f"Warning: Could not reset ChromaDB client. Deletion might fail. Error: {e}")
    print("Deleting storage directories...")
    if os.path.exists(CHROMA_DB_PATH):
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print(f"Successfully deleted: {CHROMA_DB_PATH}")
        except Exception as e:
            print(f"ERROR deleting ChromaDB directory: {e}")
    if os.path.exists(DOCSTORE_PATH):
        try:
            shutil.rmtree(DOCSTORE_PATH)
            print(f"Successfully deleted: {DOCSTORE_PATH}")
        except Exception as e:
            print(f"ERROR deleting Docstore directory: {e}")
    print("--- Knowledge base cleared successfully! ---")

def get_indexed_files() -> List[str]:
    """
    Returns a sorted list of unique source filenames from the vector store.
    """
    try:
        all_entries = get_vectorstore().get(include=["metadatas"])
        if not all_entries or 'metadatas' not in all_entries:
            return []
        sources = {meta['source'] for meta in all_entries['metadatas'] if meta and 'source' in meta}
        return sorted(list(sources))
    except Exception as e:
        print(f"Could not retrieve indexed files. Maybe the DB is empty? Error: {e}")
        return []

if __name__ == '__main__':
    print("--- RUNNING FINAL TEST SCRIPT ---")
    print("\n[PHASE 0] Forcing a clean slate...")
    clear_knowledge_base()
    test_folder = "files"
    test_file_name = "literature1.pdf"
    test_file_path = os.path.join(test_folder, test_file_name)
    if not os.path.exists(test_file_path):
        print(f"FATAL: Test file not found at '{test_file_path}'. Halting.")
    else:
        print(f"\n[PHASE 1] Processing document: {test_file_name}")
        processed_data = process_document(test_file_path)
        if not processed_data:
            print("FATAL: Document processing returned no data. Halting.")
        else:
            print("\n[PHASE 2] Adding processed data to the knowledge base...")
            add_to_knowledge_base(processed_data)
    print("\n--- Final test script finished. ---")
    print("--- Now, run check_db.py to verify. ---")