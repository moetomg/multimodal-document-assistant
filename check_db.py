from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
STORE_PATH = "storage"
DB_PATH = os.path.join(STORE_PATH, "chroma_db")

def check_vector_store():
    print(f"--- Checking ChromaDB at path: {DB_PATH} ---")
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database path does not exist: '{DB_PATH}'")
        print("Please run vector_store.py to create the database.")
        return
    try:
        # Use the same embedding model as your RAG chain
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vector_store = Chroma(
            collection_name="multimodal_rag_persistent",
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        total_docs = vector_store._collection.count()
        print(f"Total documents in the database: {total_docs}")
        if total_docs == 0:
            print("Database is EMPTY. Please run vector_store.py to add documents.")
        else:
            print("Database contains data. Running a test search...")
            test_query = "What is the document about?"
            try:
                results = vector_store.similarity_search(test_query, k=2)
                if results:
                    print("\nTest search SUCCESSFUL. Found relevant documents:")
                    for doc in results:
                        print(f"  - Page {doc.metadata.get('page')}, Source: {doc.metadata.get('source')}")
                else:
                    print("\nTest search found no documents. Possible embedding model mismatch.")
            except Exception as e:
                print(f"\nError during test search: {e}")
    except Exception as e:
        print(f"\nError connecting to the database: {e}")

if __name__ == "__main__":
    check_vector_store()