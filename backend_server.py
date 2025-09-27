import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse 
from typing import Optional
from pydantic import BaseModel
import uvicorn

from document_processor import process_document
import vector_store
import rag_chain

app = FastAPI(
    title="Multimodal RAG Backend API",
    description="API for processing documents, managing a vector store, and answering questions.",
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Receive uploaded file, process it, and add to the knowledge base.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if vector_store.document_exists(file.filename):
            return JSONResponse(
                status_code=200,
                content={"status": "exists", "filename": file.filename, "message": "Document already exists in the knowledge base."}
            )
        processed_data = process_document(file_path)
        if not processed_data:
            raise HTTPException(status_code=400, detail="Failed to process the document or document is empty.")
        vector_store.add_to_knowledge_base(processed_data)
        return JSONResponse(
            status_code=200,
            content={"status": "success", "filename": file.filename, "message": "Document processed and added successfully."}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_rag(
    question: str = Form(...),
):
    """
    Receive a question and return answer and sources via the RAG chain.
    """
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        response = rag_chain.rag_chain_with_source_retrieval(question)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during RAG chain execution: {str(e)}")

@app.get("/indexed_files")
async def get_indexed_files():
    """
    Get the list of all indexed files in the knowledge base.
    """
    try:
        files = vector_store.get_indexed_files()
        return JSONResponse(status_code=200, content={"files": files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file list: {str(e)}")

@app.post("/clear_all")
async def clear_knowledge_base_state():
    """
    Clears the in-memory state of the knowledge base. The client is responsible for guiding the user to restart the server and manually delete files for a full reset.
    """
    try:
        vector_store.clear_knowledge_base()
        return {"status": "success", "message": "In-memory state cleared. Awaiting server restart for full reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during state reset: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)