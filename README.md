# Multimodal Document Assistant

## Features

- **End-to-End System**: Decoupled Python backend and Streamlit frontend.
- **Multimodal Queries**: Ask about text and images (diagrams, charts, figures) in your documents.
- **Retrieval-Augmented Generation (RAG)**: Answers are grounded in your documents to reduce hallucination.
- **Cited Sources**: Every answer includes document and page references.
- **Interactive UI**: Easy document upload, querying, and result visualization.

## Tech Stack

- **Backend**: FastAPI, LangChain, Ollama (Llama 3.1), ChromaDB
- **Frontend**: Streamlit
- **Embeddings**: qwen3-embedding:4b (or compatible)

## Getting Started

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-rag-platform.git
cd multimodal-rag-platform
```

### 2. Set Up the Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Ollama Models
```bash
ollama pull llama3.1:8b
ollama pull qwen3-embedding:4b
```

### 4. Run the Backend
```bash
uvicorn backend_server:app --reload
```

### 5. Run the Frontend
```bash
streamlit run app.py
```

## Usage

1. Upload PDF files in the sidebar to build your knowledge base.
2. Ask questions in the main chat area.
3. Optionally upload an image with your question for multimodal queries.
4. Answers will include citations to the source documents.

## 📝 Future Work
- Implement an offline evaluation pipeline to benchmark different embedding and generation models
- Improved document chunking strategies
- Explore on-device/mobile deployment

---

