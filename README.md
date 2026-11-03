# Local RAG Application 🤖

A locally running **Retrieval-Augmented Generation (RAG)** application built using FastAPI, LangChain, HuggingFace, Qdrant, and Ollama.

The application allows users to ask questions about PDF documents by retrieving relevant information from a vector database and generating context-aware answers using a locally hosted LLM.

## 🚀 Tech Stack

- **Python** – Core programming language
- **FastAPI** – Backend API
- **LangChain** – RAG pipeline orchestration
- **HuggingFace** – Text embedding generation (`all-MiniLM-L6-v2`)
- **Qdrant** – Vector database for storing and retrieving embeddings
- **Ollama** – Running Llama 3.2 (3B) locally
- **PyPDF** – PDF document loading

## ⚙️ How It Works

1. **Document Loading:** Extracts text from PDF documents.
2. **Text Splitting:** Divides documents into smaller chunks with overlapping content.
3. **Embedding Generation:** Converts text chunks into vector embeddings using HuggingFace.
4. **Vector Storage:** Stores embeddings in Qdrant.
5. **Similarity Search:** Retrieves the most relevant chunks based on the user's question.
6. **Answer Generation:** Passes the retrieved context to Llama 3.2 through Ollama to generate an answer.

## 📁 Project Structure

```text
rag/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── index.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── query.py
│   └── data/
│       └── sample.pdf
├── requirements.txt
└── README.md
```

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rag
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Qdrant

Make sure Docker is installed and running.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Install and Run Ollama

Download Ollama from [ollama.com](https://ollama.com/).

Pull the model:

```bash
ollama pull llama3.2:3b
```

Run it:

```bash
ollama run llama3.2:3b
```

### 6. Index Your PDF

Place your PDF inside `app/data/` and run:

```bash
python -m app.index
```

### 7. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## 📡 API Endpoint

**POST** `/query/`

Pass a question to retrieve relevant information from the indexed PDF and generate an answer.

Example:

```text
/query/?question=What is network communication?
```

Response:

```json
{
  "question": "What is network communication?",
  "answer": "..."
}
```

## 🌱 Future Improvements

- Support multiple PDF uploads.
- Add conversation history.
- Improve retrieval accuracy using reranking.
- Build a frontend chat interface.
- Add source citations and page references.

## 💡 Key Learning

This project explores how embeddings, vector databases, semantic search, and Large Language Models work together to build a local question-answering system without relying on paid LLM APIs.
