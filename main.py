from fastapi import FastAPI
from app.routes.query import router as query_router

app = FastAPI(
    title="Local RAG API",
    description="RAG application using LangChain, HuggingFace, Qdrant and Ollama",
    version="1.0.0"
)

# Register query routes
app.include_router(query_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to Local RAG API!",
        "status": "running"
    }
