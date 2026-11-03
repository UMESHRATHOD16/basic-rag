from fastapi import APIRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama

router = APIRouter(
    prefix="/query",
    tags=["query"],
)

# Same embedding model used during indexing
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing Qdrant collection
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333"
)

# Local LLM using Ollama
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


@router.post("/")
def query(question: str):

    # 1. Retrieve relevant chunks
    search_results = vector_db.similarity_search(
        query=question,
        k=3
    )

    # 2. Prepare context
    context = "\n\n".join(
        [
            f"[Page {result.metadata.get('page', 'unknown')}] "
            f"{result.page_content}"
            for result in search_results
        ]
    )

    # 3. Create prompt
    SYSTEM_PROMPT = f"""
    You are a helpful assistant that answers questions
    based on the provided context.

    If the context does not contain the answer,
    respond with "I don't know."

    Include page numbers wherever applicable.

    Context:
    {context}
    """

    # 4. Generate answer using Ollama
    response = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", question)
    ])

    # 5. Return response
    return {
        "question": question,
        "answer": response.content
    }
