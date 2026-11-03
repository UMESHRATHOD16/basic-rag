
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load environment variables
load_dotenv()

# PDF path
pdf_path = Path(__file__).parent / "data" / "sample.pdf"

# Load PDF
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(docs)

# Generate embeddings locally using HuggingFace
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Qdrant vector store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="sample_collection"
)

print("Vector store created and documents embedded successfully!")
