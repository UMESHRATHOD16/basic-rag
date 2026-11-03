from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

pdf_path = Path(__file__).parent / "data" / "sample.pdf"

# PDF loader.
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

#split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(docs)

#Embedding the chunks using OpenAI embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# create a vector store using Qdrant

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="sample_collection"
)

print("Vector store created and documents embedded successfully.")