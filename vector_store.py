from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from document_loader import load_documents

def create_vector_store():
    documents = load_documents()
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store