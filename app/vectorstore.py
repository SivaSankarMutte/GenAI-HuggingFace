import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

def create_vectorstore(docs):
    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id="sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)
