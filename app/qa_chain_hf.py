import os
from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain.chains import RetrievalQA

def create_qa_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )
    chat = ChatHuggingFace(llm=llm)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=chat, retriever=retriever)
