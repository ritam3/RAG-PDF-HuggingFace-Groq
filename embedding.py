import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_embedding(session, file):
    print(session)
    if "vectors" not in session:
        session.loader=PyPDFLoader(file) ## Data Ingestion step
        session.docs=session.loader.load() ## Document Loading
        session.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        session.final_documents=session.text_splitter.split_documents(session.docs[:50])
        session.vectors=FAISS.from_documents(session.final_documents,embeddings)