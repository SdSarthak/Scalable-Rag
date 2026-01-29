from langchain.document_loaders import S3DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_documents():
    bucket = os.getenv("S3_BUCKET_NAME")
    loader = S3DirectoryLoader(bucket, prefix="docs/")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)