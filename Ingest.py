from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embedings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,"revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})

# Creates vector embeddings and saves it in the FAISS DB
faiss_db = FAISS.from_documents(texts, embedings)

# Saves and export the vector embeddings databse
faiss_db.save_local("ipc_vector_db")
