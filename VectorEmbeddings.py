   from langchain.document_loaders import PyPDFLoader, DirectoryLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   from langchain.embeddings import SentenceTransformerEmbeddings
   from langchain.vectorstores import Chroma
   
   loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
   documents = loader.load()
   
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
   texts = text_splitter.split_documents(documents)

   embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
   persist_directory = "ipc_vector_data"
   db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
