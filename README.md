<h1 align="center">LawGPT</h1>
<h3 align="center">A Generative AI Attorney Chatbot</h1>
<h3 align="center">Know Your Rights! Better Citizen, Better Nation!</h1>

<div align="center">
  <br>
  <video src="https://github.com/harshitv804/LawGPT/assets/100853494/b67d4576-70b1-4b3d-ba73-f855c8b3723b" width="400" />
  <br>
</div>
<br>

# About The Project
LawGPT is a generative AI attorney chatbot that is trained using Indian Penal Code data. This project was developed using LangChain and LaMini Flan-T5 LLM. Ask any questions to the attorney and it will give you the right justice as per the IPC. Are you a noob in knowing your rights? then this is for you!
<br>

# Getting Started

1. Install necessary packages:

   `pip install -r requirements.txt`
2. Create and store vector embeddings:
   
   ```py
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
   ```
3. Run the `LawGPT.py` file.

# Usage

# Contact

# Acknowledgments
