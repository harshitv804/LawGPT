from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import re

# ---------------- Load PDFs ---------------- #

loader = DirectoryLoader(
    "data",
    glob="./*.pdf",
    loader_cls=PyPDFLoader
)

documents = loader.load()

# ---------------- Attach Metadata ---------------- #

for doc in documents:
    filename = os.path.basename(doc.metadata["source"]).lower()

    # -------- Case Law PDFs --------
    # Example filename:
    # case_virsa_singh_vs_state_air_1958_sc_465.pdf
    if filename.startswith("case_"):
        doc.metadata["type"] = "case_law"

        # Basic filename-based parsing
        clean_name = filename.replace("case_", "").replace(".pdf", "")
        clean_name = clean_name.replace("_", " ").title()

        doc.metadata["case_name"] = clean_name
        doc.metadata["citation"] = "Refer PDF"     # Optional: improve later
        doc.metadata["court"] = "Supreme Court of India"
        doc.metadata["year"] = "Unknown"

    # -------- IPC Section PDFs --------
    # Example filename:
    # 302_murder.pdf
    else:
        # Extract IPC section number from filename
        match = re.match(r"(\d+)_", filename)
        if match:
            section = match.group(1)
        else:
            section = "Unknown"

        title = filename.replace(f"{section}_", "").replace(".pdf", "")
        title = title.replace("_", " ").title()

        doc.metadata["type"] = "ipc_section"
        doc.metadata["section"] = f"IPC {section}"
        doc.metadata["title"] = title

# ---------------- Split Documents ---------------- #

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200
)

texts = []

for doc in documents:
    chunks = text_splitter.split_documents([doc])
    for chunk in chunks:
        # Preserve metadata on each chunk
        chunk.metadata.update(doc.metadata)
        texts.append(chunk)

# ---------------- Embeddings ---------------- #

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={
        "trust_remote_code": True,
        "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"
    }
)

# ---------------- Build FAISS DB ---------------- #

faiss_db = FAISS.from_documents(texts, embeddings)

faiss_db.save_local("ipc_vector_db")

print("✅ FAISS vector database built with IPC + Case-Law metadata")
