
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler


import streamlit as st
import os

# ---------------- Streamlit Setup ---------------- #
st.set_page_config(page_title="LawGPT")
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00")

st.markdown("""
<style>
div.stButton > button:first-child { background-color: #ffd0d0; }
div.stButton > button:active { background-color: #ff6262; }
div[data-testid="stStatusWidget"] div button { display: none; }
.reportview-container { margin-top: -2em; }
#MainMenu {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
#stDecoration {display:none;}
button[title="View fullscreen"]{ visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ---------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        return_messages=True
    )

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# ---------------- Embeddings & FAISS ---------------- #
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

db = FAISS.load_local(
    "ipc_vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# ---------------- Prompt ---------------- #
prompt_template = """<s>[INST]
You are a legal assistant specializing in Indian Penal Code (IPC).

Rules:
- Use ONLY the provided context
- Cite IPC sections and case law when available
- Do NOT invent judgments or citations

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "chat_history"]
)

# ---------------- Streaming Callback ---------------- #
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# ---------------- LLM ---------------- #
TOGETHER_AI_API = os.getenv("TOGETHER_AI")
if not TOGETHER_AI_API:
    st.error("TOGETHER_AI environment variable not set")
    st.stop()

llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.3,
    max_tokens=1024,
    streaming=True,
    together_api_key=TOGETHER_AI_API
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# ---------------- Render Chat History ---------------- #
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat Input ---------------- #
user_input = st.chat_input("Ask about the Indian Penal Code...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_container = st.empty()
        stream_handler = StreamHandler(response_container)
        llm.callbacks = [stream_handler]

        # Invoke the chain
        result = qa.invoke({"question": user_input})

        # ---------------- Citations ---------------- #
        ipc_citations = set()
        case_law_citations = set()

        for doc in result["source_documents"]:
            meta = doc.metadata

            # IPC sections
            if meta.get("section"):
                ipc_citations.add(f"• **{meta.get('section')}** – {meta.get('title','')}")

            # Case law
            if meta.get("type") == "case_law":
                case_law_citations.add(
                    f"• **{meta.get('case_name')}**, "
                    f"{meta.get('citation')} "
                    f"({meta.get('court')}, {meta.get('year')})"
                )

        # Combine streaming text with citations
        final_answer = stream_handler.text

        if ipc_citations:
            final_answer += "\n\n📘 **IPC References:**\n" + "\n".join(sorted(ipc_citations))

        if case_law_citations:
            final_answer += "\n\n⚖️ **Case-Law Citations:**\n" + "\n".join(sorted(case_law_citations))

        response_container.markdown(final_answer)

    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    st.button("Reset All Chat 🗑️", on_click=reset_conversation)
