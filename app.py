from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_fireworks import Fireworks
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time
st.set_page_config(page_title="LawGPT")
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("https://github.com/harshitv804/LawGPT/assets/100853494/ecff5d3c-f105-4ba2-a93a-500282f0bf00")

st.markdown(
    """
    <style>
div.stButton > button:first-child {
    background-color: #ffd0d0;
}
div.stButton > button:active {
    background-color: #ff6262;
}
   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True,"revision":"289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

prompt_template = """Follow these instructions clearly. This is a chat template and you are a Indian lawyer chat bot who provides correct information on Indian Penal Code. Give only the required answer to the question do not give unwanted information. Do not repeat anything, Generate only the answer, don't ask any questions. You are given the following pieces of information to answer the user's question correctly. You will be given context, chat history and the question. Choose only the required context based on the user's question. If the question is not related to the chat history, then don't use the history. Use chat history when required for similar related questions. While searching for the relevant information always give priority to the context given. Always take the context related only to the question. Use your own knowledge base and answer the question when the context is not related to the user's question. Utilize the provided knowledge base and search for relevant information from the context. Follow the user's question and the format closely. The answer should be short and concise. If you don't know the answer, just say that you don't know, don't try to make up your own questions and answers. Add bullet points and bold text using markdown in the required area if needed, to make it more pleasing to eyes. Do not repeat anything from the prompt template.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

llm = Fireworks(
    model="accounts/fireworks/models/mistral-7b-instruct-4k",
    base_url="https://api.fireworks.ai/inference/v1/chat/completions",
    max_tokens=1024,
    temperature=0.7,
    top_p=1.0,
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})
