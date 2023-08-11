from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
import gradio as gr 

def chat(chat_history, user_input):

    bot_response = qa_chain({"query": user_input})
    bot_response = bot_response['result']
    response = ""
    for letter in ''.join(bot_response):
        response += letter + ""
        yield chat_history + [(user_input, response)]

checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype = torch.float32)

embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

db = Chroma(persist_directory="ipc_vector_data", embedding_function=embeddings)

pipe = pipeline(
    'text2text-generation',
    model = base_model,
    tokenizer = tokenizer,
    max_length = 512,
    do_sample = True,
    temperature = 0.3,
    top_p= 0.95
)
local_llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":2}),
        return_source_documents=True,
        )

with gr.Blocks() as gradioUI:
    
    gr.Image('lawgptlogo.png')
    
    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        input_query = gr.TextArea(label='Input',show_copy_button=True)

    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            clear_input_btn = gr.Button("Clear Input")
        with gr.Column():
            clear_chat_btn = gr.Button("Clear Chat")

    submit_btn.click(chat, [chatbot, input_query], chatbot)
    submit_btn.click(lambda: gr.update(value=""), None, input_query, queue=False)
    clear_input_btn.click(lambda: None, None, input_query, queue=False)
    clear_chat_btn.click(lambda: None, None, chatbot, queue=False)

gradioUI.queue().launch()
