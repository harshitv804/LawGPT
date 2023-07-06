from langchain import OpenAI
from llama_index import PromptHelper
from llama_index import LLMPredictor, ServiceContext
from llama_index import StorageContext, load_index_from_storage
import gradio as gr
import os
os.environ['OPENAI_API_KEY']

def chat(chat_history, user_input):

    bot_response = query_engine.query(user_input)
    response = ""
    for letter in ''.join(bot_response.response):
        response += letter + ""
        yield chat_history + [(user_input, response)]


context_window = 4096
num_outputs = 500
chunk_overlap_ratio = 0.2
chunk_size_limit = 1000

prompt_helper = PromptHelper(
    context_window, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit)

llm_predictor = LLMPredictor(llm=OpenAI(
    temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper)

storage_context = StorageContext.from_defaults(persist_dir='storage')
index = load_index_from_storage(
    storage_context, service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context)

with gr.Blocks() as demo:
    gr.Image('lawgptlogo.png')
    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        input_query = gr.TextArea(label='Input').style(show_copy_button=True)

    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            clear_input_btn = gr.Button("Clear Input")
        with gr.Column():
            clear_chat_btn = gr.Button("Clear Chat")

    submit_btn.click(chat, [chatbot, input_query], chatbot)
    submit_btn.click(lambda x: gr.update(value=""), None, input_query, queue=False)
    clear_input_btn.click(lambda: None, None, input_query, queue=False)
    clear_chat_btn.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch()
