import logging
import os

#for UI
import streamlit as st
from langchain.callbacks import StdOutCallbackHandler
from PIL import Image

# for Milvus 
from pymilvus import connections, utility, Collection

# for function
from function import (connect_to_milvus, connect_watsonx_llm, initiate_username, read_pdf, generate_prompt_th, create_milvus_db, 
                       split_text_with_overlap, embedding_data, find_answer, display_hits_dataframe)


#---------- settings ----------- #

# model_id_llm='meta-llama/llama-3-8b-instruct'
model_id_llm='meta-llama/llama-3-1-8b-instruct'
model_id_emb="kornwtp/simcse-model-phayathaibert"

# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🍰",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")

connect_to_milvus()
model_llm = connect_watsonx_llm(model_id_llm)

handler = StdOutCallbackHandler()

# Sidebar contents
with st.sidebar:
    st.title("RAG App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
    ''')
    st.text_area('LLM used', f'{model_id_llm}', height=10)
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
    image = Image.open('watsonxai.jpg')
    st.image(image, caption='Powered by watsonx.ai')    

#===========================================================================================

username = initiate_username()
if uploaded_files := st.file_uploader("กรุณาเลีอกไฟล์ PDF", accept_multiple_files=True):
    print(utility.list_collections())
    print('======',username,'======')
    if (username in utility.list_collections()):
        print('----- collection already exist')
        collection = Collection(username)
    else:
        thai_text = read_pdf(uploaded_files)
        chunks = split_text_with_overlap(thai_text, 1000, 300)
        print('----- create new collection')
        collection = create_milvus_db(username)
        collection = embedding_data(chunks, collection)
else:
    utility.drop_collection(f'{username}')
    print('dropped collection')

if uploaded_files :
    print('ready for input...')
    if user_question := st.text_input(
        "ถามคำถามเกี่ยวกับเอกสารของคุณ:"
    ): 
        print('processing...')
        hits = find_answer(user_question, collection)
        prompt = generate_prompt_th(user_question, hits[0][:4])
        response = model_llm.generate_text(prompt)
        st.text_area(label="Model Response", value=response, height=300)
        display_hits_dataframe(hits)
