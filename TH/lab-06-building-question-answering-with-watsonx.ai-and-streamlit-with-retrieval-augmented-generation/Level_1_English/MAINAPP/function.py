from pymilvus import connections
import streamlit as st
from dotenv import load_dotenv
import os
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models import Model
import pandas as pd

load_dotenv()
api_key = os.getenv("WATSONX_APIKEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)
host = str(os.getenv("MILVUS_HOST", None))
port = os.getenv("MILVUS_PORT", None)
server_pem_path = os.getenv("MILVUS_SERVER_PEM_PATH", None)
server_name = os.getenv("MILVUS_SERVER_NAME", None)
user = os.getenv("MILVUS_USER", None)
password = os.getenv("MILVUS_PASSWORD", None)

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

#--------generate promt reday to prompt in model
def generate_prompt_en(question, context):
    output = f"""[INST] You are a helpful, respectful assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

You will receive HR Policy on user queries HR POLICY DETAILS, and QUESTION from user in the ''' below. Answer the question in English.
'''
HR POLICY DETAILS:
{context}

QUESTION: {question}
'''
Answer the QUESTION use details about HR Policy from HR POLICY DETAILS, explain your reasonings if the question is not related to REFERENCE please Answer
“I don’t know the answer, it is not part of the provided HR Policy”.

QUESTION: {question} [/INST]
ANSWER: """
    return output  


#---------- connection ----------#
@st.cache_resource
def connect_watsonx_embedding(model_id_emb):
    emb = Embeddings(
        model_id=model_id_emb,
        credentials=creds,
        project_id=project_id,
        params={
            "truncate_input_tokens": 512
        }
    )
    wml_credentials = creds
    client = APIClient(credentials=wml_credentials, project_id=project_id)
    return client, emb

@st.cache_resource
def connect_watsonx_llm(model_id_llm):
    model = Model(
	model_id = model_id_llm,
	params = {
        'decoding_method': "greedy",
        'min_new_tokens': 1,
        'max_new_tokens': 400,
        'temperature': 0.0,
        'repetition_penalty': 1
    },
	credentials=creds,
    project_id=project_id)
    return model

@st.cache_data
def connect_to_milvus():
    print('connecting to milvus...')
    connections.connect(
        "default", 
        host = host, 
        port = port, 
        secure=True, 
        server_pem_path = server_pem_path,
        server_name = server_name,
        user = user,
        password=password)
    print("Milvus connected")


#----------embedding question + search in Milvus vector database
def find_answer(question, collection, watsonx_embedding):
    embedded_vector  = watsonx_embedding.embed_documents([question])    # embedding question
    print('embedding question...')
    # collection.load()           # query data from collection
    hits = collection.search(data=embedded_vector, anns_field="embeddings", param={"metric":"IP","offset":0},
                    output_fields=["text"], limit=15)
    return hits

def display_streaming_response(model_llm, prompt):
    model_response_placeholder = st.empty()
    full_response = ""

    for response in model_llm.generate_text_stream(prompt):
        for result in response:
            full_response += result
        with model_response_placeholder.container():
            st.markdown(full_response)

def display_response(model_llm, prompt, collection_name):
    response = model_llm.generate_text(prompt)
    st.text_area(label=f"Model Response to collection name {collection_name}", value=response, height=300)

def create_hits_dataframe(hits, num_hits=10):
    if len(hits[0]) < 10:
        num_hits = len(hits[0])
    dict_display = {
        f'doc{i}': [hits[0][i].text]
        for i in range(num_hits)
    }
    df = pd.DataFrame(dict_display).T
    df.columns = ['Reference from document']
    return df

def display_hits_dataframe(hits, num_hits=10, width=1000):
    df_dis = create_hits_dataframe(hits, num_hits)
    st.dataframe(df_dis, width=width)

