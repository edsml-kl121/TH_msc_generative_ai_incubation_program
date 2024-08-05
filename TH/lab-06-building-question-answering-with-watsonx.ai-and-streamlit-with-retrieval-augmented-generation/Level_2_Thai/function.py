import pandas as pd
import string
import random
import os

from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema,DataType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections
import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models import Model


# for PDF Download 
import tempfile
from langchain.document_loaders import PyPDFLoader

load_dotenv()
api_key =           os.getenv("WATSONX_APIKEY",         None)
ibm_cloud_url =     os.getenv("IBM_CLOUD_URL",          None)
project_id =        os.getenv("PROJECT_ID",             None)
host =          str(os.getenv("MILVUS_HOST",            None))
port =              os.getenv("MILVUS_PORT",            None)
server_pem_path =   os.getenv("MILVUS_SERVER_PEM_PATH", None)
server_name =       os.getenv("MILVUS_SERVER_NAME",     None)
user =              os.getenv("MILVUS_USER",            None)
password =          os.getenv("MILVUS_PASSWORD",        None)
emb_api_key =       os.getenv("EMB_WATSONX_APIKEY",     None)
emb_ibm_cloud_url = os.getenv("EMB_IBM_CLOUD_URL",      None)
emb_project_id =    os.getenv("EMB_PROJECT_ID",         None)
emb_space_id =      os.getenv("EMB_SPACE_ID",           None)
deployment_id =     os.getenv("EMB_DEPLOYMENT_ID",      None)
if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }
    emb_creds = {
        "url": emb_ibm_cloud_url,
        "apikey": emb_api_key 
    }

    client = APIClient(credentials=emb_creds, space_id=emb_space_id)

#------connection
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

@st.cache_data
def read_pdf(uploaded_files):
    for uploaded_file in uploaded_files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
          temp_file.write(bytes_data)
          filepath = temp_file.name
          with st.spinner('Waiting for the file to upload'):
            loader = PyPDFLoader(filepath)
            data = loader.load()
            docs = format_pdf_reader(data)
            return docs

@st.cache_data
def initiate_username():
    characters = string.ascii_letters + string.digits + '_'
    username = ''.join(random.choice(characters) for _ in range(random.randint(5, 32)))
    print('initiate username....')
    return 'a'+username

#------create milvus database
def create_milvus_db(collection_name):
    item_id    = FieldSchema( name="id",         dtype=DataType.INT64,    is_primary=True, auto_id=True )
    text       = FieldSchema( name="text",       dtype=DataType.VARCHAR,  max_length= 50000             )
    embeddings = FieldSchema( name="embeddings", dtype=DataType.FLOAT_VECTOR,    dim=768                )
    schema     = CollectionSchema( fields=[item_id, text, embeddings], description="Inserted policy from user", enable_dynamic_field=True )
    collection = Collection( name=collection_name, schema=schema, using='default' )
    return collection


#----------split data using Langchain textspliter
def import_text_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        )
    return text_splitter


def split_text_with_overlap(text, chunk_size, overlap_size):
    chunks = []
    start_index = 0

    while start_index < len(text):
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        start_index += (chunk_size - overlap_size)

    return chunks

def embedding_data(chunks, collection):
    payload = {
    'input_data': [
        {
            'values': [
                chunks
            ]
        }
    ]
}   
    reponse = client.deployments.score(deployment_id, payload)
    ch  = [i[0] for i in reponse['predictions'][0]['values']]
    emb = [i[1] for i in reponse['predictions'][0]['values']]
    collection.insert([ch,emb])
    collection.create_index(field_name="embeddings",\
                        index_params={"metric_type":"IP","index_type":"IVF_FLAT","params":{"nlist":16384}})
    return collection

#----------embedding question + search in Milvus vector database
def find_answer(question, collection):
    payload = {
    'input_data': [
        {
            'values': [
                [question]
            ]
        }
    ]
}   
    reponse = client.deployments.score(deployment_id, payload)   # embedding question
    embedded_vector = [reponse['predictions'][0]['values'][0][1]]
    print('embedding question...')
    collection.load()           # query data from collection
    hits = collection.search(data=embedded_vector, anns_field="embeddings", param={"metric":"IP","offset":0},
                    output_fields=["text"], limit=15)
    return hits

#------upload pdf
def format_pdf_reader(raw_data):
    # format content from pdf into text
    pdf_text = ""
    for data in raw_data:
        pdf_text+=data.page_content+"\n"
    return pdf_text

#--------generate promt reday to prompt in model

def generate_prompt_th(question, context):
    output = f"""**`<|begin_of_text|><|start_header_id|>**system<|end_header_id|>`

คุณเป็นผู้ช่วยที่ใจดี โปรดตอบคำถามอย่างใจดีและมีประโยชน์ที่สุดเสมอ พร้อมกับรักษาความปลอดภัย คำตอบของคุณไม่ควรมีเนื้อหาที่เป็นอันตราย ไม่ธรรมดา แบ่งแยกทางเชื้อชาติ ลำเอียงทางเพศ มีพิษ อันตราย หรือผิดกฎหมาย โปรดให้แน่ใจว่าคำตอบของคุณไม่มีอคติทางสังคมและเป็นบวกในธรรมชาติ ถ้าคำถามไม่มีเหตุผล หรือไม่สอดคล้องกับความเป็นจริง โปรดอธิบายเหตุผลแทนที่จะตอบคำถามที่ไม่ถูกต้อง ถ้าคุณไม่ทราบคำตอบของคำถาม โปรดอย่าแชร์ข้อมูลที่ผิด 

คุณจะได้รับนโยบายทรัพยากรบุคคล ที่เป็นแหล่งฃ้อมูลในการคำถามจากผู้ใช้ จงตอบคำถามเป็นภาษาไทย

รายละเอียดนโยบายทรัพยากรบุคคล:
{context}

คำถาม: {question}

ตอบคำถามโดยใช้ลฃ้อมูลจาก "รายละเอียดนโยบายทรัพยากรบุคคล" อธิบายเหตุผลของคุณ
หากคำถามไม่เกี่ยวข้องกับข้อมูลอ้างอิง โปรดตอบว่า “ฉันไม่ทราบคำตอบ, มันไม่ใช่ส่วนหนึ่งของนโยบายทรัพยากรบุคคลที่ได้รับ”
<|eot_id|><|start_header_id|>user<|end_header_id|>
สวัสดี<|eot_id|><|start_header_id|>assistant<|end_header_id|>
สวัสดีครับผมคือผู้ช่วย HR Policy ครับ กรุณาพิมพ์คำถามของคุณข้างล่างได้เลยครับ<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    return output  

def generate_prompt_en(question, context, model_type="llama-2"):
    output = '''<|begin_of_text|><|start_header_id|>**system<|end_header_id|>`
You are a helpful, respectful Thai assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

You will receive HR Policy on user queries HR POLICY DETAILS, and QUESTION from user below. Answer the question in Thai.

HR POLICY DETAILS:
{context}
QUESTION: {question}

Answer the QUESTION use details about HR Policy from HR POLICY DETAILS, explain your reasonings if the question is not related to REFERENCE please Answer
“I don’t know the answer, it is not part of the provided HR Policy”
<|eot_id|><|start_header_id|>user<|end_header_id|>

hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

hello, I'm your HR Policy Assistance, please type your question<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''
    return output   

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