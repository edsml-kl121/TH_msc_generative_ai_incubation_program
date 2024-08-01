import logging
import os
import streamlit as st
from pymilvus import Collection
from function import (generate_prompt_en, connect_watsonx_embedding, display_streaming_response,
                      connect_watsonx_llm, connect_to_milvus, find_answer, 
                      display_response, display_hits_dataframe)

#---------- settings ----------- #

model_id_llm = 'mistralai/mixtral-8x7b-instruct-v01'
model_id_emb = "" # Example "ibm/slate-125m-english-rtrvr"
vector_index_id = "" # Example "d56b0957-942c-4059-baac-436d15a3b288"
collection_name = "" # Choose the collection name generated from ingestion.py, Example jEEODDYY1aSc

#===========================================================================================
# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🫐",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")

connect_to_milvus()
client, model_emb = connect_watsonx_embedding(model_id_emb)
collection = Collection(collection_name)
collection.load()
model_llm = connect_watsonx_llm(model_id_llm)

if user_question := st.text_input(
"Ask a question about your uploaded Document:"
): 
    print('processing...')
    hits = find_answer(user_question, collection, model_emb)
    prompt = generate_prompt_en(user_question, hits[0][:10])
    print('☕️',prompt)

    st.markdown(f"### Model response to collection {collection_name}")
    display_streaming_response(model_llm, prompt)
    # display_response(model_llm, prompt, collection_name)
    display_hits_dataframe(hits)

