import gzip
import json
import operator
import string
import random
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema,DataType
import os
from dotenv import load_dotenv
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models import Embeddings

embedding_dimension = 384 #adjust the value according to your choice of embedding

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

def initiate_username():
    characters = string.ascii_letters + string.digits + '_'
    username = ''.join(random.choice(characters) for _ in range(random.randint(10, 32)))
    print('initiate username....')
    return 'b'+ username

#------create milvus database
def create_milvus_db(collection_name):
    print("Existing collections:", utility.list_collections())
    if collection_name in utility.list_collections():
        utility.drop_collection(collection_name)
        print("Dropped old collection")

    # Ensure collection is dropped
    existing_collections = utility.list_collections()
    print(f"Existing collections after drop operation: {existing_collections}")
    
    # Define the fields for the collection schema
    item_id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=50000)
    embeddings = FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim= embedding_dimension)
    
    # Create the collection schema
    schema = CollectionSchema(
        fields=[item_id, text, embeddings],
        description="Inserted policy from user",
        enable_dynamic_field=True
    )

    # Print schema for debugging
    print("Collection schema:", schema)
    
    # Create the collection
    collection = Collection(name=collection_name, schema=schema, using='default')
    
    # Print collection details for debugging
    print("Collection created with schema:", collection.schema)
    
    return collection

#----------embedding data + store in milvus
def embedding_data_vector(collection_name, chunk, vector):
    collection = create_milvus_db(collection_name)
    collection.insert([chunk,vector])
    collection.create_index(field_name="embeddings",\
                            index_params={"metric_type":"IP","index_type":"IVF_FLAT","params":{"nlist":16384}})
    return collection

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

def hydrade_milvus(collection_name, client, vector_index_id, model_id_emb):
    vector_index_id_name = vector_index_id.replace('-', '_')[10]
    vector_index_details = client.data_assets.get_details(vector_index_id)
    vector_index_properties = vector_index_details["entity"]["vector_index"]
    data = client.data_assets.get_content(vector_index_id)
    content = gzip.decompress(data)
    stringified_vectors = str(content, "utf-8")
    vectors = json.loads(stringified_vectors)
    # vector of 40 chunks, each chunk is dict with {'content, 'embedding', 'version'}
    # store_vector = list(map(lambda d: (d['content'], d['embedding']), vectors))
    chunk = list(map(operator.itemgetter('content'), vectors))
    vector = list(map(operator.itemgetter('embedding'), vectors))
    print('storing vector..')
    print('----------------------')
    if collection_name in utility.list_collections():
        utility.drop_collection(f'{collection_name}')
    collection = embedding_data_vector(collection_name, chunk, vector)
    return collection