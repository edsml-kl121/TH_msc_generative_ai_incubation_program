from function import initiate_username, connect_watsonx_embedding, hydrade_milvus, connect_to_milvus

#---------- settings ----------- #

model_id_emb = "" # Example "ibm/slate-125m-english-rtrvr"
vector_index_id = "" # Example "d56b0957-942c-4059-baac-436d15a3b288"
#===========================================================================================

collection_name = initiate_username()
client, model_emb = connect_watsonx_embedding(model_id_emb)
connect_to_milvus()
collection = hydrade_milvus(collection_name, client, vector_index_id, model_id_emb)

print("The Collection Name is: ", collection_name)
with open("collection_name.txt", "w") as file:
    file.write(collection_name)