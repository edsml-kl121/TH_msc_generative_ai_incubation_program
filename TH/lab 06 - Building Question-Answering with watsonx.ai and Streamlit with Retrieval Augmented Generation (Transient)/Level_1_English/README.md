# Building  Apps with watsonx.ai and Streamlit
So I'm guessing you've been hearing a bit about watsonx. Well...now you can build your very own app with it🙌 (I know...crazy right?!). In this tutorial you'll learn how to build your own LLM powered Streamlit with the Watson Machine Learning library.  

# Set up your vector Database
1. Open your terminal or console window
2. cd into this lab's base directory then locate to INGESTION folder
3. put `cert.pem` file inside the INGESTION folder
4. Fill in the embedding model you used followed by the vector index id in `ingestion.py`
4. Start ingesting documents from watsonx to the platform using `python ingestion.py`
5. Here `collection_name.txt` should be generated, please keep note of this collection name

Starting the app
1. cd into this lab's base directory then locate to MAINAPP folder
2. put `cert.pem` file inside the MAINAPP folder
3. edit the settings section in `app.py` and change the model_id_emb, vector_index_id, collection_name (got in previous step)
4. Run the app by running the command `streamlit run app.py`.
