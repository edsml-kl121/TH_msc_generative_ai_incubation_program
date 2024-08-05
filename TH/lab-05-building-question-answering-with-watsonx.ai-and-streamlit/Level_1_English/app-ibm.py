# Import environment loading library
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model 
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
# Import system libraries
import os
# Import streamlit for the UI 
import streamlit as st


# Load environment vars
load_dotenv()

# Define credentials 
api_key = os.getenv("WATSONX_APIKEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

# Define generation parameters 
params = {
    'decoding_method': "greedy",
    'min_new_tokens': 1,
    'max_new_tokens': 300,
    'random_seed': 42,
    # 'temperature': 0.2,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    'repetition_penalty': 1.05
}

# maybe we can add llama3 model here
models = {
    "llama3": "meta-llama/llama-3-70b-instruct",
    "granite_chat":"ibm/granite-13b-chat-v2",
    "flanul": "google/flan-ul2",
    "llama2": "meta-llama/llama-2-70b-chat",
    "mixstral": 'mistralai/mixtral-8x7b-instruct-v01'
}

def detect_language(text):
    thai_chars = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์")
    if any(char in thai_chars for char in text):
        return "th"
    else:
        return "en"
    
# input is List of Dict from the session state
def format_chat_history(session_state):
    # made up a chat history
    chat_history=""
    for turn in session_state:
        chat_history+=f"""
        {turn['role']}: {turn['content']}
        """
    return chat_history



def prompt_template(question, lang="en"):
    if lang == "en":
        text = f"""[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    QUESTION: {question} [/INST] ANSWER:"""
    elif lang == "th":
        text = f"""[INST] <<SYS>>
    คุณเป็นผู้ช่วยที่เป็นประโยชน์และให้ความเคารพ โปรดตอบอย่างเป็นประโยชน์ที่สุดเท่าที่จะเป็นไปได้เสมอโดยต้องปลอดภัย คำตอบของคุณไม่ควรมีเนื้อหาที่เป็นอันตราย ผิดจรรยาบรรณ เหยียดเชื้อชาติ เหยียดเพศ เป็นพิษ เป็นอันตราย หรือผิดกฎหมาย โปรดตรวจสอบให้แน่ใจว่าคำตอบของคุณมีลักษณะเป็นกลางทางสังคมและมีลักษณะเชิงบวก

     หากคำถามไม่สมเหตุสมผลหรือไม่สอดคล้องกันตามข้อเท็จจริง ให้อธิบายเหตุผลแทนที่จะตอบสิ่งที่ไม่ถูกต้อง หากคุณไม่ทราบคำตอบสำหรับคำถาม โปรดอย่าเปิดเผยข้อมูลที่เป็นเท็จ
    
     คุณจะได้รับคำถามจากผู้ใช้ใน ''' ด้านล่าง กรุณาตอบคำถามเป็นภาษาไทย
    <</SYS>>
    ```
    คำถาม: {question}
    ```
    คำถาม: {question} [/INST] คำตอบ:"""
    return text

# Title for the app
st.title('🤖 Our First Q&A Front End')
option = st.selectbox(
    "select model for Q&A",
    tuple(models),
)

model = Model(
    model_id=models[option],
    params=params,
    credentials=creds,
    project_id=project_id,
    space_id=None)

llm = WatsonxLLM(model)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_hist = format_chat_history(st.session_state.messages)

if prompt := st.chat_input("Enter your prompt here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    users_language = detect_language(prompt)
    if users_language == "th":
        current_prompt = prompt_template(prompt, lang="th")
    elif users_language == "en":
        current_prompt = prompt_template(prompt, lang="en")

    # this is the generation part of the model
    response = llm(chat_hist+current_prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
