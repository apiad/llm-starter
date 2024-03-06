import streamlit as st
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage, Function

import os

st.set_page_config("Demo LLM", "🦜")

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

@st.cache_resource
def get_client() -> MistralClient:
    return MistralClient(api_key=API_KEY)

client = get_client()

def chat(client:MistralClient, msg):
    messages = [ChatMessage(role="user", content=msg)]

    for chunk in client.chat_stream(messages, model="mistral-small"):
        yield chunk.choices[0].delta.content

query = st.chat_input()

if not query:
    st.stop()

with st.chat_message("user"):
    st.write(query)

with st.chat_message("ai"):
    st.write_stream(chat(client, query))
