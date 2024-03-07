import streamlit as st
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

import os

st.set_page_config("Demo LLM", "🦜")

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")


if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def get_client() -> MistralClient:
    return MistralClient(api_key=API_KEY)

client = get_client()

def chat(client:MistralClient, msg):
    messages = [] #ChatMessage(role=m['role'], content=m['content']) for m in st.session_state.history]
    messages.append(ChatMessage(role="user", content=msg))

    # print(messages, flush=True)

    response = ""

    for chunk in client.chat_stream(messages, model="mistral-small"):
        response += chunk.choices[0].delta.content
        yield chunk.choices[0].delta.content

    st.session_state.history.append(dict(role="ai", content=response))

for message in st.session_state.history:
    with st.chat_message(message['role']):
        st.write(message['content'])

query = st.chat_input()

if not query:
    st.stop()

with st.chat_message("user"):
    st.session_state.history.append(dict(role="user", content=query))
    st.write(query)

with st.chat_message("ai"):
    st.write_stream(chat(client, query))
