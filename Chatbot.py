import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from datetime import datetime
import json

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Page Config
st.set_page_config(
    page_title="💎 Conversational AI",
    page_icon="🤖",
    layout="wide"
)

# Title with style
st.markdown(
    """
    <h1 style="text-align:center; color:#6C63FF;">💎 Conversational Chatbot</h1>
    <p style="text-align:center; color:gray;">Made By Mohsin Raza</p>
    """,
    unsafe_allow_html=True,
)

# Sidebar – Settings
st.sidebar.header("⚙️ Chatbot Settings")

with st.sidebar.expander("🤖 Model Configuration", expanded=True):
    model_name = st.selectbox(
        "Select Model",
        ["gemma2-9b-it", "deepseek-r1-distill-llama-70b", "openai/gpt-oss-120b"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 300, 150)

with st.sidebar.expander("🧹 Conversation Controls", expanded=True):
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.success("Chat history cleared!")

with st.sidebar.expander("📤 Export Options"):
    if st.button("Download Chat as JSON"):
        if "history" in st.session_state:
            chat_data = json.dumps(st.session_state.history, indent=4)
            st.download_button(
                "Save JSON",
                chat_data,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Chat Input
user_input = st.chat_input("💬 Type your message here...")

if user_input:
    # Append user input to history
    st.session_state.history.append(("user", user_input))

    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=True
    )

    ai_response = conv.predict(input=user_input)

    # Append AI response
    st.session_state.history.append(("assistant", ai_response))

# Display Chat
st.subheader("📝 Conversation")
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Show Stats
with st.expander("📊 Conversation Stats"):
    st.write(f"Total messages: {len(st.session_state.history)}")
    st.write(f"Selected Model: {model_name}")
    st.write(f"Temperature: {temperature}")
    st.write(f"Max Tokens: {max_tokens}")

