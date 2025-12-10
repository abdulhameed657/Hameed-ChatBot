
# import streamlit as st
# import requests
# import os
# import re
# import time
# import tempfile
# import mimetypes
# import pathlib
# import json
# import hashlib
# from dotenv import load_dotenv
# from codecs import getincrementaldecoder

# # LangChain imports
# from langchain_community.document_loaders import (
#     PDFPlumberLoader,
#     TextLoader,
#     CSVLoader,
#     Docx2txtLoader
# )
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_ollama import OllamaEmbeddings

# load_dotenv()

# # ---------------- CONFIG ----------------
# BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# DEFAULT_MODEL = "llama3.2:latest"
# DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
# USERS_FILE = "users.json"

# THINKING_STYLE = """
# <div style="
#     background: #f8f9fa;
#     border-left: 4px solid #dee2e6;
#     color: #6c757d;
#     padding: 0.5rem 1rem;
#     margin: 1rem 0;
#     border-radius: 0.25rem;
#     font-size: 0.9em;
# ">
# 🤔 <strong>Thinking:</strong><br/>
# {}
# </div>
# """

# RAG_TEMPLATE = """
# You are an assistant for question-answering tasks. Use the following context to answer the question. 
# If you don't know the answer, say you don't know. Be concise and helpful.

# Context: {context}

# Question: {question}

# Answer:
# """

# # ---------------- USER FUNCTIONS ----------------
# def load_users():
#     if os.path.exists(USERS_FILE):
#         with open(USERS_FILE, "r") as f:
#             return json.load(f)
#     return {}

# def save_users(users):
#     with open(USERS_FILE, "w") as f:
#         json.dump(users, f, indent=2)

# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def login_signup():
#     if "user" not in st.session_state:
#         st.session_state.user = None

#     users = load_users()
#     option = st.sidebar.selectbox("Login or Signup", ["Login", "Signup"])
#     username = st.sidebar.text_input("Username")
#     password = st.sidebar.text_input("Password", type="password")

#     if st.sidebar.button(option):
#         if not username or not password:
#             st.sidebar.error("Please enter both username and password")
#             return False

#         if option == "Signup":
#             if username in users:
#                 st.sidebar.error("Username already exists")
#             else:
#                 users[username] = {"password": hash_password(password), "chats": []}
#                 save_users(users)
#                 st.sidebar.success("Signup successful! Please login.")
#         elif option == "Login":
#             if username in users and users[username]["password"] == hash_password(password):
#                 st.session_state.user = username
#                 st.sidebar.success(f"Logged in as {username}")
#                 return True
#             else:
#                 st.sidebar.error("Invalid username or password")
#     return st.session_state.user is not None

# def get_user_chats(username):
#     users = load_users()
#     return users.get(username, {}).get("chats", [])

# def save_user_chats(username, chats):
#     users = load_users()
#     if username in users:
#         users[username]["chats"] = chats
#         save_users(users)

# # ---------------- SESSION ----------------
# def initialize_session():
#     defaults = {
#         "messages": [],
#         "model": DEFAULT_MODEL,
#         "embed_model": DEFAULT_EMBED_MODEL,
#         "vector_store": None,
#         "streaming": True,
#         "current_chat_index": None
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value

# # ---------------- BACKEND ----------------
# def get_models():
#     try:
#         response = requests.get(f"{BACKEND_URL}/models", timeout=5)
#         response.raise_for_status()
#         return response.json().get("models", [])
#     except Exception as e:
#         st.error(f"Error fetching models: {e}")
#         return []

# def chat_completion(messages, stream=False):
#     payload = {"model": st.session_state.model, "messages": messages, "stream": stream}
#     try:
#         return requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=220, stream=stream)
#     except Exception as e:
#         st.error(f"Chat request failed: {e}")
#         return None

# # ---------------- UTILITIES ----------------
# def process_response(content):
#     if not isinstance(content, str):
#         content = str(content)
#     think = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
#     answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
#     return think, answer

# def display_message(role, content):
#     with st.chat_message(role):
#         if role == "assistant":
#             think_blocks, answer = process_response(content)
#             for block in think_blocks:
#                 st.markdown(THINKING_STYLE.format(block), unsafe_allow_html=True)
#             st.markdown(answer)
#         else:
#             st.markdown(content)

# def process_documents(files):
#     try:
#         documents = []
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
#         for file in files:
#             if file.size > 200 * 1024 * 1024:
#                 st.error(f"{file.name} exceeds 200MB limit.")
#                 continue
#             suffix = pathlib.Path(file.name).suffix
#             tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
#             tmp.write(file.getbuffer())
#             tmp.close()
#             temp_path = tmp.name
#             mime = file.type or mimetypes.guess_type(file.name)[0]
#             ext = pathlib.Path(file.name).suffix.lower()
#             try:
#                 if mime == "application/pdf" or ext == ".pdf":
#                     loader = PDFPlumberLoader(temp_path)
#                 elif mime == "text/plain" or ext == ".txt":
#                     loader = TextLoader(temp_path)
#                 elif mime == "text/csv" or ext == ".csv":
#                     loader = CSVLoader(temp_path)
#                 elif ext == ".docx":
#                     loader = Docx2txtLoader(temp_path)
#                 else:
#                     st.error(f"Unsupported file: {file.name}")
#                     continue
#                 raw_docs = loader.load()
#                 docs = text_splitter.split_documents(raw_docs)
#                 documents.extend(docs)
#             except Exception as e:
#                 st.error(f"Error processing {file.name}: {e}")
#             finally:
#                 os.remove(temp_path)

#         if not documents:
#             return None

#         embeddings = OllamaEmbeddings(model=st.session_state.embed_model, base_url=ollama_host)
#         embeddings.embed_query("test connection")
#         return InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)

#     except Exception as e:
#         st.error(f"Document processing failed: {e}")
#         return None

# # ---------------- MAIN APP ----------------
# def main():
#     st.set_page_config(page_title="Hameed Chatbot", layout="centered")
    
#     # Gradient background for full app and chat messages
#     st.markdown("""
#     <style>
#     /* App background */
#     .stApp {
#         background: linear-gradient(135deg, #e60974, #000078);
#         background-attachment: fixed;
#     }

#     /* All chat messages */
#     .stChatMessage {
#         border-radius: 12px;
#         padding: 8px;
#         margin-bottom: 8px;
#     }

#     /* User messages gradient */
#     .stChatMessage[data-testid="stChatMessage-user"] {
#         background: linear-gradient(135deg, #fbc2eb, #a6c1ee);
#     }

#     /* Assistant messages gradient */
#     .stChatMessage[data-testid="stChatMessage-assistant"] {
#         background: linear-gradient(135deg, #89f7fe, #66a6ff);
#     }

#     /* Footer style */
#     .footer {
#         position: fixed;
#         bottom:0;
#         width:37%;
#         text-align:center;
#         padding:10px 0;
#         border-top:1px solid #eaeaea;
#         font-size:14px;
#         z-index:9999;
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     st.title("------------Mini Chatbot 🤖------------")

#     initialize_session()

#     if not login_signup():
#         st.stop()

#     chats = get_user_chats(st.session_state.user)
#     if st.session_state.current_chat_index is None:
#         st.session_state.current_chat_index = len(chats) - 1 if chats else None
#     if st.session_state.current_chat_index is not None and st.session_state.current_chat_index < len(chats):
#         st.session_state.messages = chats[st.session_state.current_chat_index]["messages"]
#     else:
#         st.session_state.messages = []

#     # ---------------- SIDEBAR ----------------
#     with st.sidebar:
#         st.header("Settings")
#         models = get_models()
#         if models:
#             names = [m.get("name") for m in models]
#             st.session_state.model = st.selectbox("Chat Model", names, index=names.index(DEFAULT_MODEL) if DEFAULT_MODEL in names else 0)
#             st.session_state.embed_model = st.selectbox("Embedding Model", names, index=names.index(DEFAULT_EMBED_MODEL) if DEFAULT_EMBED_MODEL in names else 0)

#         st.markdown("---")
#         uploaded = st.file_uploader("Upload Documents", type=["pdf","txt","csv","docx"], accept_multiple_files=True)
#         if uploaded:
#             with st.spinner("Processing documents..."):
#                 st.session_state.vector_store = process_documents(uploaded)

#         st.checkbox("Enable Streaming", value=st.session_state.get("streaming", True), key="streaming")
#         st.markdown("---")
#         st.subheader("Chats")
#         search_query = st.text_input("Search Chats", "")
#         for i, chat in enumerate(chats):
#             if search_query.lower() not in chat["title"].lower():
#                 continue
#             col1, col2 = st.columns([0.8,0.2])
#             with col1:
#                 if st.button(chat["title"], key=f"chat_{i}"):
#                     st.session_state.current_chat_index = i
#                     st.session_state.messages = chat["messages"]
#             with col2:
#                 if st.button("🗑️", key=f"del_{i}"):
#                     del chats[i]
#                     save_user_chats(st.session_state.user, chats)
#                     if st.session_state.current_chat_index == i:
#                         st.session_state.current_chat_index = None
#                         st.session_state.messages = []
#                     elif st.session_state.current_chat_index > i:
#                         st.session_state.current_chat_index -= 1
#                     st.experimental_rerun()

#         if st.button("➕ New Chat"):
#             new_chat = {"id":f"chat_{int(time.time())}","title":f"New Chat {time.strftime('%Y-%m-%d %H:%M:%S')}","messages":[]}
#             chats.append(new_chat)
#             save_user_chats(st.session_state.user, chats)
#             st.session_state.current_chat_index = len(chats) - 1
#             st.session_state.messages = []
#             st.experimental_rerun()

#     # ---------------- CHAT INTERFACE ----------------
#     for msg in st.session_state.messages:
#         display_message(msg["role"], msg["content"])

#     prompt = st.chat_input("How can I help you?")
#     if prompt:
#         st.session_state.messages.append({"role":"user","content":prompt})
#         display_message("user", prompt)

#         with st.chat_message("assistant"):
#             try:
#                 context = ""
#                 if st.session_state.vector_store:
#                     docs = st.session_state.vector_store.similarity_search(prompt, k=3)
#                     context = "\n\n".join(d.page_content for d in docs)

#                 final_prompt = RAG_TEMPLATE.format(context=context, question=prompt) if context else prompt
#                 messages = [{**m} for m in st.session_state.messages]
#                 messages[-1]["content"] = final_prompt
#                 full_response = ""

#                 if st.session_state.get("streaming", True):
#                     container = st.empty()
#                     buffer = ""
#                     decoder = getincrementaldecoder("utf-8")()
#                     resp = chat_completion(messages, stream=True)
#                     for chunk in resp.iter_content(chunk_size=1024):
#                         if not chunk: continue
#                         text = decoder.decode(chunk)
#                         buffer += text
#                         while "<think>" in buffer and "</think>" in buffer:
#                             start = buffer.index("<think>")
#                             end = buffer.index("</think>")
#                             think_text = buffer[start+7:end]
#                             st.markdown(THINKING_STYLE.format(think_text), unsafe_allow_html=True)
#                             buffer = buffer[end+8:]
#                         container.markdown(buffer + "▌")
#                     buffer += decoder.decode(b"", final=True)
#                     full_response = buffer
#                     container.markdown(full_response)
#                 else:
#                     resp = chat_completion(messages)
#                     data = resp.json()
#                     full_response = data.get("response", str(data))
#                     think, answer = process_response(full_response)
#                     for block in think:
#                         st.markdown(THINKING_STYLE.format(block), unsafe_allow_html=True)
#                     st.markdown(answer)

#                 st.session_state.messages.append({"role":"assistant","content":full_response})
#                 if st.session_state.current_chat_index is not None and st.session_state.current_chat_index < len(chats):
#                     chats[st.session_state.current_chat_index]["messages"] = st.session_state.messages
#                     save_user_chats(st.session_state.user, chats)
#             except Exception as e:
#                 st.error(f"Response error: {e}")

#     # ---------------- FOOTER ----------------
#     st.markdown("""
#         <div class="footer">Developed By <strong>Abdul Hameed Rajput</strong></div>
#     """, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()

import streamlit as st
import requests
import os
import re
import time
import tempfile
import mimetypes
import pathlib
import json
import hashlib
from dotenv import load_dotenv
from codecs import getincrementaldecoder

# LangChain imports
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# ---------------- CONFIG ----------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
USERS_FILE = "users.json"

THINKING_STYLE = """
<div style="
    background: #f8f9fa;
    border-left: 4px solid #dee2e6;
    color: #6c757d;
    padding: 0.5rem 1rem;
    margin: 1rem 0;
    border-radius: 0.25rem;
    font-size: 0.9em;
">
🤔 <strong>Thinking:</strong><br/>
{}
</div>
"""

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following context to answer the question. 
If you don't know the answer, say you don't know. Be concise and helpful.

Context: {context}

Question: {question}

Answer:
"""

# ---------------- USER FUNCTIONS ----------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_signup():
    if "user" not in st.session_state:
        st.session_state.user = None

    users = load_users()
    option = st.sidebar.selectbox("Login or Signup", ["Login", "Signup"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    # Guest button
    if st.sidebar.button("Play as Guest"):
        st.session_state.user = f"guest_{int(time.time())}"
        st.sidebar.success(f"Playing as Guest")
        return True

    if st.sidebar.button(option):
        if not username or not password:
            st.sidebar.error("Please enter both username and password")
            return False

        if option == "Signup":
            if username in users:
                st.sidebar.error("Username already exists")
            else:
                users[username] = {"password": hash_password(password), "chats": []}
                save_users(users)
                st.sidebar.success("Signup successful! Please login.")
        elif option == "Login":
            if username in users and users[username]["password"] == hash_password(password):
                st.session_state.user = username
                st.sidebar.success(f"Logged in as {username}")
                return True
            else:
                st.sidebar.error("Invalid username or password")
    return st.session_state.user is not None

def get_user_chats(username):
    users = load_users()
    return users.get(username, {}).get("chats", [])

def save_user_chats(username, chats):
    users = load_users()
    if username in users:
        users[username]["chats"] = chats
        save_users(users)

# ---------------- SESSION ----------------
def initialize_session():
    defaults = {
        "messages": [],
        "model": DEFAULT_MODEL,
        "embed_model": DEFAULT_EMBED_MODEL,
        "vector_store": None,
        "streaming": True,
        "current_chat_index": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ---------------- BACKEND ----------------
def get_models():
    try:
        response = requests.get(f"{BACKEND_URL}/models", timeout=5)
        response.raise_for_status()
        return response.json().get("models", [])
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def chat_completion(messages, stream=False):
    payload = {"model": st.session_state.model, "messages": messages, "stream": stream}
    try:
        return requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=220, stream=stream)
    except Exception as e:
        st.error(f"Chat request failed: {e}")
        return None

# ---------------- UTILITIES ----------------
def process_response(content):
    if not isinstance(content, str):
        content = str(content)
    think = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
    answer = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return think, answer

def display_message(role, content):
    with st.chat_message(role):
        if role == "assistant":
            think_blocks, answer = process_response(content)
            for block in think_blocks:
                st.markdown(THINKING_STYLE.format(block), unsafe_allow_html=True)
            st.markdown(answer)
        else:
            st.markdown(content)

def process_documents(files):
    try:
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        for file in files:
            if file.size > 200 * 1024 * 1024:
                st.error(f"{file.name} exceeds 200MB limit.")
                continue
            suffix = pathlib.Path(file.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(file.getbuffer())
            tmp.close()
            temp_path = tmp.name
            mime = file.type or mimetypes.guess_type(file.name)[0]
            ext = pathlib.Path(file.name).suffix.lower()
            try:
                if mime == "application/pdf" or ext == ".pdf":
                    loader = PDFPlumberLoader(temp_path)
                elif mime == "text/plain" or ext == ".txt":
                    loader = TextLoader(temp_path)
                elif mime == "text/csv" or ext == ".csv":
                    loader = CSVLoader(temp_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(temp_path)
                else:
                    st.error(f"Unsupported file: {file.name}")
                    continue
                raw_docs = loader.load()
                docs = text_splitter.split_documents(raw_docs)
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
            finally:
                os.remove(temp_path)

        if not documents:
            return None

        embeddings = OllamaEmbeddings(model=st.session_state.embed_model, base_url=ollama_host)
        embeddings.embed_query("test connection")
        return InMemoryVectorStore.from_documents(documents=documents, embedding=embeddings)

    except Exception as e:
        st.error(f"Document processing failed: {e}")
        return None

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="Hameed Chatbot", layout="centered")
    
    # Gradient background for full app and chat messages
    st.markdown("""
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #e60974, #000078);
        background-attachment: fixed;
    }

    /* All chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 8px;
    }

    /* User messages gradient */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, #fbc2eb, #a6c1ee);
    }

    /* Assistant messages gradient */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #89f7fe, #66a6ff);
    }

    /* Footer style */
    .footer {
        position: fixed;
        bottom:0;
        width:37%;
        text-align:center;
        padding:10px 0;
        border-top:1px solid #eaeaea;
        font-size:14px;
        z-index:9999;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("------------Mini Chatbot 🤖------------")

    initialize_session()

    if not login_signup():
        st.stop()

    chats = get_user_chats(st.session_state.user)
    if st.session_state.current_chat_index is None:
        st.session_state.current_chat_index = len(chats) - 1 if chats else None
    if st.session_state.current_chat_index is not None and st.session_state.current_chat_index < len(chats):
        st.session_state.messages = chats[st.session_state.current_chat_index]["messages"]
    else:
        st.session_state.messages = []

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("Settings")
        models = get_models()
        if models:
            names = [m.get("name") for m in models]
            st.session_state.model = st.selectbox("Chat Model", names, index=names.index(DEFAULT_MODEL) if DEFAULT_MODEL in names else 0)
            st.session_state.embed_model = st.selectbox("Embedding Model", names, index=names.index(DEFAULT_EMBED_MODEL) if DEFAULT_EMBED_MODEL in names else 0)

        st.markdown("---")
        uploaded = st.file_uploader("Upload Documents", type=["pdf","txt","csv","docx"], accept_multiple_files=True)
        if uploaded:
            with st.spinner("Processing documents..."):
                st.session_state.vector_store = process_documents(uploaded)

        st.checkbox("Enable Streaming", value=st.session_state.get("streaming", True), key="streaming")
        st.markdown("---")
        st.subheader("Chats")
        search_query = st.text_input("Search Chats", "")
        for i, chat in enumerate(chats):
            if search_query.lower() not in chat["title"].lower():
                continue
            col1, col2 = st.columns([0.8,0.2])
            with col1:
                if st.button(chat["title"], key=f"chat_{i}"):
                    st.session_state.current_chat_index = i
                    st.session_state.messages = chat["messages"]
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    del chats[i]
                    save_user_chats(st.session_state.user, chats)
                    if st.session_state.current_chat_index == i:
                        st.session_state.current_chat_index = None
                        st.session_state.messages = []
                    elif st.session_state.current_chat_index > i:
                        st.session_state.current_chat_index -= 1
                    st.experimental_rerun()

        if st.button("➕ New Chat"):
            new_chat = {"id":f"chat_{int(time.time())}","title":f"New Chat {time.strftime('%Y-%m-%d %H:%M:%S')}","messages":[]}
            chats.append(new_chat)
            save_user_chats(st.session_state.user, chats)
            st.session_state.current_chat_index = len(chats) - 1
            st.session_state.messages = []
            st.experimental_rerun()

    # ---------------- CHAT INTERFACE ----------------
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    prompt = st.chat_input("How can I help you?")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        display_message("user", prompt)

        with st.chat_message("assistant"):
            try:
                context = ""
                if st.session_state.vector_store:
                    docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                    context = "\n\n".join(d.page_content for d in docs)

                final_prompt = RAG_TEMPLATE.format(context=context, question=prompt) if context else prompt
                messages = [{**m} for m in st.session_state.messages]
                messages[-1]["content"] = final_prompt
                full_response = ""

                if st.session_state.get("streaming", True):
                    container = st.empty()
                    buffer = ""
                    decoder = getincrementaldecoder("utf-8")()
                    resp = chat_completion(messages, stream=True)
                    for chunk in resp.iter_content(chunk_size=1024):
                        if not chunk: continue
                        text = decoder.decode(chunk)
                        buffer += text
                        while "<think>" in buffer and "</think>" in buffer:
                            start = buffer.index("<think>")
                            end = buffer.index("</think>")
                            think_text = buffer[start+7:end]
                            st.markdown(THINKING_STYLE.format(think_text), unsafe_allow_html=True)
                            buffer = buffer[end+8:]
                        container.markdown(buffer + "▌")
                    buffer += decoder.decode(b"", final=True)
                    full_response = buffer
                    container.markdown(full_response)
                else:
                    resp = chat_completion(messages)
                    data = resp.json()
                    full_response = data.get("response", str(data))
                    think, answer = process_response(full_response)
                    for block in think:
                        st.markdown(THINKING_STYLE.format(block), unsafe_allow_html=True)
                    st.markdown(answer)

                st.session_state.messages.append({"role":"assistant","content":full_response})
                if st.session_state.current_chat_index is not None and st.session_state.current_chat_index < len(chats):
                    chats[st.session_state.current_chat_index]["messages"] = st.session_state.messages
                    save_user_chats(st.session_state.user, chats)
            except Exception as e:
                st.error(f"Response error: {e}")

    # ---------------- FOOTER ----------------
    st.markdown("""
        <div class="footer">Developed By <strong>Abdul Hameed Rajput</strong></div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
