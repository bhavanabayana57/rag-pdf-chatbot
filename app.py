import streamlit as st
import fitz
import faiss
import numpy as np
import os
import pickle
import shutil
import time
import json
import streamlit.components.v1 as components
import uuid
import json
import hashlib
import easyocr
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.markdown("""
<style>

div[data-testid="stButton"] button[kind="secondary"]{
    border-radius:12px;
}
            
/* Input height */
div[data-testid="stTextInput"] input{
    height:49px;
}

/* Mic button height */
div[data-testid="stHorizontalBlock"] button{
    height:52px;
}

</style>
""", unsafe_allow_html=True)

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from PIL import Image
from pdf2image import convert_from_bytes
from streamlit_mic_recorder import (
    mic_recorder,
    speech_to_text
)
# -------------------- CONFIG --------------------

BASE_CHAT_DIR = "chats"
os.makedirs(BASE_CHAT_DIR, exist_ok=True)

USERS_FILE = "users.json"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

def hash_password(password):
    return hashlib.sha256(
        password.encode()
    ).hexdigest()

SHARED_DIR = "shared_chats"
os.makedirs(SHARED_DIR, exist_ok=True)

st.set_page_config(
    page_title="RAG PDF Chatbot",
    layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

query_params = st.query_params

if "share" in query_params:

    share_id = query_params["share"]

    share_file = (
        f"{SHARED_DIR}/{share_id}.json"
    )

    if os.path.exists(share_file):

        with open(
            share_file,
            "r",
            encoding="utf-8"
        ) as f:

            shared_chat = json.load(f)

        st.title("Shared Conversation")

        for msg in shared_chat["messages"]:

            with st.chat_message(
                msg["role"]
            ):
                st.markdown(
                    msg["content"]
                )

        st.stop()
# -------------------- LOAD ENV --------------------

load_dotenv()

groq_key = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)

client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1"
)

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = get_ocr_reader()

def extract_text_from_image(img):

    img_np = np.array(img)

    result = reader.readtext(img_np)

    text = " ".join([item[1] for item in result])

    return text

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-mpnet-base-v2"
    )

model = load_embedding_model()

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
# -------------------- PDF PROCESSING --------------------

@st.cache_data
def process_pdf(file_bytes):

    pdf_document = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    documents = []

    for page_num, page in enumerate(pdf_document):

        text = page.get_text().strip()

        if len(text) < 50:

            page_image = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img = Image.frombytes(
                "RGB",
                [page_image.width, page_image.height],
                page_image.samples
            )

            text = extract_text_from_image(img)

        if not text.strip():
            continue

        documents.append({
            "text": text,
            "page": page_num + 1
        })

    return documents

# -------------------- SESSION STATE --------------------

if "current_chat" not in st.session_state:

    st.session_state.current_chat = "Chat 1"

if "chat_sessions" not in st.session_state:

    st.session_state.chat_sessions = {}

# -------------------- LOAD SAVED CHATS --------------------

current_user = st.session_state.username

CHAT_DIR = os.path.join(
    BASE_CHAT_DIR,
    current_user
)

os.makedirs(CHAT_DIR, exist_ok=True)

existing_chats = sorted([
    folder for folder in os.listdir(CHAT_DIR)
    if os.path.isdir(os.path.join(CHAT_DIR, folder))
])

for chat_name in existing_chats:

    data_path = os.path.join(
        CHAT_DIR,
        chat_name,
        "data.pkl"
    )

    if os.path.exists(data_path):

        with open(data_path, "rb") as f:
            chat_data = pickle.load(f)


            if "chunks" in chat_data:

                tokenized_chunks = [
                    chunk.split()
                    for chunk in chat_data["chunks"]
                ]

            chat_data["bm25"] = BM25Okapi(tokenized_chunks)

        st.session_state.chat_sessions[chat_name] = chat_data

    if chat_name not in st.session_state.chat_sessions:

        st.session_state.chat_sessions[chat_name] = {}

        current_chat = st.session_state.current_chat

    chat_data = st.session_state.chat_sessions.get(
        current_chat,
        {
            "messages": [],
            "pdf_name": "",
            "chunks": [],
            "chunk_pages": [],
            "index": None
        }
    )

    if "index_path" in chat_data:

            if os.path.exists(chat_data["index_path"]):

                chat_data["index"] = faiss.read_index(
                    chat_data["index_path"]
                )    

# -------------------- FIRST CHAT --------------------

if len(st.session_state.chat_sessions) == 0:

    st.session_state.chat_sessions["Chat 1"] = {

        "messages": [],
        "pdf_name": "",
        "pdf_bytes": None,
        "chunks": [],
        "chunk_pages": [],
        "index": None,
        "index_path": ""
    }

    os.makedirs(
        os.path.join(CHAT_DIR, "Chat 1"),
        exist_ok=True
    )

# -------------------- SIDEBAR --------------------

if not st.session_state.logged_in:

        st.title("🔐 Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button("Login"):

                with open(USERS_FILE, "r") as f:
                    users = json.load(f)

                if (
                    username in users
                    and users[username]
                    == hash_password(password)
                ):

                    st.session_state.logged_in = True

                    st.session_state.username = username

                    st.session_state.chat_sessions = {}
                    st.session_state.current_chat = "Chat 1"

                    st.rerun()

                else:

                    st.error("Invalid Username or Password")

        with col2:

            if st.button("Register"):

                with open(USERS_FILE, "r") as f:
                    users = json.load(f)

                if username in users:

                    st.error("User already exists")

                else:

                    users[username] = hash_password(password)

                    with open(
                        USERS_FILE,
                        "w"
                    ) as f:

                        json.dump(
                            users,
                            f,
                            indent=4
                        )

                    st.success(
                        "Registration Successful"
                    )

        if not st.session_state.logged_in:
            st.stop()                

st.sidebar.write(
    f"👤 {st.session_state.username}"
)

with st.sidebar:

    if st.button("Logout"):

        st.session_state.clear()

        st.rerun()

    st.title("Chats")

    # ---------- NEW CHAT ----------

    if st.button("➕ New Chat", use_container_width=True):

        new_chat = f"Chat {len(st.session_state.chat_sessions)+1}"

        st.session_state.chat_sessions[new_chat] = {

            "messages": [],
            "pdf_name": "",
            "pdf_bytes": None,
            "chunks": [],
            "chunk_pages": [],
            "index": None,
            "index_path": ""
        }

        os.makedirs(
            os.path.join(CHAT_DIR, new_chat),
            exist_ok=True
        )

        st.session_state.pop("voice_question", None)
        st.session_state.pop("pending_question", None)
        st.session_state.pop("last_voice_text", None)

        st.session_state.current_chat = new_chat

        st.rerun()

    st.subheader("Previous Chats")
     
    # ---------- RENAME CHAT ----------

    new_name = st.text_input("Rename Current Chat")

    if st.button("Rename Chat"):

        old_name = st.session_state.current_chat

        old_path = os.path.join(CHAT_DIR, old_name)

        new_path = os.path.join(CHAT_DIR, new_name)

        os.rename(old_path, new_path)

        st.session_state.chat_sessions[new_name] = (
            st.session_state.chat_sessions.pop(old_name)
        )

        st.session_state.current_chat = new_name

        st.rerun()

    # ---------- CHAT LIST ----------

    for chat_name in existing_chats:

        col1, col2 = st.columns([4, 1])

        with col1:

            if st.button(
                chat_name,
                key=f"chat_{chat_name}",
                use_container_width=True
            ):
               st.session_state.pop("voice_question", None)
               st.session_state.pop("pending_question", None)
               st.session_state.pop("last_voice_text", None)

               st.session_state.current_chat = new_chat

               st.rerun()

        with col2:

            if st.button(
                "🗑",
                key=f"delete_{chat_name}",
                use_container_width=True
            ):

                chat_path = os.path.join(CHAT_DIR, chat_name)

                if os.path.exists(chat_path):

                    shutil.rmtree(chat_path)

                del st.session_state.chat_sessions[chat_name]

                if st.session_state.current_chat == chat_name:

                    st.session_state.current_chat = "Chat 1"

                st.rerun()
# -------------------- CURRENT CHAT --------------------

# ---------------- CURRENT CHAT ----------------

current_chat = st.session_state.current_chat

chat_data = st.session_state.chat_sessions.get(
    current_chat,
    {
        "messages": [],
        "pdf_name": "",
        "pdf_path": "",
        "chunks": [],
        "chunk_pages": [],
        "index": None,
        "index_path": ""
    }
)

# -------- LOAD FAISS INDEX --------

index = None

if "index" in chat_data:
    index = chat_data["index"]

if index is None:

    if "index_path" in chat_data:

        if os.path.exists(chat_data["index_path"]):

            index = faiss.read_index(
                chat_data["index_path"]
            )

            chat_data["index"] = index

if "index" not in chat_data:
    chat_data["index"] = None

if "chunks" not in chat_data:
    chat_data["chunks"] = []

if "chunk_pages" not in chat_data:
    chat_data["chunk_pages"] = []

if "messages" not in chat_data:
    chat_data["messages"] = []

if "pdf_name" not in chat_data:
    chat_data["pdf_name"] = ""

if "pdf_path" not in chat_data:
    chat_data["pdf_path"] = ""

if "index_path" not in chat_data:
    chat_data["index_path"] = ""

if (
    chat_data["index"] is None
    and chat_data["index_path"] != ""
    and os.path.exists(chat_data["index_path"])
):

    chat_data["index"] = faiss.read_index(
        chat_data["index_path"]
    )
# -------------------- TITLE --------------------

if not st.session_state.logged_in:

    st.title("🔐 Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Login"):

            with open(USERS_FILE, "r") as f:
                users = json.load(f)

            if (
                username in users
                and users[username]
                == hash_password(password)
            ):

                st.session_state.logged_in = True

                st.session_state.username = username

                st.rerun()

            else:

                st.error("Invalid Username or Password")

    with col2:

        if st.button("Register"):

            with open(USERS_FILE, "r") as f:
                users = json.load(f)

            if username in users:

                st.error("User already exists")

            else:

                users[username] = hash_password(password)

                with open(
                    USERS_FILE,
                    "w"
                ) as f:

                    json.dump(
                        users,
                        f,
                        indent=4
                    )

                st.success(
                    "Registration Successful"
                )

    if not st.session_state.logged_in:
        st.stop()                

colA, colB = st.columns([10,2])

with colA:
    st.title("RAG PDF Chatbot")

with colB:
    if st.button("📤 Share", key="share_btn"):

        share_id = uuid.uuid4().hex

        share_data = {
            "messages": chat_data.get("messages", [])
        }

        with open(
            f"{SHARED_DIR}/{share_id}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                share_data,
                f,
                ensure_ascii=False,
                indent=4
            )

        base_url = "https://bhavana-rag-chatbot.streamlit.app"
        share_link = f"https://bhavana-rag-chatbot.streamlit.app/?share={share_id}"

        st.success("Share link created!")
        st.code(share_link)

st.caption(f"Research Thread: {st.session_state.current_chat}")

if chat_data["pdf_name"] != "":


    st.subheader("Ask questions from your PDF using AI")

# -------------------- FILE UPLOAD --------------------

# ---------------- PDF SECTION ----------------

if chat_data["pdf_name"]:

    st.info(f"Uploaded PDF: {chat_data['pdf_name']}")

    uploaded_file = None

else:

    uploaded_file = st.file_uploader(
        "Upload PDF or Image",
        type=["pdf", "png", "jpg", "jpeg"],
        key=current_chat
    )

    # ================= VOICE CONTAINER =================

voice_text = None

voice_container = st.container()

with voice_container:

    st.markdown("### 🎤 Voice Input")

    voice_text = speech_to_text(
        language="en",
        just_once=False,
        key=f"voice_{current_chat}"
    )

    st.write("VOICE TEXT RAW:", voice_text)

    if voice_text:

        if st.session_state.get("last_voice_text") != voice_text:

            st.session_state.voice_question = voice_text

            st.session_state.last_voice_text = voice_text

if uploaded_file is not None and chat_data["index"] is None:

    # ---------- ONLY PROCESS NEW PDF ----------

    if isinstance(uploaded_file, str) is False:

        file_bytes = uploaded_file.read()

        # ---------- SAVE PDF TO DISK ----------

        chat_path = os.path.join(CHAT_DIR, current_chat)

        os.makedirs(chat_path, exist_ok=True)

        pdf_path = os.path.join(chat_path, uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        chat_data["pdf_path"] = pdf_path

        # ---------- SAVE PDF ----------

        chat_data["pdf_name"] = uploaded_file.name

        chat_data["messages"] = []
        messages = []

        st.session_state.pop("voice_question", None)
        st.session_state.pop("pending_question", None)
        st.session_state.pop("last_voice_text", None)

        chat_data["pdf_bytes"] = file_bytes

        # ---------- PROCESS ----------

        chat_data["messages"] = []
        chat_data["chunks"] = []
        chat_data["chunk_pages"] = []
        chat_data["index"] = None
        chat_data["bm25"] = None

        if uploaded_file.type.startswith("image"):

            image = Image.open(uploaded_file)

            text = extract_text_from_image(image)

            if not text.strip():
                st.error("No text extracted from image")
                st.stop()

            documents = [{
                "text": text,
                "page": 1
            }]
        else:

            documents = process_pdf(file_bytes)

            save_data = chat_data.copy()

            with open(
                os.path.join(CHAT_DIR, current_chat, "data.pkl"),
                "wb"
            ) as f:
                pickle.dump(save_data, f)
      


        # ---------- CHUNKING ----------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )

        chunks = []

        chunk_pages = []

        for doc in documents:

            split_chunks = text_splitter.split_text(doc["text"])

            for chunk in split_chunks:

                chunks.append(chunk)

                chunk_pages.append(doc["page"])

        # ---------- EMBEDDINGS ----------

        if len(chunks) == 0:
            st.error("No text extracted from PDF")
            st.stop()

        embeddings = model.encode(chunks)

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        # ---------- FAISS ----------

        embeddings = np.array(embeddings).astype("float32")

        dimension = len(embeddings[0])

        index = faiss.IndexFlatL2(dimension)

        chat_path = os.path.join(CHAT_DIR, current_chat)

        os.makedirs(chat_path, exist_ok=True)

        index_path = os.path.join(chat_path, "faiss.index")

        faiss.write_index(index, index_path)

        chat_data["index_path"] = index_path

        save_data = chat_data.copy()

        save_data["index"] = None

        data_path = os.path.join(chat_path, "data.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(save_data, f)

        embeddings = np.array(embeddings).astype("float32")

        index.add(embeddings)

        chat_path = os.path.join(CHAT_DIR, current_chat)

        os.makedirs(chat_path, exist_ok=True)

        index_path = os.path.join(chat_path, "faiss.index")

        faiss.write_index(index, index_path)

        chat_data["index_path"] = index_path

        

        save_data = chat_data.copy()

        save_data["index"] = None

        data_path = os.path.join(chat_path, "data.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(save_data, f)

        # ---------- STORE ----------

        chat_path = os.path.join(CHAT_DIR, current_chat)

        chat_data["chunks"] = chunks

        chat_data["chunk_pages"] = chunk_pages

        tokenized_chunks = [chunk.split() for chunk in chunks]

        bm25 = BM25Okapi(tokenized_chunks)

        chat_data["bm25"] = bm25

        chat_data["index"] = index

        chat_path = os.path.join(CHAT_DIR, current_chat)

        os.makedirs(chat_path, exist_ok=True)

        index_path = os.path.join(chat_path, "faiss.index")

        faiss.write_index(index, index_path)

        chat_data["index_path"] = index_path

        save_data = chat_data.copy()
        save_data["index"] = None

        data_path = os.path.join(chat_path, "data.pkl")

        with open(data_path, "wb") as f:
            pickle.dump(save_data, f)

        st.success("PDF uploaded successfully!")    

        st.rerun()

        # ---------- SAVE CHAT ----------

        save_data = {

            "messages": chat_data["messages"],
            "pdf_name": chat_data["pdf_name"],
            "pdf_bytes": chat_data["pdf_bytes"],
            "chunks": chat_data["chunks"],
            "chunk_pages": chat_data["chunk_pages"]
        }

        with open(
            os.path.join(CHAT_DIR, current_chat, "data.pkl"),
            "wb"
        ) as f:

            pickle.dump(save_data, f)

    # -------- LOAD FAISS INDEX --------


if "index_path" not in chat_data:
    chat_data["index_path"] = ""

if (
    chat_data["index"] is None
    and chat_data["index_path"] != ""
    and os.path.exists(chat_data["index_path"])
):

    chat_data["index"] = faiss.read_index(
        chat_data["index_path"]
    )
# ---------------- CHAT VARIABLES ----------------

messages = chat_data["messages"]

st.write("MESSAGES LENGTH:", len(messages))

chunks = chat_data["chunks"]

chunk_pages = chat_data["chunk_pages"]

index = None

if chat_data["index"] is None:

    if "index_path" in chat_data:

        if os.path.exists(chat_data["index_path"]):

            chat_data["index"] = faiss.read_index(
                chat_data["index_path"]
            )

index = chat_data["index"]

# -------------------- SHOW CHAT HISTORY --------------------

for message in messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "page" in message:

            st.caption(f"Source Pages: {message['page']}")
# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask a question from PDF"
)

st.write("VOICE:", st.session_state.get("voice_question"))
st.write("QUESTION:", question)

# ---------------- USER QUESTION ----------------

user_question = None

if question:
    user_question = question

elif "voice_question" in st.session_state:
    user_question = st.session_state.voice_question

    st.session_state.pop("voice_question", None)
    st.session_state.pop("pending_question", None)
    st.session_state.pop("last_voice_text", None)

    st.write("VOICE:", st.session_state.get("voice_question"))
    st.write("QUESTION:", question)
    st.write("USER QUESTION:", user_question)
    
if user_question:

    if chat_data["index"] is None:

        if (
            "index_path" in chat_data
            and chat_data["index_path"] != ""
            and os.path.exists(chat_data["index_path"])
        ):

            chat_data["index"] = faiss.read_index(
                chat_data["index_path"]
            )

    index = chat_data["index"]


    if index is None:
        st.error("FAISS index not loaded")
        st.stop()

    chunks = chat_data["chunks"]

    chunk_pages = chat_data["chunk_pages"]

    bm25 = chat_data["bm25"]

    general_questions = [
        "what this pdf",
        "what is this pdf",
        "what this document",
        "summary",
        "overview",
        "briefly say"
    ]

    if any(q in user_question.lower() for q in general_questions):

        retrieved_chunks = chunks[:20]

        retrieved_pages = chunk_pages[:20]

    # USER MESSAGE

    if user_question is None:
        st.stop()

    messages.append({
        "role": "user",
        "content": user_question
    })

    chat_data["messages"] = messages

    conversation_history = []

    for msg in messages[-6:]:

        conversation_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

        if len(messages) == 1:
            recent_context = ""
        else:
            recent_context = " ".join(
                msg["content"]
                for msg in messages[:-1]
                if msg["role"] == "user"
            )[-500:]

    context_question = f"""
    Previous conversation:
    {recent_context}

    Current question:
    {user_question}

    Rewrite the current question using the previous conversation.

    If the question contains pronouns such as:
    - it
    - its
    - they
    - them
    - this
    - that

    replace them with the actual topic from the conversation.

    Only return the rewritten standalone question.
    Do not explain anything.
    """

    ##rewrite_response = client.chat.completions.create(
      #  model="llama-3.1-8b-instant",
       # messages=[
        #    {
         #       "role": "user",
          #      "content": context_question
           # }
        #]
    #)

    #user_question= rewrite_response.choices[0].message.content.strip()

    question_embedding = model.encode([user_question])

    # FAISS SEARCH
    D, I = index.search(question_embedding, k=5)

    faiss_results = I[0]

    avg_distance = sum(D[0]) / len(D[0])

# BM25 SEARCH
    tokenized_query = user_question.split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_results = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:5]

# COMBINE RESULTS
    combined_results = list(dict.fromkeys(
        list(faiss_results) + bm25_results
    ))

# TOP RESULTS
    top_results = combined_results[:5]

    retrieved_chunks = []

    retrieved_pages = []

    for idx in top_results:

        if idx < len(chunks):

            retrieved_chunks.append(chunks[idx])

            retrieved_pages.append(chunk_pages[idx])  
    
    # -------- RERANKING --------

    pairs = [
        (user_question, chunk)
        for chunk in retrieved_chunks
    ]

    # SUMMARY QUESTION CHECK
    summary_questions = [
        "what is this pdf about",
        "what does this pdf define",
        "what this pdf defines",
        "what this pdf actually defines",
        "briefly say about this",
        "what is this document",
        "summary",
        "overview",
        "explain this pdf",
        "summarize this pdf"
    ]

    is_summary_question = any(
        q in user_question.lower()
        for q in summary_questions
    )

    if is_summary_question:
        retrieved_chunks = chunks[:min(50, len(chunks))]
        retrieved_pages = chunk_pages[:min(50, len(chunk_pages))]

    scores = reranker.predict(pairs)

    best_score = max(scores)

    if best_score < -999 and not is_summary_question:
        st.error(
            "This question does not appear to be related to the uploaded PDF."
        )
        st.stop()

    ranked_chunks = sorted(
        zip(scores, retrieved_chunks, retrieved_pages),
        reverse=True
    )

    retrieved_chunks = [
        chunk
        for score, chunk, page in ranked_chunks[:10]
    ]

    retrieved_pages = [
        page
        for score, chunk, page in ranked_chunks[:5]
    ]

    # CONTEXT

    if not retrieved_chunks:
        retrieved_chunks = chunks[:5]

    context = "\n\n".join(retrieved_chunks[:8])

    source_pages = ", ".join(
        [str(p) for p in retrieved_pages]
    )

    # PROMPT

    prompt = f"""
    You are an intelligent AI study assistant.

    Use the uploaded PDF as your primary knowledge source, but explain concepts naturally in your own words.

    Your job is to understand the user's intent and answer naturally like ChatGPT.

    You may:

    - Explain concepts
    - Summarize chapters
    - Generate exam questions
    - Create study plans
    - Create topic-wise roadmaps
    - Give revision notes
    - Compare concepts
    - Simplify difficult topics
    - Answer follow-up questions
    - Guide students

    Rules:

    1. Base answers mainly on the PDF content.
    2. If the user asks for explanation, teach it clearly.
    3. If the user asks for summary, summarize.
    4. If the user asks for exam questions, generate them from PDF topics.
    5. If the user asks for roadmap or study guidance, create it using PDF topics.
    6. Maintain conversation context.
    7. Use proper headings and bullet points.
    8. Give professional and student-friendly answers.
    9. Do not invent topics that are completely unrelated to the PDF.
   10. If the question is completely outside the PDF, DO NOT answer it.

    Simply respond:

    "This question does not appear to be related to the uploaded PDF."

    {context}

    USER QUESTION:

    {user_question}
    """

    # LLM RESPONSE

    conversation_history = []

    for msg in messages[-6:]:
        conversation_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    conversation_history.append({
        "role": "system",
        "content": """
    You are a professional AI assistant.

    Answer naturally like ChatGPT.

    Do not simply copy chunks.

    Understand the user's intent.

    Summarize information intelligently.

    Use complete sentences and proper explanations.

    Avoid repetitive wording.

    Give direct, professional, human-like answers.
    """
    })

    conversation_history.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        
        messages=conversation_history,

        stream=True
    )

    full_response = ""

    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        for chunk in response:

            if chunk.choices[0].delta.content:

                full_response += chunk.choices[0].delta.content

                message_placeholder.markdown(full_response)

    # CREATE CHAT TEXT
    chat_text = ""

    for msg in chat_data["messages"]:

        role = msg["role"].upper()
        content = msg["content"]

        chat_text += f"{role}: {content}\n\n"

    # SAVE ASSISTANT MESSAGE

    messages.append({
        "role": "assistant",
        "content": full_response,
        "page": source_pages
    })

    st.session_state.pop("voice_question", None)
    st.session_state.pop("pending_question", None)
    st.session_state.pop("last_voice_text", None)

    chat_data["messages"] = messages

    save_data = chat_data.copy()
    save_data["index"] = None

    chat_path = os.path.join(CHAT_DIR, current_chat)

    data_path = os.path.join(chat_path, "data.pkl")

    with open(data_path, "wb") as f:
        pickle.dump(save_data, f)