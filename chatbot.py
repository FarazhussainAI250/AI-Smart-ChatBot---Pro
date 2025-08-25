# app.py — Groq Chatbot (Multi-Chat, Memory, Auto Titles, PDF)
from dotenv import load_dotenv
import os, time
import streamlit as st

from langchain_groq import ChatGroq
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- Setup ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI  Smart Chatbot Pro", page_icon="🤖", layout="wide")
st.title("🤖 AI Smart Chatbot — Pro")

if not GROQ_API_KEY:
    st.error("❌ Missing GROQ_API_KEY. Add it to your .env or deployment secrets.")
    st.stop()

# ---------------- Session Init ----------------
if "chats" not in st.session_state:
    st.session_state.chats = {}  # {chat_id: {name, memory, mode, mem_type, window_k}}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 0

# ---------- Helpers ----------
def new_chat():
    st.session_state.chat_counter += 1
    chat_id = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[chat_id] = {
        "name": "Untitled Chat",
        "memory": None,
        "mode": "🧠 General Assistant",
        "mem_type": "Buffer",
        "window_k": 6,
        "model_name": "gemma2-9b-it",
        "temperature": 0.7,
        "max_tokens": 512,
    }
    st.session_state.active_chat = chat_id

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        # select another chat if available
        if st.session_state.chats:
            st.session_state.active_chat = list(st.session_state.chats.keys())[0]
        else:
            st.session_state.active_chat = None

def ensure_memory(chat_id):
    chat = st.session_state.chats[chat_id]
    if chat["memory"] is None:
        if chat["mem_type"] == "Buffer":
            mem = ConversationBufferMemory(return_messages=True)
        elif chat["mem_type"] == "Summary":
            summarizer_llm = ChatGroq(model_name=chat["model_name"], temperature=0)
            mem = ConversationSummaryMemory(llm=summarizer_llm, return_messages=True)
        elif chat["mem_type"] == "Window":
            mem = ConversationBufferWindowMemory(k=chat["window_k"], return_messages=True)
        else:
            mem = ConversationBufferMemory(return_messages=True)
        chat["memory"] = mem

def auto_generate_title(chat_id):
    """Set title from first 4 *user* messages (like ChatGPT)."""
    chat = st.session_state.chats[chat_id]
    msgs = chat["memory"].chat_memory.messages if chat["memory"] else []
    if chat["name"] != "Untitled Chat":
        return
    user_msgs = [m.content for m in msgs if getattr(m, "type", "ai") == "human"]
    if len(user_msgs) >= 4:
        text = " ".join(user_msgs[:4]).strip()
        short = (text[:40] + "...") if len(text) > 40 else text
        chat["name"] = short if short else "New Chat"

def system_prompt_for_mode(mode: str) -> str:
    if mode == "🎓 Teaching Assistant":
        return "You are a helpful, concise teaching assistant. Prefer short, clear explanations."
    if mode == "👨‍💻 Coding Helper":
        return "You are a precise coding assistant. Respond with minimal prose, correct code, and brief tips."
    if mode == "🌍 Translator":
        return "You are a professional translator. Translate clearly; if not specified, provide Roman Urdu and English."
    return "You are a friendly, efficient assistant. Be brief, accurate, and helpful."

# ---------------- Sidebar (ONLY chats) ----------------
with st.sidebar:
    st.subheader("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()

    # Show chat buttons
    for cid, chat in st.session_state.chats.items():
        if st.button(chat["name"], key=f"sel_{cid}", use_container_width=True):
            st.session_state.active_chat = cid

    st.markdown("---")
    if st.session_state.active_chat:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Delete Current", use_container_width=True):
                delete_chat(st.session_state.active_chat)
                st.rerun()
        with col_b:
            if st.button("🧹 Clear Messages", use_container_width=True):
                curr = st.session_state.chats[st.session_state.active_chat]
                if curr["memory"]:
                    curr["memory"].clear()
                st.rerun()

# ---------------- Main Area (Q/A only) ----------------
if not st.session_state.active_chat:
    st.info("Click **➕ New Chat** to start.")
    st.stop()
    

chat_id = st.session_state.active_chat
chat = st.session_state.chats[chat_id]
# ---- Ensure a chat is active
if not st.session_state.active_chat:
    st.info("Click **➕ New Chat** to start.")
    st.stop()

chat_id = st.session_state.active_chat
chat = st.session_state.chats[chat_id]

# === Chat Settings: move to SIDEBAR but AFTER chat_id/chat are defined ===
with st.sidebar:
    st.header("⚙️ Chat Settings")

    # Rename chat
    new_name = st.text_input("Rename chat", value=chat["name"], key=f"rename_{chat_id}")

    # Mode selector full width
    chat["mode"] = st.selectbox(
        "Mode",
        ["🎓 Teaching Assistant", "👨‍💻 Coding Helper", "🌍 Translator", "🧠 General Assistant"],
        index=["🎓 Teaching Assistant", "👨‍💻 Coding Helper", "🌍 Translator", "🧠 General Assistant"].index(chat["mode"]),
        key=f"mode_{chat_id}"
    )

    # Model selector full width
    chat["model_name"] = st.selectbox(
        "Model",
        ["gemma2-9b-it", "llama-3.3-70b-versatile"],
        index=["gemma2-9b-it", "llama-3.3-70b-versatile"].index(chat["model_name"]),
        key=f"model_{chat_id}"
    )

    # Memory selector full width
    chat["mem_type"] = st.selectbox(
        "Memory",
        ["Buffer", "Summary", "Window"],
        index=["Buffer", "Summary", "Window"].index(chat["mem_type"]),
        key=f"memtype_{chat_id}"
    )





# ---- Ensure memory exists for this chat
ensure_memory(chat_id)
memory = chat["memory"]

# ---- Build LLM + Chain
llm = ChatGroq(
    model_name=chat["model_name"],
    temperature=chat["temperature"],
    max_tokens=chat["max_tokens"],
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_for_mode(chat["mode"])),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)

# ---- Render history (center only)
for m in memory.chat_memory.messages:
    role = getattr(m, "type", "ai")
    if role == "human":
        with st.chat_message("user"):
            st.markdown(m.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(m.content)

# ---- Chat input (unique key per chat)
user_input = st.chat_input("Type your message…", key=f"in_{chat_id}")
if user_input:
    st.chat_message("user").markdown(user_input)
    full_response = conversation.predict(input=user_input)

    # Typing animation
    placeholder = st.chat_message("assistant").empty()
    accum = ""
    for ch_ in full_response:
        accum += ch_
        placeholder.markdown(accum)
        time.sleep(0.012)

    # Set auto title after 4 user messages
    auto_generate_title(chat_id)
    st.rerun()

# ---- PDF download
if memory.chat_memory.messages:
    pdf = BytesIO()
    c = canvas.Canvas(pdf, pagesize=letter)
    W, H = letter
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, H - 40, f"Chat History — {chat['name']}")
    c.setFont("Helvetica", 11)
    y = H - 70
    for m in memory.chat_memory.messages:
        role = getattr(m, "type", "ai").upper()
        text = f"{role}: {m.content}"
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 15
            if y < 40:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = H - 50
    c.save()
    pdf.seek(0)
    st.download_button(
        "📥 Download Chat as PDF",
        data=pdf,
        file_name=f"{chat['name']}.pdf",
        mime="application/pdf",
    )

st.caption("Made with LangChain + Groq — Multi-Chat, Memory, Auto Titles.")
# ----------- Custom CSS with Header & Footer ------------
st.markdown("""
<style>
/* Background Image */
.stApp {
    background-image: url("");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}



/* Top-right Header */
#top-header {
    position: fixed;
    top: 80px;
    right: 20px;
    background-color: rgba(0,0,0,0.5);
    padding: 8px 16px;
    border-radius: 8px;
    color: white;
    font-size: 18px;
    font-weight: bold;
    z-index: 100;
}

/* Bottom-left Footer */
#bottom-footer {
    position: fixed;
    bottom: 10px;
    left: 300px;
    background-color: rgba(0,0,0,0.5);
    padding: 6px 14px;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    z-index: 100;
}
</style>

<div id="top-header">Respected Sir Shahzaib & Sir Ali Hamza</div>
<div id="bottom-footer">Developed by Faraz Hussain</div>
""", unsafe_allow_html=True)
