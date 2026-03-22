"""
Task 4: Context-Aware RAG Chatbot – Streamlit Deployment
DevelopersHub Corporation – AI/ML Engineering Internship

Dependencies: faiss-cpu, sentence-transformers, wikipedia, groq, streamlit
"""

import os
import time
import wikipedia
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot | DevelopersHub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .source-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #64b5f6;
        border: 1px solid #1976d2;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px 3px;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    .retrieval-card {
        background: #0d1117;
        border-left: 3px solid #2196f3;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #8b949e;
    }

    .title-gradient {
        background: linear-gradient(135deg, #2196f3, #21cbf3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
STEP             = CHUNK_SIZE - CHUNK_OVERLAP
MEMORY_WINDOW    = 5

DEFAULT_TOPICS = [
    'Artificial intelligence',
    'Machine learning',
    'Deep learning',
    'Natural language processing',
    'Transformer (machine learning model)',
    'Retrieval-augmented generation',
    'Large language model',
    'Python (programming language)',
]

SYSTEM_TEMPLATE = """You are a knowledgeable and concise assistant.
Answer the user's question using ONLY the retrieved context provided below.
If the answer is not in the context, say so honestly.

Retrieved Context:
------------------
{context}
------------------"""

# ─────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def build_index(topics: tuple):
    """Fetch Wikipedia articles, chunk them, embed, and build FAISS index."""
    embed_model = load_embed_model()

    corpus, metadata = [], []
    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            text = page.content[:6000]
            for i in range(0, len(text), STEP):
                chunk = text[i : i + CHUNK_SIZE]
                if len(chunk) > 80:
                    corpus.append(chunk)
                    metadata.append({'title': page.title, 'url': page.url})
        except Exception:
            pass

    if not corpus:
        return None, [], []

    embeddings = embed_model.encode(
        corpus, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    ).astype('float32')

    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, corpus, metadata


# ─────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────
def retrieve(query: str, index, corpus: list, metadata: list, k: int = 4):
    embed_model = load_embed_model()
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, k)
    return [
        {
            'text'  : corpus[idx],
            'title' : metadata[idx]['title'],
            'url'   : metadata[idx]['url'],
            'score' : float(score),
        }
        for score, idx in zip(scores[0], indices[0])
    ]


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com",
    )

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Settings")
    top_k  = st.slider("Chunks retrieved per query (k)", 2, 8, 4)
    mem_k  = st.slider("Memory window (turns)", 1, 10, MEMORY_WINDOW)

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Topics")
    selected_topics = st.multiselect(
        "Wikipedia articles",
        options=DEFAULT_TOPICS,
        default=DEFAULT_TOPICS[:4],
    )

    if st.button("🔄 Rebuild Index", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    **Stack**
    ```
    sentence-transformers
    faiss-cpu
    groq SDK (LLaMA-3 8B)
    streamlit
    ```
    """)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# Main Page Header
# ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="title-gradient">🤖 Context-Aware RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown("""
<p style="color:#8b949e; font-size:0.9rem;">
Task 4 · DevelopersHub AI/ML Internship · FAISS + SentenceTransformers + Groq LLaMA-3
</p>
""", unsafe_allow_html=True)

# Metrics row
c1, c2, c3, c4 = st.columns(4)
for col, icon, label, val in [
    (c1, "🔍", "Vector Store", "FAISS"),
    (c2, "🧠", "LLM",         "LLaMA-3 8B"),
    (c3, "📐", "Embeddings",  "MiniLM-L6"),
    (c4, "💾", "Memory",      f"Last {mem_k} turns"),
]:
    col.markdown(f"""<div class="metric-card">
        <div style="font-size:1.4rem">{icon}</div>
        <div style="font-size:0.75rem;color:#8b949e">{label}</div>
        <div style="font-weight:600">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Build / Load Index
# ─────────────────────────────────────────────────────────────────
if not selected_topics:
    st.warning("Select at least one topic from the sidebar.")
    st.stop()

with st.spinner("📚 Building knowledge base..."):
    index, corpus, metadata = build_index(tuple(selected_topics))

if index is None:
    st.error("Failed to load any Wikipedia articles. Check your internet connection.")
    st.stop()

st.success(f"✅ {index.ntotal} passages indexed from {len(selected_topics)} articles.")

# ─────────────────────────────────────────────────────────────────
# API Key Check
# ─────────────────────────────────────────────────────────────────
if not groq_key:
    st.info("🔑 Enter your **Groq API key** in the sidebar to start chatting. Get one free at [console.groq.com](https://console.groq.com)")
    st.stop()

groq_client = Groq(api_key=groq_key)

# ─────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {role, content}

# ─────────────────────────────────────────────────────────────────
# Render existing messages
# ─────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            badges = " ".join(
                f'<span class="source-badge">📄 {s}</span>'
                for s in msg["sources"]
            )
            st.markdown(f"<div style='margin-top:6px'>{badges}</div>",
                        unsafe_allow_html=True)
        if msg.get("latency"):
            st.caption(f"⏱ {msg['latency']:.2f}s")

# ─────────────────────────────────────────────────────────────────
# Chat Input
# ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about AI, ML, transformers, Python..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving & generating..."):
            t0 = time.time()
            try:
                # 1. Retrieve
                docs    = retrieve(prompt, index, corpus, metadata, k=top_k)
                context = "\n\n".join(
                    f'[{d["title"]}]\n{d["text"]}' for d in docs
                )
                sources = list(dict.fromkeys(d["title"] for d in docs))

                # 2. Build messages
                window   = st.session_state.chat_history[-(mem_k * 2):]
                messages = (
                    [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
                    + window
                    + [{"role": "user", "content": prompt}]
                )

                # 3. Call Groq
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=512,
                )
                answer  = response.choices[0].message.content
                latency = time.time() - t0

                # 4. Update sliding-window history
                st.session_state.chat_history.append(
                    {"role": "user",      "content": prompt}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

                # 5. Display
                st.markdown(answer)

                if sources:
                    badges = " ".join(
                        f'<span class="source-badge">📄 {s}</span>'
                        for s in sources
                    )
                    st.markdown(
                        f"<div style='margin-top:8px'>{badges}</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander(f"🔍 Retrieved {len(docs)} chunks"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"""<div class="retrieval-card">
                            <strong>Chunk {i}</strong> · {doc['title']}
                            · score={doc['score']:.3f}<br><br>
                            {doc['text'][:300]}...
                        </div>""", unsafe_allow_html=True)

                st.caption(f"⏱ {latency:.2f}s · {len(docs)} chunks retrieved")

                # Store for re-render
                st.session_state.messages.append({
                    "role"   : "assistant",
                    "content": answer,
                    "sources": sources,
                    "latency": latency,
                })

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ─────────────────────────────────────────────────────────────────
# Memory Inspector
# ─────────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    with st.expander("🧠 Conversation Memory Inspector"):
        for m in st.session_state.chat_history:
            icon = "👤" if m["role"] == "user" else "🤖"
            st.markdown(f"**{icon} {m['role'].capitalize()}:** {m['content'][:200]}...")
