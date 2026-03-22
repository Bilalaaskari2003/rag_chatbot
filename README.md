# Task 4: Context-Aware Chatbot Using RAG

> **DevelopersHub Corporation – AI/ML Engineering Advanced Internship**

---

## Objective

Build a conversational chatbot that:
- Remembers multi-turn conversation history using a **sliding-window context memory**
- Retrieves answers from a **vectorized document store** using RAG
- Uses **FAISS** for fast semantic search and **Groq (LLaMA-3.1)** as the LLM
- Is deployed as an interactive web app via **Streamlit**

---

## Architecture

```
User Query
    ↓
SentenceTransformer Embedding (all-MiniLM-L6-v2)
    ↓
FAISS IndexFlatIP (cosine similarity) → Top-K Relevant Chunks
    ↓
Groq API — LLaMA-3.1 8B Instant
    ├── System prompt + retrieved context
    └── Sliding window chat history (last 5 turns)
    ↓
Response + Source Attribution
```

---

## Tech Stack

| Component     | Tool                                      |
|---------------|-------------------------------------------|
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2`  |
| Vector Store  | FAISS (IndexFlatIP — cosine similarity)   |
| LLM           | Groq — LLaMA-3.1 8B Instant (free API)   |
| Memory        | Sliding window (last 5 turns)             |
| Corpus        | Wikipedia (AI/ML topics, 8 articles)      |
| Deployment    | Streamlit + ngrok (Google Colab)          |

> **No LangChain required.** Uses only `faiss-cpu`, `sentence-transformers`, and the `groq` SDK — no version conflict issues.

---

## Setup & Run

### Option A — Google Colab (Recommended)

#### Step 1 — Open the notebook
Upload `task4_rag_chatbot.ipynb` to [colab.research.google.com](https://colab.research.google.com)

#### Step 2 — Get a free Groq API key
1. Sign up at [console.groq.com](https://console.groq.com)
2. Go to **API Keys** → create a new key
3. In Colab, click the **🔑 Secrets icon** in the left sidebar
4. Add secret: Name = `GROQ_API_KEY`, Value = your key
5. Toggle **Notebook access ON**

#### Step 3 — Run all notebook cells in order
```
Cell 1  →  Install dependencies
Cell 2  →  Import libraries
Cell 3  →  Fetch Wikipedia corpus & chunk
Cell 4  →  Build FAISS index
Cell 5  →  Define retrieve() function
Cell 6  →  Initialize Groq client & chat() function
Cell 7+ →  Multi-turn conversation tests & evaluation
```

#### Step 4 — Run the Streamlit app

Install and launch:
```python
!pip install -q streamlit pyngrok
```

Upload `app.py` via the 📁 Files panel, then run:
```python
from pyngrok import ngrok
import subprocess, time

ngrok.kill()
ngrok.set_auth_token('YOUR_NGROK_TOKEN')  # free at dashboard.ngrok.com

subprocess.Popen([
    'streamlit', 'run', 'app.py',
    '--server.port=8501',
    '--server.headless=true'
])
time.sleep(4)

public_url = ngrok.connect(8501)
print(f'App live at: {public_url}')
```

Open the printed URL in your browser, enter your Groq API key in the sidebar, and start chatting.

> ⚠️ **Important — ngrok tunnel conflict (`ERR_NGROK_334`):**
> If you see `The endpoint is already online`, you must delete the existing tunnel first:
> 1. Go to **[dashboard.ngrok.com/endpoints](https://dashboard.ngrok.com/endpoints)**
> 2. Find the active endpoint → click **3 dots (⋮) → Delete**
> 3. Wait 5 seconds, then rerun the cell above
>
> Alternatively, go to **Runtime → Disconnect and delete runtime** in Colab and start fresh — this clears stuck tunnels automatically.

---

### Option B — Local Machine

#### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
pip install streamlit
```

#### Step 2 — Run the notebook
```bash
jupyter notebook task4_rag_chatbot.ipynb
```
Paste your Groq API key into the `GROQ_API_KEY` variable in Cell 6.

#### Step 3 — Run the Streamlit app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Getting a Ngrok Token (for Colab)

1. Go to [dashboard.ngrok.com](https://dashboard.ngrok.com)
2. Sign up for free (Google login works)
3. Click **"Your Authtoken"** in the left sidebar
4. Copy the token and paste it into `ngrok.set_auth_token('...')`

---

## Key Results & Observations

### Retrieval Performance
- FAISS indexes **200+ document chunks** from 8 Wikipedia articles
- Average retrieval latency: **~2–5ms** per query
- Top-4 most relevant chunks retrieved per query with source attribution

### Context Memory
- Sliding window of the last **5 conversation turns** passed with every LLM call
- Enables pronoun resolution across turns — e.g. *"How does **it** compare to RNNs?"* correctly resolves to transformers from the previous turn

### RAG vs Pure LLM

| Aspect            | Pure LLM        | RAG                      |
|-------------------|-----------------|--------------------------|
| Factual grounding | May hallucinate | Grounded in corpus       |
| Source attribution| None            | Document titles shown    |
| Out-of-corpus     | Guesses         | Admits it doesn't know   |
| Retrieval overhead| None            | +2–5ms                   |

---

## Project Structure

```
task4_rag_chatbot/
├── task4_rag_chatbot.ipynb   # Full notebook with analysis & visualisations
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Dependencies

```
faiss-cpu
sentence-transformers
wikipedia
groq
numpy
scikit-learn
matplotlib
streamlit       # for app.py only
pyngrok         # for Colab deployment only
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Skills Demonstrated

- Conversational AI with sliding-window context memory
- Document embedding and semantic vector search (FAISS)
- Retrieval-Augmented Generation (RAG)
- LLM integration via Groq API (LLaMA-3.1 8B Instant)
- Streamlit deployment via ngrok on Google Colab
