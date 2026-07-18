"""
AAPL_rag_chatbot_streamlit.py

Streamlit UI for the AAPL Investment Banking RAG Chatbot.

Backed by:
- Three ChromaDB databases (Balance Sheet, Cash Flow, Income Statement)
- Sentence Transformers embeddings
- Ollama (Gemma3) as the LLM

Run with:
    streamlit run AAPL_rag_chatbot_streamlit.py
"""

import chromadb
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ==========================================================
# Configuration
# ==========================================================

BALANCE_DB = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\vector_db\AAPL\chroma_balance_sheet_db"
CASHFLOW_DB = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\vector_db\AAPL\chroma_cash_flow_db"
INCOMESTATEMENT_DB = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\vector_db\AAPL\chroma_income_statement_db"

OLLAMA_MODEL_DEFAULT = "gemma3"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K_DEFAULT = 3

DB_CONFIG = {
    "Balance Sheet": BALANCE_DB,
    "Cash Flow": CASHFLOW_DB,
    "Income Statement": INCOMESTATEMENT_DB,
}

# ==========================================================
# Page Setup
# ==========================================================

st.set_page_config(
    page_title="AAPL IB Research Assistant",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f7f8fa; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e3e6eb; }
    .source-chip {
        display: inline-block;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        border-radius: 999px;
        background-color: #eef2fa;
        border: 1px solid #cdd7e8;
        color: #2f4a73;
        font-size: 0.75rem;
    }
    .stChatMessage { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# Cached Resources
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_collections():
    """Connect to all three Chroma DBs and load their first collection each."""
    loaded = {}
    errors = {}

    for label, path in DB_CONFIG.items():
        try:
            client = chromadb.PersistentClient(path=path)
            collections = client.list_collections()

            if len(collections) == 0:
                errors[label] = "No collection found in this database."
                continue

            collection = client.get_collection(collections[0].name)
            loaded[label] = {
                "collection": collection,
                "collection_name": collections[0].name,
                "count": collection.count(),
            }

        except Exception as error:
            errors[label] = str(error)

    return loaded, errors


def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return True, models
    except Exception:
        return False, []


# ==========================================================
# Retrieval + Generation
# ==========================================================

def retrieve(collection, embedder, question, k):
    embedding = embedder.encode(question, normalize_embeddings=True).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return result


def build_context_and_sources(question, embedder, loaded_dbs, selected_labels, top_k):
    documents = []
    sources = []

    for label in selected_labels:
        if label not in loaded_dbs:
            continue

        result = retrieve(loaded_dbs[label]["collection"], embedder, question, top_k)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            documents.append(doc)
            sources.append(
                {
                    "statement": label,
                    "similarity": round(1 - dist, 3) if dist is not None else None,
                    "metadata": meta,
                    "preview": doc[:220].replace("\n", " ") + ("..." if len(doc) > 220 else ""),
                }
            )

    context = "\n\n".join(documents)
    return context, sources


def ask_ollama(question, context, model_name):
    prompt = f"""
You are an Investment Banking Financial Analyst.

Answer ONLY using the supplied context.

If the answer is unavailable, reply:

"I could not find this information in the financial statements."

-------------------------
CONTEXT
-------------------------

{context}

-------------------------
QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""

    response = requests.post(
        OLLAMA_URL,
        json={"model": model_name, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:
    st.markdown("## \U0001F4CA AAPL IB Research Assistant")
    st.caption("Retrieval-augmented analysis over AAPL's financial statements")

    st.divider()

    st.markdown("### Data Sources")
    embedder = load_embedder()
    loaded_dbs, db_errors = load_collections()

    for label in DB_CONFIG:
        if label in loaded_dbs:
            info = loaded_dbs[label]
            st.success(f"**{label}**  \n`{info['collection_name']}` · {info['count']} chunks", icon="✅")
        else:
            st.error(f"**{label}**  \n{db_errors.get(label, 'Unavailable')}", icon="⚠️")

    st.divider()

    st.markdown("### Retrieval Settings")
    selected_labels = st.multiselect(
        "Statements to search",
        options=list(DB_CONFIG.keys()),
        default=list(loaded_dbs.keys()),
    )
    top_k = st.slider("Chunks per statement (Top K)", min_value=1, max_value=10, value=TOP_K_DEFAULT)

    st.divider()

    st.markdown("### Model")
    ollama_online, available_models = check_ollama_status()
    if ollama_online:
        st.success("Ollama is running", icon="🟢")
        model_options = available_models if available_models else [OLLAMA_MODEL_DEFAULT]
        default_index = model_options.index(OLLAMA_MODEL_DEFAULT) if OLLAMA_MODEL_DEFAULT in model_options else 0
        model_name = st.selectbox("Model", options=model_options, index=default_index)
    else:
        st.error("Ollama not reachable at localhost:11434", icon="🔴")
        model_name = OLLAMA_MODEL_DEFAULT

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================================
# Main Chat Area
# ==========================================================

st.title("AAPL Financial Statement Assistant")
st.caption("Ask questions about AAPL's Balance Sheet, Cash Flow, and Income Statement. Answers are grounded strictly in retrieved filings data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested starter questions
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    example_cols = st.columns(3)
    examples = [
        "What was AAPL's revenue and net income for FY2025?",
        "How did R&D expense change from FY2024 to FY2025?",
        "What was AAPL's diluted EPS in the latest quarter?",
    ]
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} sources used"):
                for src in message["sources"]:
                    meta = src["metadata"] or {}
                    st.markdown(
                        f"<span class='source-chip'>{src['statement']}</span>"
                        f"<span class='source-chip'>Similarity: {src['similarity']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(src["preview"])
                    st.divider()

# Chat input
question = st.chat_input("Ask about AAPL's financials...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not selected_labels:
            answer = "Please select at least one financial statement in the sidebar to search."
            sources = []
            st.markdown(answer)
        else:
            with st.spinner("Retrieving relevant filings data..."):
                context, sources = build_context_and_sources(
                    question, embedder, loaded_dbs, selected_labels, top_k
                )

            with st.spinner(f"Asking {model_name}..."):
                try:
                    answer = ask_ollama(question, context, model_name)
                except Exception as error:
                    answer = f"Error contacting Ollama: {error}"

            st.markdown(answer)

            if sources:
                with st.expander(f"📎 {len(sources)} sources used"):
                    for src in sources:
                        st.markdown(
                            f"<span class='source-chip'>{src['statement']}</span>"
                            f"<span class='source-chip'>Similarity: {src['similarity']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(src["preview"])
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
