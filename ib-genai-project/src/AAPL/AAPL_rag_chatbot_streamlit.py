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

import time
from datetime import datetime

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
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"

TOP_K_DEFAULT = 3
MAX_RETRIES = 2

DB_CONFIG = {
    "Balance Sheet": BALANCE_DB,
    "Cash Flow": CASHFLOW_DB,
    "Income Statement": INCOMESTATEMENT_DB,
}

STATEMENT_ICONS = {
    "Balance Sheet": "\U0001F3E6",
    "Cash Flow": "\U0001F4B8",
    "Income Statement": "\U0001F4C8",
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

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def inject_theme(dark: bool) -> None:
    if dark:
        app_bg, sidebar_bg, sidebar_border = "#0e1117", "#161a23", "#2d3648"
        chip_bg, chip_border, chip_text = "#1c2333", "#2d3648", "#9fb4d1"
        text_color, muted = "#e6e9ef", "#9aa4b2"
        card_bg, card_border = "#161a23", "#2a3040"
    else:
        app_bg, sidebar_bg, sidebar_border = "#f7f8fa", "#ffffff", "#e3e6eb"
        chip_bg, chip_border, chip_text = "#eef2fa", "#cdd7e8", "#2f4a73"
        text_color, muted = "#1a1d23", "#5b6472"
        card_bg, card_border = "#ffffff", "#e3e6eb"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {app_bg}; color: {text_color}; }}
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            border-right: 1px solid {sidebar_border};
        }}
        .source-chip {{
            display: inline-block;
            padding: 2px 10px;
            margin: 2px 4px 2px 0;
            border-radius: 999px;
            background-color: {chip_bg};
            border: 1px solid {chip_border};
            color: {chip_text};
            font-size: 0.75rem;
        }}
        .stChatMessage {{ border-radius: 12px; }}
        .metric-card {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 10px;
            padding: 10px 14px;
            text-align: center;
        }}
        .muted-text {{ color: {muted}; font-size: 0.8rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme(st.session_state.dark_mode)

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
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
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
            meta = meta or {}
            documents.append(doc)
            sources.append(
                {
                    "statement": label,
                    "similarity": round(1 - dist, 3) if dist is not None else None,
                    "fiscal_year": meta.get("fiscalYear", meta.get("fiscal_year", "—")),
                    "period": meta.get("period", "—"),
                    "section": meta.get("group", meta.get("section", "—")),
                    "report_date": meta.get("date", "—"),
                    "preview": doc[:220].replace("\n", " ") + ("..." if len(doc) > 220 else ""),
                }
            )

    # Highest similarity first so the model (and user) sees best evidence up top
    sources.sort(key=lambda s: (s["similarity"] if s["similarity"] is not None else -1), reverse=True)
    context = "\n\n".join(documents)
    return context, sources


def ask_ollama(question, context, model_name, temperature, max_tokens, retries=MAX_RETRIES):
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

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                OLLAMA_GENERATE_URL,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Ollama request failed after {retries + 1} attempt(s): {last_error}")


def format_transcript(messages):
    lines = [f"# AAPL Research Assistant — Conversation Export", f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    for msg in messages:
        speaker = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{speaker}: {msg['content']}")
        lines.append("")
    return "\n".join(lines)


# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("## \U0001F4CA AAPL IB Research Assistant")
        st.caption("Retrieval-augmented analysis over AAPL's financial statements")
    with header_col2:
        if st.button("\U0001F319" if not st.session_state.dark_mode else "\u2600\uFE0F"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.divider()

    st.markdown("### Data Sources")
    embedder = load_embedder()
    loaded_dbs, db_errors = load_collections()
    total_chunks = sum(info["count"] for info in loaded_dbs.values())

    for label in DB_CONFIG:
        icon = STATEMENT_ICONS.get(label, "")
        if label in loaded_dbs:
            info = loaded_dbs[label]
            st.success(f"{icon} **{label}**  \n`{info['collection_name']}` · {info['count']} chunks", icon="✅")
        else:
            st.error(f"{icon} **{label}**  \n{db_errors.get(label, 'Unavailable')}", icon="⚠️")

    st.caption(f"Total indexed chunks across all statements: **{total_chunks}**")

    st.divider()

    st.markdown("### Retrieval Settings")
    st.caption("Statements to search")
    selected_labels = []
    for label in DB_CONFIG:
        icon = STATEMENT_ICONS.get(label, "")
        is_available = label in loaded_dbs
        checked = st.checkbox(
            f"{icon} {label}",
            value=is_available,
            disabled=not is_available,
            key=f"chk_{label}",
            help=None if is_available else db_errors.get(label, "Unavailable"),
        )
        if checked and is_available:
            selected_labels.append(label)

    top_k = st.slider("Chunks per statement (Top K)", min_value=1, max_value=10, value=TOP_K_DEFAULT)

    with st.expander("Advanced generation settings"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                 help="Lower = more precise and deterministic. Recommended for financial data.")
        max_tokens = st.slider("Max response length (tokens)", min_value=128, max_value=2048, value=512, step=128)

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
        st.caption("Run `ollama serve` in a terminal, then refresh this page.")
        model_name = OLLAMA_MODEL_DEFAULT

    st.divider()

    st.markdown("### Session")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0

    m1, m2 = st.columns(2)
    m1.markdown(f"<div class='metric-card'><b>{st.session_state.query_count}</b><br><span class='muted-text'>Questions asked</span></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><b>{len(st.session_state.messages)}</b><br><span class='muted-text'>Messages</span></div>", unsafe_allow_html=True)

    st.write("")

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
    with col_export:
        transcript = format_transcript(st.session_state.messages) if st.session_state.messages else "No conversation yet."
        st.download_button(
            "📥 Export",
            data=transcript,
            file_name=f"aapl_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )

# ==========================================================
# Main Chat Area
# ==========================================================

st.title("AAPL Financial Statement Assistant")
st.caption("Ask questions about AAPL's Balance Sheet, Cash Flow, and Income Statement. Answers are grounded strictly in retrieved filings data.")

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


def render_sources(sources, key_prefix=""):
    with st.expander(f"📎 {len(sources)} sources used"):
        st.markdown(
            "| Statement | Fiscal Year | Period | Section | Similarity |\n"
            "|---|---|---|---|---|"
        )
        for src in sources:
            st.markdown(
                f"| {STATEMENT_ICONS.get(src['statement'], '')} {src['statement']} "
                f"| {src['fiscal_year']} | {src['period']} | {src['section']} | {src['similarity']} |"
            )
        st.write("")
        for i, src in enumerate(sources):
            st.markdown(
                f"<span class='source-chip'>{src['statement']}</span>"
                f"<span class='source-chip'>Similarity: {src['similarity']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(src["preview"])
            if i < len(sources) - 1:
                st.divider()


# Render chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("latency"):
            st.caption(f"⏱️ {message['latency']:.1f}s")
        if message.get("sources"):
            render_sources(message["sources"], key_prefix=f"hist_{idx}")

# Chat input
question = st.chat_input("Ask about AAPL's financials...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


def run_query(question_text):
    st.session_state.messages.append({"role": "user", "content": question_text, "sources": None})
    st.session_state.query_count += 1

    with st.chat_message("user"):
        st.markdown(question_text)

    with st.chat_message("assistant"):
        start_time = time.time()

        if not selected_labels:
            answer = "Please select at least one financial statement in the sidebar to search."
            sources = []
            st.warning(answer)
        else:
            try:
                with st.spinner("Retrieving relevant filings data..."):
                    context, sources = build_context_and_sources(
                        question_text, embedder, loaded_dbs, selected_labels, top_k
                    )

                with st.spinner(f"Asking {model_name}..."):
                    answer = ask_ollama(question_text, context, model_name, temperature, max_tokens)

                st.markdown(answer)

            except Exception as error:
                answer = f"⚠️ Something went wrong: {error}"
                sources = []
                st.error(answer)

        latency = time.time() - start_time
        st.caption(f"⏱️ {latency:.1f}s · {model_name} · {len(sources)} chunks retrieved")

        if sources:
            render_sources(sources, key_prefix="live")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "latency": latency}
    )


if question:
    run_query(question)

# Regenerate last answer
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if st.button("🔄 Regenerate last answer"):
        last_user_msg = next(
            (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None
        )
        if last_user_msg:
            st.session_state.messages = st.session_state.messages[:-1]
            run_query(last_user_msg)
