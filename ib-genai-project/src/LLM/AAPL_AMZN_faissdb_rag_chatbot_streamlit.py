"""
AAPL_AMZN_faissdb_rag_chatbot_streamlit.py

Streamlit UI for the Investment Banking RAG Chatbot - works for BOTH
AAPL and AMZN from a single file. Pick the company from the sidebar;
everything else (FAISS index, metadata, chat history, exports) follows
that selection automatically.

Backed by:
- A FAISS vector index per company (single index covering Balance Sheet,
  Cash Flow, Income Statement, Growth, Key Metrics, Ratios, and
  Cross-Statement Relationships chunks, distinguished via metadata)
- OpenAI embeddings (text-embedding-3-small) - the embedding/index
  loading helpers are inlined below rather than imported from
  AAPL_build_faiss_index.py / AMZN_build_faiss_index.py, precisely so
  this one file doesn't have to pick a single company's build script
  at import time.
- LangChain (PromptTemplate + ConversationBufferMemory) with ChatOpenAI as the LLM

Enhancements in this version:
- A company switcher (AAPL / AMZN) in the sidebar. Switching companies
  swaps the FAISS index/metadata and clears chat history + memory, so
  the two companies' conversations never mix.
- Token-by-token streaming answers (st.write_stream) instead of one big
  block appearing after the full response finishes
- Best-effort token usage / estimated cost tracking per answer and per
  session (via langchain's get_openai_callback, with a no-op fallback)
- A confidence badge (High/Medium/Low) derived from the top retrieval
  similarity score, so users can see at a glance how well-grounded an
  answer is
- Thumbs up/down feedback buttons on every assistant answer
- Redesigned source cards (colored statement-type chips + highlighted
  query terms) instead of a plain markdown table
- Sidebar reorganized into tabs (Retrieval / Model / Memory / Session)
  so it doesn't read as one long scroll
- Fixed a pre-existing bug where "Regenerate last answer" duplicated the
  user's question as a second chat bubble
- Toast notifications for housekeeping actions (clear chat)
- A "view system prompt" expander for transparency/debugging

Run with the following command:
    streamlit run AAPL_AMZN_faissdb_rag_chatbot_streamlit.py
"""

import os
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import __main__

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

# LangChain conversational memory, with a safe fallback if the installed
# LangChain version has moved/removed the classic memory module.
try:
    from langchain.memory import ConversationBufferWindowMemory

    LANGCHAIN_MEMORY_AVAILABLE = True
except ImportError:
    LANGCHAIN_MEMORY_AVAILABLE = False

    class ConversationBufferWindowMemory:
        """Minimal drop-in fallback mirroring the LangChain memory interface
        used in this app, so the UI still works if langchain.memory isn't
        importable (e.g. on LangChain >=1.0, where it was removed)."""

        def __init__(self, k=5, memory_key="chat_history", **kwargs):
            self.k = k
            self.memory_key = memory_key
            self._turns = []

        def save_context(self, inputs, outputs):
            self._turns.append((inputs.get("input", ""), outputs.get("output", "")))
            self._turns = self._turns[-self.k:]

        def load_memory_variables(self, _inputs):
            history = "\n".join(f"Human: {q}\nAI: {a}" for q, a in self._turns)
            return {self.memory_key: history}

        def clear(self):
            self._turns = []


# Token usage / cost tracking, with a no-op fallback if the installed
# LangChain version doesn't ship the community callback helper.
try:
    from langchain_community.callbacks import get_openai_callback

    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False

    from contextlib import contextmanager

    @contextmanager
    def get_openai_callback():
        class _NullCallback:
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            total_cost = 0.0

        yield _NullCallback()


@dataclass
class ChunkRecord:
    chunk_id: str = ""
    text: str = ""
    statement_type_source: str = ""
    financial_section: str = ""


# Make pickle able to resolve __main__.ChunkRecord. The metadata .pkl
# files were originally written by AAPL_build_faiss_index.py /
# AMZN_build_faiss_index.py while each ran as "__main__", so unpickling
# them from any other script (this one included) requires a ChunkRecord
# class registered under __main__ with the same name. Pickle restores
# the full original __dict__ regardless of which fields are declared
# above, so this minimal stub is enough even though the original
# ChunkRecord has a few extra fields (source_file, source_row, etc.).
__main__.ChunkRecord = ChunkRecord

load_dotenv()

# ==========================================================
# Embedding / FAISS helpers
# ==========================================================
# Inlined from AAPL_build_faiss_index.py / AMZN_build_faiss_index.py
# (identical in both) rather than imported, so this single file isn't
# tied to either company's build script.

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_RETRY_BACKOFF_SECONDS = 2.0


def load_embedding_client() -> OpenAI:
    """Create the OpenAI client. Reads OPENAI_API_KEY from the environment."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before running this script, e.g.:\n"
            "  setx OPENAI_API_KEY \"sk-...\"   (Windows, new terminal needed after)\n"
            "  export OPENAI_API_KEY=\"sk-...\" (macOS/Linux)"
        )
    return OpenAI()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows so FAISS inner product behaves as cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on a degenerate zero vector
    return embeddings / norms


def embed_text_batch(
    client: OpenAI,
    texts: list[str],
    model_name: str,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[list[float]]:
    """Call the OpenAI embeddings endpoint for one batch, with basic retry."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(model=model_name, input=texts)
            return [item.embedding for item in response.data]
        except Exception as error:  # noqa: BLE001 - broad by design, retried below
            last_error = error
            wait_seconds = retry_backoff_seconds * attempt
            print(
                f"  Embedding batch failed (attempt {attempt}/{max_retries}): "
                f"{error}. Retrying in {wait_seconds:.0f}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Embedding batch failed after {max_retries} attempts."
    ) from last_error


def load_faiss_index(index_file: Path) -> faiss.Index:
    return faiss.read_index(str(index_file))


def load_metadata(metadata_file: Path) -> list[ChunkRecord]:
    with metadata_file.open("rb") as file:
        return pickle.load(file)


def search_index(
    query: str,
    client: OpenAI,
    index: faiss.Index,
    records: list[ChunkRecord],
    top_k: int = 5,
    statement_type_filter: list[str] | str | None = None,
) -> list[tuple[float, ChunkRecord]]:
    """Embed a query and return the top_k most similar chunk records.

    statement_type_filter can be one (or a list) of the CHUNK_SOURCES
    keys (e.g. 'balance_sheet') to restrict results to specific
    statement types. Filtering is done by over-fetching then narrowing,
    since FAISS IndexFlatIP has no native metadata filter.
    """
    raw_embedding = embed_text_batch(
        client=client,
        texts=[query],
        model_name=EMBEDDING_MODEL_NAME,
        max_retries=EMBEDDING_MAX_RETRIES,
        retry_backoff_seconds=EMBEDDING_RETRY_BACKOFF_SECONDS,
    )
    query_embedding = normalize_embeddings(np.array(raw_embedding, dtype="float32"))

    if isinstance(statement_type_filter, str):
        statement_type_filter = [statement_type_filter]

    fetch_k = top_k * 5 if statement_type_filter else top_k
    scores, ids = index.search(query_embedding, fetch_k)

    results: list[tuple[float, ChunkRecord]] = []
    for score, record_id in zip(scores[0], ids[0]):
        if record_id == -1:
            continue

        record = records[record_id]

        if statement_type_filter and record.statement_type_source not in statement_type_filter:
            continue

        results.append((float(score), record))

        if len(results) >= top_k:
            break

    return results


# ==========================================================
# Configuration
# ==========================================================

# One entry per supported company. FAISS_INDEX_FILE / METADATA_FILE /
# METADATA_JSON_PREVIEW_FILE follow the exact naming pattern written by
# AAPL_build_faiss_index.py / AMZN_build_faiss_index.py:
#   data/faiss_db/<COMPANY>/<company_lower>_financials.index
#   data/faiss_db/<COMPANY>/<company_lower>_financials_metadata.pkl
BASE_FAISS_DB_DIR = Path(
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\faiss_db"
)

COMPANIES = ["AAPL", "AMZN"]
COMPANY_DEFAULT = "AAPL"

COMPANY_LABELS = {
    "AAPL": "Apple Inc. (AAPL)",
    "AMZN": "Amazon.com, Inc. (AMZN)",
}


def get_company_paths(company: str) -> dict[str, Path]:
    folder = BASE_FAISS_DB_DIR / company
    prefix = company.lower()
    return {
        "index": folder / f"{prefix}_financials.index",
        "metadata": folder / f"{prefix}_financials_metadata.pkl",
        "preview": folder / f"{prefix}_financials_metadata_preview.json",
    }


MODEL_OPTIONS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]
MODEL_DEFAULT = "gpt-4.1-mini"

TOP_K_DEFAULT = 5
MAX_RETRIES = 2
MEMORY_WINDOW_DEFAULT = 5

# Keys here match the raw `statement_type_source` values written into
# each ChunkRecord by *_build_faiss_index.py (i.e. the CHUNK_SOURCES
# dict keys: "balance_sheet", "cash_flow", etc.) - NOT a Title Case
# label. Using the raw key format is what lets these actually match
# against record data; a prior version of this app kept the display
# label ("Balance Sheet") as the dict key here, which meant icons and
# colors never matched anything and silently fell back to defaults.
STATEMENT_TYPES = [
    "balance_sheet",
    "cash_flow",
    "income_statement",
    "growth",
    "key_metrics",
    "ratios",
    "relationships",
]

STATEMENT_LABELS = {
    "balance_sheet": "Balance Sheet",
    "cash_flow": "Cash Flow",
    "income_statement": "Income Statement",
    "growth": "Growth",
    "key_metrics": "Key Metrics",
    "ratios": "Ratios",
    "relationships": "Cross-Statement Relationships",
}

STATEMENT_ICONS = {
    "balance_sheet": "\U0001F3E6",
    "cash_flow": "\U0001F4B8",
    "income_statement": "\U0001F4C8",
    "growth": "\U0001F331",
    "key_metrics": "\U0001F9EE",
    "ratios": "\u2696\uFE0F",
    "relationships": "\U0001F517",
}

STATEMENT_COLORS = {
    "balance_sheet": "#2f6fed",
    "cash_flow": "#1f9e6d",
    "income_statement": "#e08a2b",
    "growth": "#8e44ad",
    "key_metrics": "#0e9aa7",
    "ratios": "#c0392b",
    "relationships": "#5d6d7e",
}

# Similarity thresholds used for the confidence badge. These assume a
# 0-1 cosine-similarity-style score; retune them if your embedding
# client reports similarity on a different scale.
CONFIDENCE_THRESHOLDS = {"high": 0.60, "medium": 0.35}

USER_AVATAR = "\U0001F9D1\u200D\U0001F4BC"
ASSISTANT_AVATAR = "\U0001F4CA"

# ==========================================================
# Page Setup
# ==========================================================

st.set_page_config(
    page_title="IB Research Assistant",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Centralize all session-state defaults in one place so it's obvious
# what persists across reruns.
_SESSION_DEFAULTS = {
    "dark_mode": False,
    "company": COMPANY_DEFAULT,
    "messages": [],
    "query_count": 0,
    "total_cost": 0.0,
    "total_tokens": 0,
}
for _key, _default in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


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
    return load_embedding_client()


@st.cache_resource(show_spinner=False)
def load_index_and_records(company: str):
    """Load the FAISS index + metadata records for one company, reporting
    any load errors. Cached per-company so switching companies doesn't
    re-embed anything, and re-selecting a previously loaded company is
    instant on subsequent reruns."""
    paths = get_company_paths(company)
    try:
        index = load_faiss_index(paths["index"])
        records = load_metadata(paths["metadata"])
        return index, records, None
    except Exception as error:
        return None, [], str(error)


def get_openai_status():
    return bool(os.environ.get("OPENAI_API_KEY"))


# ==========================================================
# Retrieval + Generation
# ==========================================================

def build_context_and_sources(question, client, index, records, top_k, statement_type_filter):
    """Run FAISS search and shape results into a context string + source list."""
    results = search_index(
        query=question,
        client=client,
        index=index,
        records=records,
        top_k=top_k,
        statement_type_filter=statement_type_filter,
    )

    parts = []
    sources = []
    for score, record in results:
        statement = getattr(record, "statement_type_source", "") or "—"
        section = getattr(record, "financial_section", "") or "—"
        chunk_id = getattr(record, "chunk_id", "") or "—"
        text = getattr(record, "text", "") or ""

        parts.append(
            f"""Statement Type: {statement}

Financial Section: {section}

Chunk ID: {chunk_id}

Content:
{text}"""
        )

        sources.append(
            {
                "statement": statement,
                "section": section,
                "chunk_id": chunk_id,
                "similarity": round(float(score), 3) if score is not None else None,
                "preview": text[:220].replace("\n", " ") + ("..." if len(text) > 220 else ""),
            }
        )

    # Highest similarity first so the model (and user) sees best evidence up top
    sources.sort(key=lambda s: (s["similarity"] if s["similarity"] is not None else -1), reverse=True)
    context = "\n\n" + ("-" * 80 + "\n\n").join(parts) if parts else ""
    return context, sources


def confidence_badge(sources):
    """Best-effort High/Medium/Low badge from the top retrieval similarity
    score, so a user can see at a glance how well-grounded an answer is."""
    if not sources:
        return None
    top_score = sources[0].get("similarity")
    if top_score is None:
        return None
    if top_score >= CONFIDENCE_THRESHOLDS["high"]:
        return "🟢 High confidence"
    if top_score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "🟡 Medium confidence"
    return "🔴 Low confidence — consider rephrasing"


def highlight_terms(text, query, limit=8):
    """Bold the query's keywords inside a source preview so the user can
    quickly see why a chunk was retrieved."""
    terms = sorted({t for t in re.findall(r"[A-Za-z]{3,}", query or "")}, key=len, reverse=True)[:limit]
    for term in terms:
        text = re.sub(rf"(?i)\b({re.escape(term)})\b", r"**\1**", text)
    return text


PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are an expert Investment Banking financial analyst.

Answer ONLY using the supplied context. Use the conversation history solely
to resolve follow-up questions (e.g. "what about the year before?"), never
as a source of financial facts.

If the answer is unavailable, reply:

"I could not find this information in the financial statements."

Conversation:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
)


def build_llm_chain(model_name, temperature, max_tokens, streaming=True):
    """Construct the PromptTemplate | ChatOpenAI | StrOutputParser chain.

    stream_usage=True asks OpenAI to include token counts in the final
    streamed chunk so get_openai_callback can report cost even though
    we're streaming. Older langchain-openai versions may not accept the
    kwarg, so we fall back gracefully.
    """
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            stream_usage=True,
        )
    except TypeError:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
    return PROMPT | llm | StrOutputParser()


def stream_llm(question, context, chat_history, model_name, temperature, max_tokens):
    """Yield the answer token-by-token (for st.write_stream) and stash
    best-effort token usage / cost in session state once exhausted."""
    chain = build_llm_chain(model_name, temperature, max_tokens, streaming=True)
    with get_openai_callback() as cb:
        for chunk in chain.stream(
            {"chat_history": chat_history, "context": context, "question": question}
        ):
            yield chunk

        st.session_state["_last_usage"] = {
            "total_tokens": getattr(cb, "total_tokens", 0),
            "prompt_tokens": getattr(cb, "prompt_tokens", 0),
            "completion_tokens": getattr(cb, "completion_tokens", 0),
            "total_cost": getattr(cb, "total_cost", 0.0),
        }


def stream_with_retry(question, context, chat_history, model_name, temperature, max_tokens, retries=MAX_RETRIES):
    """Wrap stream_llm with a retry that only fires if the *first* token
    fails to arrive, so a transient error never duplicates partially
    streamed text on screen."""
    last_error = None
    for attempt in range(retries + 1):
        generator = stream_llm(question, context, chat_history, model_name, temperature, max_tokens)
        try:
            first_chunk = next(generator)
        except StopIteration:
            return iter(())
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenAI request failed after {retries + 1} attempt(s): {last_error}") from last_error

        def _chained(first=first_chunk, rest=generator):
            yield first
            yield from rest

        return _chained()

    return iter(())


def pop_last_memory_turn(memory):
    """Remove the most recently saved turn from the buffer, so regenerating
    an answer doesn't leave the old (discarded) answer chained in context."""
    chat_memory = getattr(memory, "chat_memory", None)
    if chat_memory is not None and getattr(chat_memory, "messages", None):
        # LangChain stores turns as alternating Human/AI messages.
        for _ in range(2):
            if chat_memory.messages:
                chat_memory.messages.pop()
    elif hasattr(memory, "_turns") and memory._turns:
        memory._turns.pop()


def format_transcript(messages, company):
    lines = [f"# {company} Research Assistant — Conversation Export", f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
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
        st.markdown("## \U0001F4CA IB Research Assistant")
        st.caption("Retrieval-augmented analysis over public filings")
    with header_col2:
        if st.button("\U0001F319" if not st.session_state.dark_mode else "\u2600\uFE0F"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.divider()

    st.markdown("### Company")
    selected_company = st.selectbox(
        "Which company's filings to search",
        options=COMPANIES,
        index=COMPANIES.index(st.session_state.company),
        format_func=lambda c: COMPANY_LABELS.get(c, c),
        label_visibility="collapsed",
    )
    if selected_company != st.session_state.company:
        # Switching companies invalidates the current conversation - the
        # chat history and memory buffer would otherwise mix Apple and
        # Amazon context together, which the LLM has no way to untangle.
        st.session_state.company = selected_company
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.total_cost = 0.0
        st.session_state.total_tokens = 0
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.toast(f"Switched to {COMPANY_LABELS.get(selected_company, selected_company)} — chat cleared", icon="\U0001F504")
        st.rerun()

    company = st.session_state.company

    st.divider()

    st.markdown("### Data Source")
    embedder = load_embedder()
    company_paths = get_company_paths(company)
    index, records, index_error = load_index_and_records(company)

    if index is not None:
        st.success(f"\U0001F4C8 **FAISS Index**  \n`{company_paths['index']}` · {len(records)} chunks", icon="✅")
    else:
        st.error(f"\U0001F4C8 **FAISS Index**  \n{index_error or 'Unavailable'}", icon="⚠️")

    st.caption(f"Total indexed chunks: **{len(records)}**")
    if not COST_TRACKING_AVAILABLE:
        st.caption("⚠️ `langchain_community.callbacks` not importable — token/cost tracking disabled.")


    st.divider()

    tab_retrieval, tab_model, tab_memory, tab_session = st.tabs(
        ["🔎 Retrieval", "🤖 Model", "🧠 Memory", "📁 Session"]
    )

    with tab_retrieval:
        st.caption("Statement types to search")
        available_types = sorted({getattr(r, "statement_type_source", "") for r in records if getattr(r, "statement_type_source", "")}) or STATEMENT_TYPES

        selected_types = []
        for type_key in available_types:
            icon = STATEMENT_ICONS.get(type_key, "\U0001F4C4")
            display_label = STATEMENT_LABELS.get(type_key, type_key)
            checked = st.checkbox(f"{icon} {display_label}", value=True, key=f"chk_{type_key}")
            if checked:
                selected_types.append(type_key)

        # If every available type is selected, no filter is needed (search all).
        statement_type_filter = None if set(selected_types) == set(available_types) else (selected_types or None)

        top_k = st.slider("Chunks to retrieve (Top K)", min_value=1, max_value=15, value=TOP_K_DEFAULT)

        with st.expander("Advanced generation settings"):
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                     help="Lower = more precise and deterministic. Recommended for financial data.")
            max_tokens = st.slider("Max response length (tokens)", min_value=128, max_value=2048, value=512, step=128)

    with tab_model:
        st.markdown("#### Model")
        openai_key_present = get_openai_status()
        if openai_key_present:
            st.success("OpenAI API key detected", icon="🟢")
        else:
            st.error("OPENAI_API_KEY not found in environment", icon="🔴")
            st.caption("Set `OPENAI_API_KEY` in your `.env` file, then refresh this page.")

        model_labeled_options = MODEL_OPTIONS + ["\u270F\uFE0F Custom model name..."]
        default_index = MODEL_OPTIONS.index(MODEL_DEFAULT) if MODEL_DEFAULT in MODEL_OPTIONS else 0
        selected_label = st.selectbox("Model", options=model_labeled_options, index=default_index)

        if selected_label == "\u270F\uFE0F Custom model name...":
            model_name = st.text_input("Enter OpenAI model name", value=MODEL_DEFAULT)
        else:
            model_name = selected_label

    with tab_memory:
        st.markdown("#### Conversation Memory")
        if not LANGCHAIN_MEMORY_AVAILABLE:
            st.caption("⚠️ `langchain.memory` not importable — using a built-in fallback buffer with the same behavior.")

        use_memory = st.checkbox(
            "Remember conversation context",
            value=True,
            help="Feeds prior Q&A turns to the model so follow-up questions "
                 "(e.g. 'what about the year before?') resolve correctly.",
        )

        memory_window = st.slider(
            "Memory window (turns to remember)",
            min_value=1,
            max_value=15,
            value=MEMORY_WINDOW_DEFAULT,
            disabled=not use_memory,
            help="Number of most recent question/answer pairs kept in the buffer. "
                 "Older turns are chained out automatically (sliding window).",
        )

        # (Re)initialize the LangChain buffer if window size changes or on first run.
        if "memory" not in st.session_state or st.session_state.get("memory_window") != memory_window:
            st.session_state.memory = ConversationBufferWindowMemory(k=memory_window, memory_key="chat_history")
            st.session_state.memory_window = memory_window

        turns_in_buffer = len(getattr(st.session_state.memory, "_turns", getattr(st.session_state.memory, "buffer", [])) or [])
        st.caption(f"🧠 Buffer holds up to **{memory_window}** turn(s) · chaining {'enabled' if use_memory else 'disabled'}")

    with tab_session:
        st.markdown("#### Session")
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='metric-card'><b>{st.session_state.query_count}</b><br><span class='muted-text'>Questions</span></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='metric-card'><b>{len(st.session_state.messages)}</b><br><span class='muted-text'>Messages</span></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='metric-card'><b>${st.session_state.total_cost:.3f}</b><br><span class='muted-text'>Est. cost</span></div>", unsafe_allow_html=True)

        st.write("")

        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.query_count = 0
                st.session_state.total_cost = 0.0
                st.session_state.total_tokens = 0
                st.session_state.memory.clear()
                st.toast("Chat cleared", icon="🗑️")
                st.rerun()
        with col_export:
            transcript = format_transcript(st.session_state.messages, company) if st.session_state.messages else "No conversation yet."
            st.download_button(
                "📥 Export",
                data=transcript,
                file_name=f"{company.lower()}_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=not st.session_state.messages,
            )

        with st.expander("🧾 View system prompt template"):
            st.code(PROMPT.template, language="markdown")

# ==========================================================
# Main Chat Area
# ==========================================================

st.title(f"{company} Financial Statement Assistant")
st.caption(f"Ask questions about {company}'s Balance Sheet, Cash Flow, Income Statement, Growth, Key Metrics, Ratios, and Cross-Statement Relationships. Answers are grounded strictly in retrieved filings data.")

# Suggested starter questions
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    example_cols = st.columns(3)
    examples = [
        f"What was {company}'s revenue and net income for FY2025?",
        "How did R&D expense change from FY2024 to FY2025?",
        f"What was {company}'s diluted EPS in the latest quarter?",
    ]
    for col, example in zip(example_cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example


def render_sources(sources, key_prefix="", query=""):
    with st.expander(f"📎 {len(sources)} source{'s' if len(sources) != 1 else ''} used"):
        for i, src in enumerate(sources):
            color = STATEMENT_COLORS.get(src["statement"], "#7a7f8c")
            icon = STATEMENT_ICONS.get(src["statement"], "\U0001F4C4")
            display_label = STATEMENT_LABELS.get(src["statement"], src["statement"])
            st.markdown(
                f"<span class='source-chip' style='border-color:{color};color:{color};'>"
                f"{icon} {display_label}</span>"
                f"<span class='source-chip'>{src['section']}</span>"
                f"<span class='source-chip'>Chunk: {src['chunk_id']}</span>"
                f"<span class='source-chip'>Similarity: {src['similarity']}</span>",
                unsafe_allow_html=True,
            )
            preview = highlight_terms(src["preview"], query) if query else src["preview"]
            st.caption(preview)
            if i < len(sources) - 1:
                st.divider()


def render_feedback(idx):
    """Thumbs up/down on an assistant message, recorded in session state."""
    feedback = st.session_state.messages[idx].get("feedback")
    cols = st.columns([1, 1, 10])
    if cols[0].button("👍", key=f"fb_up_{idx}", disabled=feedback is not None):
        st.session_state.messages[idx]["feedback"] = "up"
        st.toast("Thanks for the feedback!", icon="👍")
        st.rerun()
    if cols[1].button("👎", key=f"fb_down_{idx}", disabled=feedback is not None):
        st.session_state.messages[idx]["feedback"] = "down"
        st.toast("Got it — thanks for flagging this.", icon="👎")
        st.rerun()
    if feedback:
        cols[2].caption("Feedback recorded ✅" if feedback == "up" else "Feedback recorded — flagged ⚠️")


# Render chat history
for idx, message in enumerate(st.session_state.messages):
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        footer_bits = []
        if message.get("latency"):
            footer_bits.append(f"⏱️ {message['latency']:.1f}s")
        usage = message.get("usage")
        if usage and usage.get("total_tokens"):
            footer_bits.append(
                f"${usage['total_cost']:.4f}" if usage.get("total_cost") else f"{usage['total_tokens']} tok"
            )
        badge = message.get("confidence")
        if badge:
            footer_bits.append(badge)
        if footer_bits:
            st.caption(" · ".join(footer_bits))

        if message.get("sources"):
            render_sources(message["sources"], key_prefix=f"hist_{idx}", query=message.get("query", ""))

        if message["role"] == "assistant":
            render_feedback(idx)

# Chat input
question = st.chat_input(f"Ask about {company}'s financials...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


def run_query(question_text, is_regenerate=False):
    # On regenerate, the user bubble is already in history — don't duplicate it.
    if not is_regenerate:
        st.session_state.messages.append({"role": "user", "content": question_text, "sources": None})
        st.session_state.query_count += 1

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question_text)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        start_time = time.time()
        usage = None
        badge = None

        if index is None:
            answer = "The FAISS index is unavailable. Please check the sidebar for details."
            sources = []
            st.warning(answer)
        elif not selected_types:
            answer = "Please select at least one statement type in the sidebar to search."
            sources = []
            st.warning(answer)
        else:
            try:
                with st.spinner("Retrieving relevant filings data..."):
                    context, sources = build_context_and_sources(
                        question_text, embedder, index, records, top_k, statement_type_filter
                    )

                chat_history = ""
                if use_memory:
                    chat_history = st.session_state.memory.load_memory_variables({}).get("chat_history", "")

                answer = st.write_stream(
                    stream_with_retry(question_text, context, chat_history, model_name, temperature, max_tokens)
                )
                usage = st.session_state.pop("_last_usage", None)
                badge = confidence_badge(sources)

                if use_memory:
                    st.session_state.memory.save_context({"input": question_text}, {"output": answer})

            except Exception as error:
                answer = f"⚠️ Something went wrong: {error}"
                sources = []
                st.error(answer)

        latency = time.time() - start_time

        footer_bits = [f"⏱️ {latency:.1f}s", model_name, f"{len(sources)} chunks retrieved"]
        if usage and usage.get("total_tokens"):
            footer_bits.append(f"${usage['total_cost']:.4f}" if usage.get("total_cost") else f"{usage['total_tokens']} tok")
            st.session_state.total_cost += usage.get("total_cost", 0.0)
            st.session_state.total_tokens += usage.get("total_tokens", 0)
        if badge:
            footer_bits.append(badge)
        st.caption(" · ".join(footer_bits))

        if sources:
            render_sources(sources, key_prefix="live", query=question_text)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency": latency,
                "usage": usage,
                "confidence": badge,
                "query": question_text,
                "feedback": None,
            }
        )
        render_feedback(len(st.session_state.messages) - 1)


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
            if use_memory:
                pop_last_memory_turn(st.session_state.memory)
            run_query(last_user_msg, is_regenerate=True)
