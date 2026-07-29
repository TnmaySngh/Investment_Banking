"""
AAPL_faissdb_rag_chatbot_streamlit.py

Streamlit UI for the AAPL Investment Banking RAG Chatbot.

Backed by:
- A FAISS vector index (single index covering Balance Sheet, Cash Flow,
  and Income Statement chunks, distinguished via metadata)
- Existing project embedding client (AAPL_build_faiss_index)
- LangChain (PromptTemplate + ConversationBufferMemory) with ChatOpenAI as the LLM

Run with the following command:
    streamlit run AAPL_faissdb_rag_chatbot__streamlit.py
"""

import os
import time
from datetime import datetime
from dataclasses import dataclass
import __main__

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


@dataclass
class ChunkRecord:
    chunk_id: str = ""
    text: str = ""
    statement_type_source: str = ""
    financial_section: str = ""


# Make pickle able to resolve __main__.ChunkRecord
__main__.ChunkRecord = ChunkRecord

# Existing project imports
from AAPL_build_faiss_index import (
    load_embedding_client,
    load_faiss_index,
    load_metadata,
    search_index,
    FAISS_INDEX_FILE,
    METADATA_FILE,
)

load_dotenv()

# ==========================================================
# Configuration
# ==========================================================

MODEL_OPTIONS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]
MODEL_DEFAULT = "gpt-4.1-mini"

TOP_K_DEFAULT = 5
MAX_RETRIES = 2
MEMORY_WINDOW_DEFAULT = 5

STATEMENT_TYPES = ["Balance Sheet", "Cash Flow", "Income Statement"]

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
    return load_embedding_client()


@st.cache_resource(show_spinner=False)
def load_index_and_records():
    """Load the FAISS index + metadata records, reporting any load errors."""
    try:
        index = load_faiss_index(FAISS_INDEX_FILE)
        records = load_metadata(METADATA_FILE)
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


def ask_llm(question, context, chat_history, model_name, temperature, max_tokens, retries=MAX_RETRIES):
    llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    chain = PROMPT | llm | StrOutputParser()

    last_error = None
    for attempt in range(retries + 1):
        try:
            return chain.invoke(
                {
                    "chat_history": chat_history,
                    "context": context,
                    "question": question,
                }
            )
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"OpenAI request failed after {retries + 1} attempt(s): {last_error}")


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

    st.markdown("### Data Source")
    embedder = load_embedder()
    index, records, index_error = load_index_and_records()

    if index is not None:
        st.success(f"\U0001F4C8 **FAISS Index**  \n`{FAISS_INDEX_FILE}` · {len(records)} chunks", icon="✅")
    else:
        st.error(f"\U0001F4C8 **FAISS Index**  \n{index_error or 'Unavailable'}", icon="⚠️")

    st.caption(f"Total indexed chunks: **{len(records)}**")

    st.divider()

    st.markdown("### Retrieval Settings")
    st.caption("Statement types to search")
    available_types = sorted({getattr(r, "statement_type_source", "") for r in records if getattr(r, "statement_type_source", "")}) or STATEMENT_TYPES

    selected_types = []
    for label in available_types:
        icon = STATEMENT_ICONS.get(label, "\U0001F4C4")
        checked = st.checkbox(f"{icon} {label}", value=True, key=f"chk_{label}")
        if checked:
            selected_types.append(label)

    # If every available type is selected, no filter is needed (search all).
    statement_type_filter = None if set(selected_types) == set(available_types) else (selected_types or None)

    top_k = st.slider("Chunks to retrieve (Top K)", min_value=1, max_value=15, value=TOP_K_DEFAULT)

    with st.expander("Advanced generation settings"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                 help="Lower = more precise and deterministic. Recommended for financial data.")
        max_tokens = st.slider("Max response length (tokens)", min_value=128, max_value=2048, value=512, step=128)

    st.divider()

    st.markdown("### Model")
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

    st.divider()

    st.markdown("### Conversation Memory")
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
            st.session_state.memory.clear()
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

st.title("Financial Statement Assistant")
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
            "| Statement | Section | Chunk ID | Similarity |\n"
            "|---|---|---|---|"
        )
        for src in sources:
            st.markdown(
                f"| {STATEMENT_ICONS.get(src['statement'], '')} {src['statement']} "
                f"| {src['section']} | {src['chunk_id']} | {src['similarity']} |"
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

                with st.spinner(f"Asking {model_name}..."):
                    answer = ask_llm(
                        question_text, context, chat_history, model_name, temperature, max_tokens,
                    )

                st.markdown(answer)

                if use_memory:
                    st.session_state.memory.save_context({"input": question_text}, {"output": answer})

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
            if use_memory:
                pop_last_memory_turn(st.session_state.memory)
            run_query(last_user_msg)
