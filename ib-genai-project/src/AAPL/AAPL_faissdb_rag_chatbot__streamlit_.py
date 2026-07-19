from __future__ import annotations

import streamlit as st

from AAPL_build_faiss_index import (
    ChunkRecord,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    load_embedding_client,
    load_faiss_index,
    load_metadata,
    search_index,
)
from AAPL_faiss_chatbot import (
    SYSTEM_PROMPT,
    build_context_block,
    generate_answer,
)

# =========================================================
# CONFIGURATION
# =========================================================

PAGE_TITLE = "AAPL Financial Statement Chatbot"
DEFAULT_TOP_K = 5

FILTER_OPTIONS = {
    "All statement types": None,
    "Balance sheet": "balance_sheet",
    "Cash flow": "cash_flow",
    "Income statement": "income_statement",
}


# =========================================================
# CACHED RESOURCES
# =========================================================

@st.cache_resource(show_spinner="Loading FAISS index and metadata...")
def get_index_and_records() -> tuple:
    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index file not found: {FAISS_INDEX_FILE}\n"
            "Run AAPL_build_faiss_index.py first."
        )
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            "Run AAPL_build_faiss_index.py first."
        )

    index = load_faiss_index(FAISS_INDEX_FILE)
    records = load_metadata(METADATA_FILE)
    return index, records


@st.cache_resource(show_spinner=False)
def get_client():
    return load_embedding_client()


# =========================================================
# SIDEBAR
# =========================================================

def render_sidebar() -> tuple[str | None, int]:
    st.sidebar.header("Settings")

    filter_label = st.sidebar.selectbox(
        "Restrict search to",
        options=list(FILTER_OPTIONS.keys()),
        index=0,
    )
    statement_type_filter = FILTER_OPTIONS[filter_label]

    top_k = st.sidebar.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=15,
        value=DEFAULT_TOP_K,
    )

    st.sidebar.divider()

    if st.sidebar.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    with st.sidebar.expander("System prompt"):
        st.text(SYSTEM_PROMPT)

    return statement_type_filter, top_k


# =========================================================
# CHAT DISPLAY HELPERS
# =========================================================

def render_sources(results: list[tuple[float, ChunkRecord]]) -> None:
    with st.expander(f"Sources ({len(results)})"):
        for rank, (score, record) in enumerate(results, start=1):
            st.markdown(
                f"**[{rank}]** score={score:.3f} | "
                f"`{record.statement_type_source}` | "
                f"{record.financial_section} | "
                f"`{record.chunk_id}`"
            )
            st.code(record.text, language="text")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("results"):
            render_sources(message["results"])


# =========================================================
# MAIN APP
# =========================================================

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="\U0001F4CA", layout="centered")
    st.title(PAGE_TITLE)
    st.caption(
        "Ask questions about AAPL's balance sheet, cash flow, or income "
        "statement filings. Answers are generated only from retrieved chunks."
    )

    try:
        index, records = get_index_and_records()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    try:
        client = get_client()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    statement_type_filter, top_k = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        render_message(message)

    user_question = st.chat_input("Ask about AAPL's financials...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks..."):
                try:
                    results = search_index(
                        query=user_question,
                        client=client,
                        index=index,
                        records=records,
                        top_k=top_k,
                        statement_type_filter=statement_type_filter,
                    )
                except Exception as error:  # noqa: BLE001 - surface any retrieval error to the user
                    error_text = f"Retrieval failed: {error}"
                    st.error(error_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_text, "results": []}
                    )
                    st.stop()

            if not results:
                answer = "No matching chunks found for that query."
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "results": []}
                )
            else:
                context_block = build_context_block(results)

                with st.spinner("Generating answer..."):
                    try:
                        answer = generate_answer(client, user_question, context_block)
                    except Exception as error:  # noqa: BLE001 - surface any chat API error to the user
                        answer = f"Answer generation failed: {error}"

                st.markdown(answer)
                render_sources(results)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "results": results}
                )


if __name__ == "__main__":
    main()
