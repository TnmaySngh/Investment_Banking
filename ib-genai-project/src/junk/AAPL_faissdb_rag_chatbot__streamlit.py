
"""
AAPL_faissdb_rag_chatbot_streamlit.py

Template showing how to integrate:
- Streamlit
- LangChain PromptTemplate
- ConversationBufferMemory
- ChatOpenAI
- Existing FAISS retrieval functions

NOTE:
Replace the imports below with your existing project modules if needed.
"""

import os
from dataclasses import dataclass
import __main__

@dataclass
class ChunkRecord:
    chunk_id: str = ""
    text: str = ""
    statement_type_source: str = ""
    financial_section: str = ""

# Make pickle able to resolve __main__.ChunkRecord
__main__.ChunkRecord = ChunkRecord

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser

# Existing project imports
from AAPL_build_faiss_index import (
    load_embedding_client,
    load_faiss_index,
    load_metadata,
    search_index,
    FAISS_INDEX_FILE,
    METADATA_FILE,
)

def build_context_block(results):
    """Convert retrieved FAISS chunks into a context string."""
    parts=[]
    for score, record in results:
        parts.append(f"""Statement Type: {getattr(record,'statement_type_source','')}

Financial Section: {getattr(record,'financial_section','')}

Chunk ID: {getattr(record,'chunk_id','')}

Content:
{getattr(record,'text','')}""")
    return "\n\n" + ("-"*80 + "\n\n").join(parts)


load_dotenv()

st.set_page_config(page_title="AAPL Financial Chatbot")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False,
    )

prompt = PromptTemplate(
    input_variables=["chat_history","context","question"],
    template="""
You are an expert financial analyst.

Conversation:
{chat_history}

Context:
{context}

Question:
{question}

Answer only from the supplied context.
If unavailable, say so.

Answer:
"""
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

chain = prompt | llm | StrOutputParser()

index = load_faiss_index(FAISS_INDEX_FILE)
records = load_metadata(METADATA_FILE)
client = load_embedding_client()

st.title("Financial Chatbot")

question = st.chat_input("Ask a question...")

if question:
    results = search_index(
        query=question,
        client=client,
        index=index,
        records=records,
        top_k=5,
        statement_type_filter=None,
    )

    context = build_context_block(results)

    history = st.session_state.memory.load_memory_variables({})

    answer = chain.invoke(
        {
            "chat_history": history["chat_history"],
            "context": context,
            "question": question,
        }
    )

    st.session_state.memory.save_context(
        {"input": question},
        {"output": answer},
    )

    st.chat_message("user").write(question)
    st.chat_message("assistant").write(answer)

    with st.expander("Retrieved Sources"):
        for score, record in results:
            st.write(score)
            st.code(record.text)
