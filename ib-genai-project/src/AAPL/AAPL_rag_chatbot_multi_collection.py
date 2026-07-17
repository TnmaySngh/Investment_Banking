"""
AAPL_rag_chatbot.py

RAG Chatbot using:
- Two ChromaDB databases
- Sentence Transformers
- Ollama (Gemma3)
"""

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ==========================================================
# Configuration
# ==========================================================

BALANCE_DB = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\vector_db\AAPL\chroma_balance_sheet_db"

CASHFLOW_DB = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\vector_db\AAPL\chroma_cash_flow_db"

OLLAMA_MODEL = "gemma3"
OLLAMA_URL = "http://localhost:11434/api/generate"

TOP_K = 3

# ==========================================================
# Load Embedding Model
# ==========================================================

print("Loading embedding model...")

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================================
# Connect to Chroma Databases
# ==========================================================

print("Connecting to Chroma databases...")

balance_client = chromadb.PersistentClient(path=BALANCE_DB)
cashflow_client = chromadb.PersistentClient(path=CASHFLOW_DB)

# ==========================================================
# Show Available Collections
# ==========================================================

balance_collections = balance_client.list_collections()
cashflow_collections = cashflow_client.list_collections()

print("\nBalance Sheet Collections:")
for c in balance_collections:
    print(" -", c.name)

print("\nCash Flow Collections:")
for c in cashflow_collections:
    print(" -", c.name)

if len(balance_collections) == 0:
    raise Exception("No collection found in Balance Sheet database.")

if len(cashflow_collections) == 0:
    raise Exception("No collection found in Cash Flow database.")

# Automatically use first collection
balance = balance_client.get_collection(balance_collections[0].name)
cashflow = cashflow_client.get_collection(cashflow_collections[0].name)

print("\nCollections Loaded Successfully!")

# ==========================================================
# Retrieval Function
# ==========================================================

def retrieve(collection, question, k=TOP_K):

    embedding = embedder.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return result

# ==========================================================
# Ask Function
# ==========================================================

def ask(question):

    balance_result = retrieve(balance, question)
    cashflow_result = retrieve(cashflow, question)

    documents = []

    if balance_result["documents"]:
        documents.extend(balance_result["documents"][0])

    if cashflow_result["documents"]:
        documents.extend(cashflow_result["documents"][0])

    context = "\n\n".join(documents)

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
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["response"]

# ==========================================================
# Chat Loop
# ==========================================================

def main():

    print("\n========================================")
    print(" Investment Banking RAG Chatbot ")
    print("========================================")

    while True:

        question = input("\nYou : ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            answer = ask(question)

            print("\nAssistant:\n")
            print(answer)

        except Exception as e:
            print("\nError:", e)

if __name__ == "__main__":
    main()