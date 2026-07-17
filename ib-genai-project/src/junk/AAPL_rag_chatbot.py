from __future__ import annotations
import re
from typing import Any
import chromadb
import requests
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

CHROMA_DB_PATH = (
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
    r"\ib-genai-project\data\vector_db\AAPL"
    r"\chroma_balance_sheet_db"
)

COLLECTION_NAME = "balance_sheet"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"

TOP_K = 5


# =========================================================
# INITIALIZATION
# =========================================================

def load_embedding_model() -> SentenceTransformer:
    """Load the same embedding model used when building ChromaDB."""

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


def load_chroma_collection():
    """Open the existing persistent Chroma database."""

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    available_collections = [
        item.name
        for item in client.list_collections()
    ]

    if COLLECTION_NAME not in available_collections:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' was not found.\n"
            f"Available collections: {available_collections}"
        )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Connected to collection '{COLLECTION_NAME}' "
        f"with {collection.count()} documents."
    )

    return collection


# =========================================================
# QUERY UNDERSTANDING
# =========================================================

def extract_fiscal_year(question: str) -> int | None:
    """
    Extract a fiscal year such as 2025 from the question.

    Examples:
        What was total debt in 2025?
        Compare cash for 2024 and 2025.
    """

    match = re.search(
        r"\b(20\d{2})\b",
        question
    )

    if not match:
        return None

    return int(match.group(1))


def detect_financial_section(
    question: str
) -> str | None:
    """Identify an optional financial section from the question."""

    question_lower = question.lower()

    section_keywords = {
        "Current Assets": [
            "current asset",
            "cash",
            "inventory",
            "receivable",
            "marketable securities",
        ],
        "Non-Current Assets": [
            "non-current asset",
            "noncurrent asset",
            "property",
            "equipment",
            "goodwill",
            "intangible",
        ],
        "Current Liabilities": [
            "current liability",
            "accounts payable",
            "short-term liability",
        ],
        "Non-Current Liabilities": [
            "non-current liability",
            "noncurrent liability",
            "long-term liability",
        ],
        "Shareholders Equity": [
            "shareholder equity",
            "shareholders equity",
            "stockholder equity",
            "retained earnings",
            "equity",
        ],
        "Debt and Investment Summary": [
            "debt",
            "net debt",
            "investment",
            "short-term debt",
            "long-term debt",
        ],
    }

    for section, keywords in section_keywords.items():
        if any(
            keyword in question_lower
            for keyword in keywords
        ):
            return section

    return None


# =========================================================
# CHROMA RETRIEVAL
# =========================================================

def retrieve_chunks(
    collection,
    embedding_model: SentenceTransformer,
    question: str,
    top_k: int = TOP_K
) -> list[dict[str, Any]]:
    """Retrieve relevant balance-sheet chunks from ChromaDB."""

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32").tolist()

    fiscal_year = extract_fiscal_year(question)
    financial_section = detect_financial_section(question)

    filters: list[dict[str, Any]] = []

    if fiscal_year is not None:
        filters.append({
            "fiscal_year": fiscal_year
        })

    if financial_section is not None:
        filters.append({
            "financial_section": financial_section
        })

    where_filter = None

    if len(filters) == 1:
        where_filter = filters[0]

    elif len(filters) > 1:
        where_filter = {
            "$and": filters
        }

    query_arguments: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(
            top_k,
            collection.count()
        ),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter is not None:
        query_arguments["where"] = where_filter

    try:
        results = collection.query(
            **query_arguments
        )

    except Exception as error:
        # If the inferred metadata filters return no usable records,
        # retry semantic search without filters.
        print(
            "Filtered retrieval failed; "
            f"retrying without filters: {error}"
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                top_k,
                collection.count()
            ),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    retrieved_chunks: list[dict[str, Any]] = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for record_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances
    ):
        retrieved_chunks.append({
            "id": record_id,
            "document": document,
            "metadata": metadata or {},
            "distance": float(distance),
            "similarity_score": 1 - float(distance),
        })

    return retrieved_chunks


# =========================================================
# PROMPT CONSTRUCTION
# =========================================================

def build_context(
    retrieved_chunks: list[dict[str, Any]]
) -> str:
    """Convert retrieved Chroma records into LLM context."""

    context_parts = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):
        metadata = chunk["metadata"]

        context_part = f"""
SOURCE {number}
Chunk ID: {chunk["id"]}
Symbol: {metadata.get("symbol", "Unknown")}
Fiscal year: {metadata.get("fiscal_year", "Unknown")}
Financial section: {metadata.get("financial_section", "Unknown")}
Report date: {metadata.get("report_date", "Unknown")}
Similarity score: {chunk["similarity_score"]:.4f}

Financial data:
{chunk["document"]}
""".strip()

        context_parts.append(context_part)

    return "\n\n" + ("\n\n" + "-" * 80 + "\n\n").join(
        context_parts
    )


def build_prompt(
    question: str,
    context: str
) -> str:
    """Build a grounded financial-analysis prompt."""

    return f"""
You are a financial statement analysis assistant.

Answer the user's question using only the financial context supplied below.

Rules:
1. Do not invent values or facts.
2. If the answer is unavailable in the supplied context, clearly say so.
3. Preserve the units and currency shown in the source.
4. Mention the fiscal year for every financial value.
5. When performing calculations, show the formula and calculation.
6. Keep the answer clear and concise.
7. At the end, include a Sources section listing the relevant chunk IDs.
8. Do not treat general model knowledge as financial evidence.

USER QUESTION:
{question}

RETRIEVED FINANCIAL CONTEXT:
{context}

ANSWER:
""".strip()


# =========================================================
# OLLAMA LLM
# =========================================================

def check_ollama() -> None:
    """Verify that the local Ollama service is available."""

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise ConnectionError(
            "Could not connect to Ollama.\n"
            "Make sure Ollama is installed and running.\n"
            "Then run: ollama pull gemma3"
        ) from error


def generate_answer(prompt: str) -> str:
    """Send the grounded prompt to the local Ollama LLM."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Ollama request failed: {error}"
        ) from error

    response_data = response.json()

    answer = response_data.get(
        "response",
        ""
    ).strip()

    if not answer:
        return (
            "The LLM returned an empty response."
        )

    return answer


# =========================================================
# DISPLAY
# =========================================================

def display_retrieved_sources(
    retrieved_chunks: list[dict[str, Any]]
) -> None:
    """Display retrieved Chroma sources for debugging."""

    print("\nRetrieved sources:")

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):
        metadata = chunk["metadata"]

        print(
            f"{number}. {chunk['id']} | "
            f"year={metadata.get('fiscal_year')} | "
            f"section={metadata.get('financial_section')} | "
            f"score={chunk['similarity_score']:.4f}"
        )


# =========================================================
# CHAT LOOP
# =========================================================

def main() -> None:
    """Run a terminal-based financial RAG chatbot."""

    print("=" * 80)
    print("APPLE BALANCE-SHEET RAG CHATBOT")
    print("=" * 80)

    embedding_model = load_embedding_model()
    collection = load_chroma_collection()
    check_ollama()

    print(
        "\nAsk a financial question."
        "\nType 'exit' or 'quit' to stop."
    )

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Chatbot closed.")
            break

        try:
            retrieved_chunks = retrieve_chunks(
                collection=collection,
                embedding_model=embedding_model,
                question=question,
                top_k=TOP_K
            )

            if not retrieved_chunks:
                print(
                    "\nAssistant: No relevant financial "
                    "chunks were found."
                )
                continue

            display_retrieved_sources(
                retrieved_chunks
            )

            context = build_context(
                retrieved_chunks
            )

            prompt = build_prompt(
                question=question,
                context=context
            )

            answer = generate_answer(
                prompt
            )

            print("\nAssistant:")
            print(answer)

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()