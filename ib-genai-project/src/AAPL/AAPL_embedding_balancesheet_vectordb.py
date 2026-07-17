import os
import re
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_TEXT_FILE = (
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
    r"\ib-genai-project\data\chunks\AAPL\balance_sheet"
    r"\balance_sheet_all_chunks.txt"
)

CHROMA_DATABASE_FOLDER = (
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
    r"\ib-genai-project\data\vector_db\AAPL"
    r"\chroma_balance_sheet_db"
)

COLLECTION_NAME = "balance_sheet"

# Lightweight local embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# READ AND SPLIT TEXT FILE
# =========================================================

def read_text_file(file_path: str) -> str:
    """Read the financial chunk text file."""

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Input text file was not found:\n{file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_financial_chunks(text: str) -> list[str]:
    """
    Split the file using CHUNK NUMBER markers.

    Expected structure:

        CHUNK NUMBER: 1
        ...
        CHUNK NUMBER: 2
        ...
    """

    pieces = re.split(
        r"(?=CHUNK NUMBER:\s*\d+)",
        text
    )

    chunks = [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]

    return chunks


# =========================================================
# METADATA EXTRACTION
# =========================================================

def extract_value(
    chunk: str,
    label: str,
    default: str = ""
) -> str:
    """Extract a single metadata value from a chunk."""

    pattern = rf"^{re.escape(label)}:\s*(.+)$"

    match = re.search(
        pattern,
        chunk,
        flags=re.MULTILINE
    )

    if not match:
        return default

    return match.group(1).strip()


def normalize_year(year_value: str) -> int:
    """
    Convert values such as '2,025' into 2025.
    """

    cleaned_year = re.sub(
        r"[^\d]",
        "",
        year_value
    )

    if not cleaned_year:
        return 0

    return int(cleaned_year)


def normalize_integer(value: str) -> int:
    """
    Convert values such as '320,193' into 320193.
    """

    cleaned_value = re.sub(
        r"[^\d-]",
        "",
        value
    )

    if not cleaned_value:
        return 0

    return int(cleaned_value)


def extract_chunk_metadata(
    chunk: str,
    fallback_number: int
) -> dict[str, Any]:
    """Extract Chroma-compatible metadata from one financial chunk."""

    chunk_number_text = extract_value(
        chunk,
        "CHUNK NUMBER",
        str(fallback_number)
    )

    chunk_id = extract_value(
        chunk,
        "CHUNK ID",
        f"financial_chunk_{fallback_number}"
    )

    fiscal_year_text = extract_value(
        chunk,
        "Fiscal Year",
        "0"
    )

    source_row_text = extract_value(
        chunk,
        "SOURCE ROW",
        "0"
    )

    cik_text = extract_value(
        chunk,
        "CIK",
        "0"
    )

    metadata = {
        "chunk_number": normalize_integer(chunk_number_text),
        "chunk_id": chunk_id,
        "source_row": normalize_integer(source_row_text),
        "symbol": extract_value(chunk, "Company Symbol"),
        "statement": extract_value(chunk, "Financial Statement"),
        "financial_section": extract_value(
            chunk,
            "FINANCIAL SECTION"
        ),
        "fiscal_year": normalize_year(fiscal_year_text),
        "period": extract_value(chunk, "Period"),
        "report_date": extract_value(chunk, "Report Date"),
        "currency": extract_value(chunk, "Reported Currency"),
        "cik": normalize_integer(cik_text),
        "filing_date": extract_value(chunk, "Filing Date"),
    }

    return metadata


# =========================================================
# EMBEDDING MODEL
# =========================================================

def load_embedding_model() -> SentenceTransformer:
    """Load the local Sentence Transformer model."""

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return model


def create_embeddings(
    model: SentenceTransformer,
    chunks: list[str]
) -> list[list[float]]:
    """Generate normalized embeddings for all chunks."""

    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Chroma accepts ordinary Python lists.
    return embeddings.astype("float32").tolist()


# =========================================================
# CHROMA DATABASE
# =========================================================

def create_chroma_collection():
    """Create or open a persistent Chroma collection."""

    os.makedirs(
        CHROMA_DATABASE_FOLDER,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DATABASE_FOLDER
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Financial statement embeddings grouped "
                "by fiscal year and financial section"
            ),
            "hnsw:space": "cosine",
        }
    )

    return collection


def store_chunks_in_chroma(
    collection,
    chunks: list[str],
    embeddings: list[list[float]],
    metadata_list: list[dict[str, Any]]
) -> None:
    """Insert or update financial chunks in Chroma."""

    ids = [
        metadata["chunk_id"]
        for metadata in metadata_list
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata_list
    )

    print(
        f"{len(chunks)} financial chunks stored "
        f"in Chroma DB."
    )


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def search_financial_data(
    collection,
    embedding_model: SentenceTransformer,
    question: str,
    top_k: int = 5,
    fiscal_year: int | None = None,
    financial_section: str | None = None
) -> list[dict[str, Any]]:
    """
    Search the Chroma DB.

    Optional filters:
        fiscal_year=2025
        financial_section="Current Assets"
    """

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32").tolist()

    filters = []

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

    query_parameters = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter is not None:
        query_parameters["where"] = where_filter

    results = collection.query(
        **query_parameters
    )

    formatted_results = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        similarity_score = 1 - float(distance)

        formatted_results.append({
            "similarity_score": similarity_score,
            "metadata": metadata,
            "document": document,
        })

    return formatted_results


def display_search_results(
    question: str,
    results: list[dict[str, Any]]
) -> None:
    """Display semantic-search results."""

    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)

    if not results:
        print("No matching financial records were found.")
        return

    for result_number, result in enumerate(
        results,
        start=1
    ):
        metadata = result["metadata"]

        print(f"\nRESULT {result_number}")
        print("-" * 80)

        print(
            "Similarity score: "
            f"{result['similarity_score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{metadata.get('chunk_id', '')}"
        )

        print(
            f"Symbol: "
            f"{metadata.get('symbol', '')}"
        )

        print(
            f"Fiscal year: "
            f"{metadata.get('fiscal_year', '')}"
        )

        print(
            f"Financial section: "
            f"{metadata.get('financial_section', '')}"
        )

        print("\nRetrieved text:\n")
        print(result["document"])


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Build the Chroma vector database and test retrieval."""

    # Read the financial text file.
    raw_text = read_text_file(
        INPUT_TEXT_FILE
    )

    # Separate the financial sections.
    chunks = split_financial_chunks(
        raw_text
    )

    if not chunks:
        raise ValueError(
            "No chunks were found in the input file."
        )

    print(f"Financial chunks found: {len(chunks)}")

    # Extract metadata.
    metadata_list = [
        extract_chunk_metadata(
            chunk=chunk,
            fallback_number=index
        )
        for index, chunk in enumerate(
            chunks,
            start=1
        )
    ]

    # Load local embedding model.
    embedding_model = load_embedding_model()

    # Create vector embeddings.
    embeddings = create_embeddings(
        model=embedding_model,
        chunks=chunks
    )

    print(
        f"Embedding dimensions: "
        f"{len(embeddings[0])}"
    )

    # Create the persistent database.
    collection = create_chroma_collection()

    # Store chunks and embeddings.
    store_chunks_in_chroma(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
        metadata_list=metadata_list
    )

    print(
        f"Total records in collection: "
        f"{collection.count()}"
    )

    # Example semantic query.
    question = (
        "What was Apple's total debt and net debt in 2025?"
    )

    results = search_financial_data(
        collection=collection,
        embedding_model=embedding_model,
        question=question,
        top_k=3,
        fiscal_year=2025
    )

    display_search_results(
        question=question,
        results=results
    )


if __name__ == "__main__":
    main()