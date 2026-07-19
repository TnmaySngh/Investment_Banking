from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

# =========================================================
# CONFIGURATION
# =========================================================

# One entry per statement type. Each points at the combined
# chunk .txt file produced by the AAPL_chunk_all_*.py scripts.
CHUNK_SOURCES: dict[str, Path] = {
    "balance_sheet": Path(
        r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
        r"\ib-genai-project\data\chunks\AAPL\balance_sheet"
        r"\balance_sheet_all_chunks.txt"
    ),
    "cash_flow": Path(
        r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
        r"\ib-genai-project\data\chunks\AAPL\cash_flow"
        r"\cash_flow_all_chunks.txt"
    ),
    "income_statement": Path(
        r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
        r"\ib-genai-project\data\chunks\AAPL\income_statement"
        r"\income_statement_all_chunks.txt"
    ),
}

OUTPUT_FOLDER = Path(
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
    r"\ib-genai-project\data\faiss_db\AAPL"
)

FAISS_INDEX_FILE = OUTPUT_FOLDER / "aapl_financials.index"
METADATA_FILE = OUTPUT_FOLDER / "aapl_financials_metadata.pkl"
METADATA_JSON_PREVIEW_FILE = OUTPUT_FOLDER / "aapl_financials_metadata_preview.json"

# OpenAI's small embedding model: 1536-dim, cheap, strong general-purpose
# quality. Requires an OPENAI_API_KEY environment variable to be set.
# Swap for "text-embedding-3-large" (3072-dim, higher quality, costs more)
# if you need better retrieval accuracy.
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# OpenAI's embeddings endpoint accepts large batches, but a smaller batch
# keeps individual request payloads and retry cost small.
EMBEDDING_BATCH_SIZE = 100

# Retry behavior for transient API errors (rate limits, timeouts).
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_RETRY_BACKOFF_SECONDS = 2.0

# Split marker used by the chunk-generation scripts.
CHUNK_SPLIT_PATTERN = re.compile(r"(?=^CHUNK NUMBER: \d+)", re.MULTILINE)

# Regex used to pull structured fields out of each chunk's header block.
CHUNK_HEADER_FIELD_PATTERN = re.compile(
    r"^(CHUNK ID|SOURCE FILE|SOURCE ROW|STATEMENT TYPE|FINANCIAL SECTION):\s*(.*)$",
    re.MULTILINE,
)


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class ChunkRecord:
    """One embeddable unit plus everything needed to trace it back to source."""

    text: str
    statement_type_source: str  # which CHUNK_SOURCES key this came from
    chunk_id: str = ""
    source_file: str = ""
    source_row: str = ""
    statement_type: str = ""
    financial_section: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# =========================================================
# PARSING
# =========================================================

def split_combined_chunk_file(raw_text: str) -> list[str]:
    """Split a combined chunk .txt file back into individual chunk blocks."""
    pieces = CHUNK_SPLIT_PATTERN.split(raw_text)
    return [piece.strip() for piece in pieces if piece.strip()]


def parse_chunk_block(chunk_block: str, statement_type_source: str) -> ChunkRecord:
    """Extract header metadata fields from a single chunk block."""
    fields = dict(CHUNK_HEADER_FIELD_PATTERN.findall(chunk_block))

    return ChunkRecord(
        text=chunk_block,
        statement_type_source=statement_type_source,
        chunk_id=fields.get("CHUNK ID", ""),
        source_file=fields.get("SOURCE FILE", ""),
        source_row=fields.get("SOURCE ROW", ""),
        statement_type=fields.get("STATEMENT TYPE", ""),
        financial_section=fields.get("FINANCIAL SECTION", ""),
    )


def load_chunk_records(chunk_sources: dict[str, Path]) -> list[ChunkRecord]:
    """Read every combined chunk file and parse it into ChunkRecord objects."""
    all_records: list[ChunkRecord] = []

    for statement_type_source, chunk_file in chunk_sources.items():
        if not chunk_file.exists():
            print(f"  Skipping missing file: {chunk_file}")
            continue

        raw_text = chunk_file.read_text(encoding="utf-8")
        chunk_blocks = split_combined_chunk_file(raw_text)

        records = [
            parse_chunk_block(block, statement_type_source)
            for block in chunk_blocks
        ]

        all_records.extend(records)
        print(f"  Loaded {len(records)} chunks from {chunk_file.name}")

    return all_records


# =========================================================
# EMBEDDING
# =========================================================

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
            # response.data is returned in the same order as the input list.
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


def embed_chunk_records(
    client: OpenAI,
    records: list[ChunkRecord],
    model_name: str,
    batch_size: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> np.ndarray:
    """Encode all chunk texts into a float32 embedding matrix via the OpenAI API."""
    texts = [record.text for record in records]
    all_embeddings: list[list[float]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_number, batch_start in enumerate(range(0, len(texts), batch_size), start=1):
        batch_texts = texts[batch_start : batch_start + batch_size]

        print(f"  Embedding batch {batch_number}/{total_batches} ({len(batch_texts)} chunks)")

        batch_embeddings = embed_text_batch(
            client=client,
            texts=batch_texts,
            model_name=model_name,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype="float32")
    return normalize_embeddings(embeddings)  # pre-normalize so inner product == cosine


# =========================================================
# FAISS INDEX
# =========================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a flat inner-product index (cosine similarity, exact search)."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array of shape (n_chunks, dim).")

    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)

    # IDs let us map FAISS result positions back to ChunkRecord entries
    # even if we later delete/rebuild subsets.
    id_mapped_index = faiss.IndexIDMap2(index)
    ids = np.arange(embeddings.shape[0]).astype("int64")
    id_mapped_index.add_with_ids(embeddings, ids)

    return id_mapped_index


def save_faiss_index(index: faiss.Index, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_file))


def save_metadata(records: list[ChunkRecord], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump(records, file)


def save_metadata_preview(records: list[ChunkRecord], output_file: Path, limit: int = 20) -> None:
    """Write a small human-readable JSON preview, useful for sanity-checking."""
    preview = [
        {
            "index": index,
            "chunk_id": record.chunk_id,
            "source_file": record.source_file,
            "statement_type": record.statement_type,
            "statement_type_source": record.statement_type_source,
            "financial_section": record.financial_section,
        }
        for index, record in enumerate(records[:limit])
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(preview, file, indent=2)


# =========================================================
# QUERY / RETRIEVAL (for testing the index after it is built)
# =========================================================

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
    statement_type_filter: str | None = None,
) -> list[tuple[float, ChunkRecord]]:
    """Embed a query and return the top_k most similar chunk records.

    statement_type_filter can be one of the CHUNK_SOURCES keys
    (e.g. 'balance_sheet') to restrict results to one statement type.
    Filtering is done by over-fetching then narrowing, since FAISS
    IndexFlatIP has no native metadata filter.
    """
    raw_embedding = embed_text_batch(
        client=client,
        texts=[query],
        model_name=EMBEDDING_MODEL_NAME,
        max_retries=EMBEDDING_MAX_RETRIES,
        retry_backoff_seconds=EMBEDDING_RETRY_BACKOFF_SECONDS,
    )
    query_embedding = normalize_embeddings(np.array(raw_embedding, dtype="float32"))

    fetch_k = top_k * 5 if statement_type_filter else top_k
    scores, ids = index.search(query_embedding, fetch_k)

    results: list[tuple[float, ChunkRecord]] = []
    for score, record_id in zip(scores[0], ids[0]):
        if record_id == -1:
            continue

        record = records[record_id]

        if statement_type_filter and record.statement_type_source != statement_type_filter:
            continue

        results.append((float(score), record))

        if len(results) >= top_k:
            break

    return results


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("-" * 80)
    print("Loading chunk files")
    print("-" * 80)

    records = load_chunk_records(CHUNK_SOURCES)

    if not records:
        raise ValueError(
            "No chunks were loaded. Verify CHUNK_SOURCES paths and "
            "that the chunk-generation scripts have already been run."
        )

    print(f"Total chunks loaded: {len(records)}")
    print("-" * 80)

    client = load_embedding_client()

    print("-" * 80)
    print(f"Generating embeddings with {EMBEDDING_MODEL_NAME}")
    print("-" * 80)
    embeddings = embed_chunk_records(
        client=client,
        records=records,
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        max_retries=EMBEDDING_MAX_RETRIES,
        retry_backoff_seconds=EMBEDDING_RETRY_BACKOFF_SECONDS,
    )
    print(f"Embedding matrix shape: {embeddings.shape}")

    print("-" * 80)
    print("Building FAISS index")
    print("-" * 80)
    index = build_faiss_index(embeddings)
    print(f"FAISS index size: {index.ntotal} vectors")

    save_faiss_index(index, FAISS_INDEX_FILE)
    save_metadata(records, METADATA_FILE)
    save_metadata_preview(records, METADATA_JSON_PREVIEW_FILE)

    print("-" * 80)
    print("Embedding and indexing completed successfully.")
    print(f"FAISS index file : {FAISS_INDEX_FILE}")
    print(f"Metadata file    : {METADATA_FILE}")
    print(f"Preview file     : {METADATA_JSON_PREVIEW_FILE}")
    print("-" * 80)

    # Quick sanity-check query so you can confirm retrieval works
    # immediately after building the index.
    sample_query = "What was Apple's total current assets?"
    print(f"Sample query: {sample_query!r}")
    sample_results = search_index(sample_query, client, index, records, top_k=3)

    for rank, (score, record) in enumerate(sample_results, start=1):
        print(
            f"  #{rank} score={score:.4f} "
            f"[{record.statement_type_source} | {record.financial_section}] "
            f"{record.chunk_id}"
        )


if __name__ == "__main__":
    main()
