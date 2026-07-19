from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from AAPL_build_faiss_index import (
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_RETRY_BACKOFF_SECONDS,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    ChunkRecord,
    load_embedding_client,
    load_faiss_index,
    load_metadata,
    search_index,
)

# =========================================================
# CONFIGURATION
# =========================================================

# Where to write a full CSV export when --export-csv is used.
CSV_EXPORT_FILE = METADATA_FILE.parent / "aapl_financials_metadata_full.csv"

# How many characters of chunk text to show in list views before truncating.
LIST_PREVIEW_CHARS = 100


# =========================================================
# SUMMARY VIEW
# =========================================================

def print_index_summary(records: list[ChunkRecord], index_total_vectors: int) -> None:
    print("=" * 80)
    print("FAISS INDEX SUMMARY")
    print("=" * 80)
    print(f"Vectors in FAISS index : {index_total_vectors}")
    print(f"Records in metadata     : {len(records)}")

    if index_total_vectors != len(records):
        print(
            "  WARNING: vector count and metadata count do not match. "
            "The index and metadata files may be out of sync "
            "(e.g. rebuilt separately)."
        )

    print()
    print("By statement type source:")
    for statement_type_source, count in Counter(
        record.statement_type_source for record in records
    ).most_common():
        print(f"  {statement_type_source:<20} {count}")

    print()
    print("By financial section:")
    for financial_section, count in Counter(
        record.financial_section for record in records
    ).most_common():
        label = financial_section or "(none)"
        print(f"  {label:<35} {count}")

    print()
    print("By statement type (annual / quarterly / ttm):")
    for statement_type, count in Counter(
        record.statement_type for record in records
    ).most_common():
        label = statement_type or "(none)"
        print(f"  {label:<15} {count}")

    print()
    print("By source file:")
    for source_file, count in Counter(
        record.source_file for record in records
    ).most_common():
        label = source_file or "(none)"
        print(f"  {label:<45} {count}")

    print("=" * 80)


# =========================================================
# LIST / BROWSE VIEW
# =========================================================

def filter_records(
    records: list[ChunkRecord],
    statement_type_source: str | None,
    financial_section: str | None,
    chunk_id_contains: str | None,
) -> list[tuple[int, ChunkRecord]]:
    """Return (original_index, record) pairs matching all provided filters."""
    indexed_records = list(enumerate(records))

    if statement_type_source:
        indexed_records = [
            (i, r) for i, r in indexed_records
            if r.statement_type_source == statement_type_source
        ]

    if financial_section:
        target = financial_section.lower()
        indexed_records = [
            (i, r) for i, r in indexed_records
            if target in r.financial_section.lower()
        ]

    if chunk_id_contains:
        target = chunk_id_contains.lower()
        indexed_records = [
            (i, r) for i, r in indexed_records
            if target in r.chunk_id.lower()
        ]

    return indexed_records


def print_record_list(indexed_records: list[tuple[int, ChunkRecord]], limit: int) -> None:
    total_matches = len(indexed_records)
    shown = indexed_records[:limit]

    print(f"Matching records: {total_matches} (showing {len(shown)})")
    print("-" * 80)

    for record_index, record in shown:
        first_line = record.text.strip().splitlines()[0] if record.text.strip() else ""
        preview = record.chunk_id or first_line[:LIST_PREVIEW_CHARS]

        print(
            f"[{record_index:>5}] "
            f"{record.statement_type_source:<16} | "
            f"{record.financial_section:<25} | "
            f"{record.statement_type:<10} | "
            f"{preview}"
        )

    if total_matches > len(shown):
        print(f"... {total_matches - len(shown)} more not shown (increase --limit to see more)")


# =========================================================
# SINGLE-CHUNK VIEW
# =========================================================

def print_full_chunk(record_index: int, records: list[ChunkRecord]) -> None:
    if record_index < 0 or record_index >= len(records):
        print(f"Index {record_index} is out of range (0-{len(records) - 1}).")
        return

    record = records[record_index]
    print("=" * 80)
    print(f"RECORD INDEX: {record_index}")
    print("=" * 80)
    print(record.text)
    print("=" * 80)


def find_by_chunk_id(chunk_id: str, records: list[ChunkRecord]) -> int | None:
    for record_index, record in enumerate(records):
        if record.chunk_id == chunk_id:
            return record_index
    return None


# =========================================================
# CSV EXPORT
# =========================================================

def export_metadata_to_csv(records: list[ChunkRecord], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "index",
                "chunk_id",
                "statement_type_source",
                "source_file",
                "source_row",
                "statement_type",
                "financial_section",
            ]
        )

        for record_index, record in enumerate(records):
            writer.writerow(
                [
                    record_index,
                    record.chunk_id,
                    record.statement_type_source,
                    record.source_file,
                    record.source_row,
                    record.statement_type,
                    record.financial_section,
                ]
            )

    print(f"Exported {len(records)} records to: {output_file}")


# =========================================================
# SEMANTIC SEARCH VIEW
# =========================================================

def run_search(
    query: str,
    records: list[ChunkRecord],
    index,
    top_k: int,
    statement_type_filter: str | None,
) -> None:
    client = load_embedding_client()

    print(f"Query: {query!r}")
    if statement_type_filter:
        print(f"Filter: statement_type_source == {statement_type_filter!r}")
    print("-" * 80)

    results = search_index(
        query=query,
        client=client,
        index=index,
        records=records,
        top_k=top_k,
        statement_type_filter=statement_type_filter,
    )

    if not results:
        print("No results.")
        return

    for rank, (score, record) in enumerate(results, start=1):
        print(
            f"#{rank} score={score:.4f} "
            f"[{record.statement_type_source} | {record.financial_section}] "
            f"{record.chunk_id}"
        )
        print("-" * 40)
        print(record.text)
        print()


# =========================================================
# CLI
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and browse the FAISS index built by AAPL_build_faiss_index.py",
    )

    parser.add_argument(
        "--index-file",
        type=Path,
        default=FAISS_INDEX_FILE,
        help="Path to the .index file (default: value from AAPL_build_faiss_index.py)",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=METADATA_FILE,
        help="Path to the metadata .pkl file (default: value from AAPL_build_faiss_index.py)",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("summary", help="Print counts and breakdowns for the index")

    list_parser = subparsers.add_parser("list", help="List chunks, optionally filtered")
    list_parser.add_argument("--statement-type-source", default=None, help="e.g. balance_sheet, cash_flow, income_statement")
    list_parser.add_argument("--financial-section", default=None, help="Substring match, e.g. 'Current Assets'")
    list_parser.add_argument("--chunk-id-contains", default=None, help="Substring match against chunk_id")
    list_parser.add_argument("--limit", type=int, default=25, help="Max rows to display (default: 25)")

    view_parser = subparsers.add_parser("view", help="Print the full text of one chunk")
    view_group = view_parser.add_mutually_exclusive_group(required=True)
    view_group.add_argument("--index", type=int, help="Record index (as shown by the 'list' command)")
    view_group.add_argument("--chunk-id", type=str, help="Exact chunk_id to look up")

    export_parser = subparsers.add_parser("export-csv", help="Export all metadata fields to a CSV file")
    export_parser.add_argument("--output-file", type=Path, default=CSV_EXPORT_FILE, help="Output CSV path")

    search_parser = subparsers.add_parser("search", help="Run a semantic search against the index (requires OPENAI_API_KEY)")
    search_parser.add_argument("query", type=str, help="Natural-language query")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    search_parser.add_argument(
        "--statement-type-source",
        default=None,
        help="Restrict results to one source, e.g. balance_sheet, cash_flow, income_statement",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command is None:
        args.command = "summary"
        print("No command given — defaulting to 'summary'. "
              "Run with -h to see all available commands.\n")

    if not args.index_file.exists():
        raise FileNotFoundError(f"FAISS index file not found: {args.index_file}")
    if not args.metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_file}")

    index = load_faiss_index(args.index_file)
    records = load_metadata(args.metadata_file)

    if args.command == "summary":
        print_index_summary(records, index.ntotal)

    elif args.command == "list":
        indexed_records = filter_records(
            records=records,
            statement_type_source=args.statement_type_source,
            financial_section=args.financial_section,
            chunk_id_contains=args.chunk_id_contains,
        )
        print_record_list(indexed_records, limit=args.limit)

    elif args.command == "view":
        if args.chunk_id is not None:
            record_index = find_by_chunk_id(args.chunk_id, records)
            if record_index is None:
                print(f"No record found with chunk_id: {args.chunk_id}")
                return
        else:
            record_index = args.index

        print_full_chunk(record_index, records)

    elif args.command == "export-csv":
        export_metadata_to_csv(records, args.output_file)

    elif args.command == "search":
        run_search(
            query=args.query,
            records=records,
            index=index,
            top_k=args.top_k,
            statement_type_filter=args.statement_type_source,
        )


if __name__ == "__main__":
    main()
