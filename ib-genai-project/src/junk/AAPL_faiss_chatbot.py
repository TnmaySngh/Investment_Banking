from __future__ import annotations

from AAPL_build_faiss_index import (
    ChunkRecord,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    load_embedding_client,
    load_faiss_index,
    load_metadata,
    search_index,
)

# =========================================================
# CONFIGURATION
# =========================================================

# Chat model used to turn retrieved chunks into a natural-language answer.
# "gpt-4o-mini" is cheap and fast; swap for "gpt-4o" for higher quality.
CHAT_MODEL_NAME = "gpt-4o-mini"

# How many chunks to retrieve per question.
TOP_K_RESULTS = 5

CHAT_TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are a financial research assistant for Apple (AAPL).
Answer the user's question using ONLY the CONTEXT chunks provided below.
Each chunk is a snippet from AAPL's balance sheet, cash flow, or income
statement filings.

Rules:
- If the context does not contain enough information to answer, say so
  clearly instead of guessing or using outside knowledge.
- When you state a number, mention which period/fiscal year it came from.
- Keep answers concise and factual. Do not speculate.
- After your answer, do not repeat the raw context back verbatim.
"""

# Valid values a user can type after "/filter" to restrict retrieval.
VALID_STATEMENT_FILTERS = {"balance_sheet", "cash_flow", "income_statement"}


# =========================================================
# CONTEXT ASSEMBLY
# =========================================================

def build_context_block(results: list[tuple[float, ChunkRecord]]) -> str:
    """Turn retrieved (score, ChunkRecord) pairs into a single context string."""
    context_sections = []

    for rank, (score, record) in enumerate(results, start=1):
        context_sections.append(
            f"[Chunk {rank} | source: {record.statement_type_source} | "
            f"section: {record.financial_section} | "
            f"chunk_id: {record.chunk_id} | similarity: {score:.3f}]\n"
            f"{record.text}"
        )

    return "\n\n".join(context_sections)


def build_user_prompt(question: str, context_block: str) -> str:
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer the question using only the context above."
    )


# =========================================================
# CHAT COMPLETION
# =========================================================

def generate_answer(client, question: str, context_block: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL_NAME,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context_block)},
        ],
    )
    return response.choices[0].message.content.strip()


# =========================================================
# SOURCE DISPLAY
# =========================================================

def print_sources(results: list[tuple[float, ChunkRecord]]) -> None:
    print("\nSources:")
    for rank, (score, record) in enumerate(results, start=1):
        print(
            f"  [{rank}] score={score:.3f} "
            f"{record.statement_type_source} | {record.financial_section} | "
            f"{record.chunk_id}"
        )


# =========================================================
# CHAT LOOP
# =========================================================

def print_help() -> None:
    print(
        "\nCommands:\n"
        "  /filter balance_sheet | cash_flow | income_statement   "
        "restrict retrieval to one statement type\n"
        "  /filter clear                                          "
        "remove the current filter\n"
        "  /topk N                                                "
        "change how many chunks are retrieved (currently "
        f"{TOP_K_RESULTS})\n"
        "  /help                                                  show this message\n"
        "  /exit  or  /quit                                       leave the chatbot\n"
    )


def run_chat_loop(client, index, records: list[ChunkRecord]) -> None:
    active_filter: str | None = None
    top_k = TOP_K_RESULTS

    print("=" * 80)
    print("AAPL Financial Statement Chatbot")
    print("=" * 80)
    print(f"Loaded {len(records)} chunks. Type /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "/quit"}:
            print("Exiting.")
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower().startswith("/filter"):
            argument = user_input[len("/filter"):].strip().lower()
            if argument in ("clear", ""):
                active_filter = None
                print("Filter cleared. Searching across all statement types.\n")
            elif argument in VALID_STATEMENT_FILTERS:
                active_filter = argument
                print(f"Filter set to: {active_filter}\n")
            else:
                print(
                    f"Unknown filter '{argument}'. Valid options: "
                    f"{', '.join(sorted(VALID_STATEMENT_FILTERS))}, or 'clear'.\n"
                )
            continue

        if user_input.lower().startswith("/topk"):
            argument = user_input[len("/topk"):].strip()
            if argument.isdigit() and int(argument) > 0:
                top_k = int(argument)
                print(f"top_k set to {top_k}\n")
            else:
                print("Usage: /topk N   (N must be a positive integer)\n")
            continue

        try:
            results = search_index(
                query=user_input,
                client=client,
                index=index,
                records=records,
                top_k=top_k,
                statement_type_filter=active_filter,
            )
        except Exception as error:  # noqa: BLE001 - surface any API/retrieval error to the user
            print(f"Retrieval failed: {error}\n")
            continue

        if not results:
            print("No matching chunks found for that query.\n")
            continue

        context_block = build_context_block(results)

        try:
            answer = generate_answer(client, user_input, context_block)
        except Exception as error:  # noqa: BLE001 - surface any chat API error to the user
            print(f"Answer generation failed: {error}\n")
            continue

        print(f"\nBot: {answer}")
        print_sources(results)
        print()


# =========================================================
# MAIN
# =========================================================

def main() -> None:
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

    print("Loading FAISS index and metadata...")
    index = load_faiss_index(FAISS_INDEX_FILE)
    records = load_metadata(METADATA_FILE)

    client = load_embedding_client()

    run_chat_loop(client, index, records)


if __name__ == "__main__":
    main()
