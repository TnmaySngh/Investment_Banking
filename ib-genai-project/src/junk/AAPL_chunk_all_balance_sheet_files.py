from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_EXCEL_FOLDER = Path(
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
    r"\ib-genai-project\data\excel\AAPL\balance_sheet\cleaned"
)

OUTPUT_CHUNK_FOLDER = Path(
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
    r"\ib-genai-project\data\chunks\AAPL\balance_sheet"
)

COMBINED_OUTPUT_FILE = OUTPUT_CHUNK_FOLDER / "balance_sheet_all_chunks.txt"

# =========================================================
# FINANCIAL COLUMN GROUPS
# =========================================================

FINANCIAL_COLUMN_GROUPS: dict[str, list[str]] = {
    "Current Assets": [
        "cashAndCashEquivalents",
        "shortTermInvestments",
        "cashAndShortTermInvestments",
        "netReceivables",
        "accountsReceivables",
        "otherReceivables",
        "inventory",
        "prepaids",
        "otherCurrentAssets",
        "totalCurrentAssets",
    ],
    "Non-Current Assets": [
        "propertyPlantEquipmentNet",
        "goodwill",
        "intangibleAssets",
        "goodwillAndIntangibleAssets",
        "longTermInvestments",
        "taxAssets",
        "otherNonCurrentAssets",
        "totalNonCurrentAssets",
        "otherAssets",
        "totalAssets",
    ],
    "Current Liabilities": [
        "totalPayables",
        "accountPayables",
        "otherPayables",
        "accruedExpenses",
        "shortTermDebt",
        "capitalLeaseObligationsCurrent",
        "taxPayables",
        "deferredRevenue",
        "otherCurrentLiabilities",
        "totalCurrentLiabilities",
    ],
    "Non-Current Liabilities": [
        "longTermDebt",
        "capitalLeaseObligationsNonCurrent",
        "deferredRevenueNonCurrent",
        "deferredTaxLiabilitiesNonCurrent",
        "otherNonCurrentLiabilities",
        "totalNonCurrentLiabilities",
        "otherLiabilities",
        "capitalLeaseObligations",
        "totalLiabilities",
    ],
    "Shareholders Equity": [
        "treasuryStock",
        "preferredStock",
        "commonStock",
        "retainedEarnings",
        "additionalPaidInCapital",
        "accumulatedOtherComprehensiveIncomeLoss",
        "otherTotalStockholdersEquity",
        "totalStockholdersEquity",
        "totalEquity",
        "minorityInterest",
        "totalLiabilitiesAndTotalEquity",
    ],
    "Debt and Investment Summary": [
        "totalInvestments",
        "shortTermDebt",
        "longTermDebt",
        "capitalLeaseObligations",
        "totalDebt",
        "cashAndCashEquivalents",
        "netDebt",
    ],
}

METADATA_COLUMNS = [
    "symbol",
    "group",
    "date",
    "reportedCurrency",
    "cik",
    "filingDate",
    "acceptedDate",
    "fiscalYear",
    "period",
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def is_empty(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def make_readable_column_name(column_name: str) -> str:
    text = str(column_name).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.title()


def format_financial_value(value: Any) -> str:
    if is_empty(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"

    return str(value).strip()


def clean_generated_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_identifier(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def detect_statement_type(excel_file: Path) -> str:
    """Infer annual, quarterly, or ttm from the filename."""
    filename = excel_file.stem.lower()

    if "annual" in filename:
        return "annual"
    if "quarterly" in filename:
        return "quarterly"
    if "ttm" in filename:
        return "ttm"

    return "unknown"


# =========================================================
# METADATA
# =========================================================

def create_metadata_text(
    row: pd.Series,
    source_file: str,
    statement_type: str,
) -> str:
    metadata_lines = [
        f"Source File: {source_file}",
        f"Statement Type: {statement_type}",
    ]

    metadata_labels = {
        "symbol": "Company Symbol",
        "group": "Financial Statement",
        "date": "Report Date",
        "reportedCurrency": "Reported Currency",
        "cik": "CIK",
        "filingDate": "Filing Date",
        "acceptedDate": "Accepted Date",
        "fiscalYear": "Fiscal Year",
        "period": "Period",
    }

    for column in METADATA_COLUMNS:
        if column not in row.index:
            continue

        value = row[column]
        if is_empty(value):
            continue

        label = metadata_labels.get(column, make_readable_column_name(column))
        metadata_lines.append(f"{label}: {format_financial_value(value)}")

    return "\n".join(metadata_lines)


# =========================================================
# CHUNK GENERATION
# =========================================================

def create_financial_section_chunk(
    row: pd.Series,
    section_name: str,
    section_columns: list[str],
    excel_row_number: int,
    source_file: str,
    statement_type: str,
) -> str | None:
    symbol = format_financial_value(row.get("symbol", "Unknown"))
    fiscal_year = format_financial_value(row.get("fiscalYear", "Unknown"))
    period = format_financial_value(row.get("period", "Unknown"))
    report_date = format_financial_value(row.get("date", "Unknown"))

    # Include statement type and report date to prevent duplicate IDs
    # across annual, quarterly, and TTM files.
    chunk_id = "_".join(
        [
            normalize_identifier(symbol),
            normalize_identifier(statement_type),
            normalize_identifier(fiscal_year),
            normalize_identifier(period),
            normalize_identifier(report_date),
            normalize_identifier(section_name),
        ]
    )

    metric_lines: list[str] = []

    for column in section_columns:
        if column not in row.index:
            continue

        value = row[column]
        if is_empty(value):
            continue

        label = make_readable_column_name(column)
        formatted_value = format_financial_value(value)
        metric_lines.append(f"- {label}: {formatted_value}")

    if not metric_lines:
        return None

    chunk_lines = [
        "=" * 80,
        f"CHUNK ID: {chunk_id}",
        f"SOURCE FILE: {source_file}",
        f"SOURCE ROW: {excel_row_number}",
        f"STATEMENT TYPE: {statement_type}",
        f"FINANCIAL SECTION: {section_name}",
        "=" * 80,
        "",
        create_metadata_text(
            row=row,
            source_file=source_file,
            statement_type=statement_type,
        ),
        "",
        f"{section_name} Metrics:",
        *metric_lines,
    ]

    return clean_generated_text("\n".join(chunk_lines))


def create_chunks_for_dataframe(
    dataframe: pd.DataFrame,
    source_file: str,
    statement_type: str,
) -> list[str]:
    chunks: list[str] = []

    for dataframe_index, row in dataframe.iterrows():
        excel_row_number = dataframe_index + 2

        for section_name, columns in FINANCIAL_COLUMN_GROUPS.items():
            chunk = create_financial_section_chunk(
                row=row,
                section_name=section_name,
                section_columns=columns,
                excel_row_number=excel_row_number,
                source_file=source_file,
                statement_type=statement_type,
            )

            if chunk:
                chunks.append(chunk)

    return chunks


# =========================================================
# FILE PROCESSING
# =========================================================

def read_and_chunk_excel_file(excel_file: Path) -> list[str]:
    dataframe = pd.read_excel(excel_file, engine="openpyxl")

    if dataframe.empty:
        print(f"Skipped empty file: {excel_file.name}")
        return []

    dataframe.columns = dataframe.columns.astype(str).str.strip()
    statement_type = detect_statement_type(excel_file)

    return create_chunks_for_dataframe(
        dataframe=dataframe,
        source_file=excel_file.name,
        statement_type=statement_type,
    )


def save_chunks_to_text(chunks: list[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for chunk_number, chunk in enumerate(chunks, start=1):
            file.write(f"CHUNK NUMBER: {chunk_number}\n")
            file.write(chunk)
            file.write("\n\n")

# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Read every Excel file and create one combined chunk file."""

    if not INPUT_EXCEL_FOLDER.exists():
        raise FileNotFoundError(
            "Input folder does not exist:\n"
            f"{INPUT_EXCEL_FOLDER}"
        )

    if not INPUT_EXCEL_FOLDER.is_dir():
        raise NotADirectoryError(
            "The configured input path is not a folder:\n"
            f"{INPUT_EXCEL_FOLDER}"
        )

    excel_files = sorted(
        file
        for file in INPUT_EXCEL_FOLDER.glob("*.xlsx")
        if not file.name.startswith("~$")
    )

    if not excel_files:
        raise FileNotFoundError(
            "No .xlsx files were found in:\n"
            f"{INPUT_EXCEL_FOLDER}"
        )

    all_chunks: list[str] = []
    total_rows = 0
    processed_files = 0
    failed_files = 0

    print("-" * 80)
    print(f"Input folder     : {INPUT_EXCEL_FOLDER}")
    print(f"Excel files found: {len(excel_files)}")
    print("-" * 80)

    for excel_file in excel_files:
        try:
            print(f"Processing: {excel_file.name}")

            chunks = read_and_chunk_excel_file(
                excel_file
            )

            dataframe = pd.read_excel(
                excel_file,
                engine="openpyxl",
            )

            row_count = len(dataframe)

            all_chunks.extend(chunks)
            total_rows += row_count
            processed_files += 1

            print(
                f"  Rows: {row_count} | "
                f"Chunks: {len(chunks)}"
            )

        except PermissionError:
            failed_files += 1
            print(
                f"  Failed: Permission denied. "
                f"Close {excel_file.name} in Excel."
            )

        except Exception as error:
            failed_files += 1
            print(f"  Failed: {error}")

    if not all_chunks:
        raise ValueError(
            "No chunks were generated. "
            "Verify the Excel files and configured column names."
        )

    save_chunks_to_text(
        chunks=all_chunks,
        output_file=COMBINED_OUTPUT_FILE,
    )

    print("-" * 80)
    print("Financial chunking completed successfully.")
    print(f"Files processed : {processed_files}")
    print(f"Files failed    : {failed_files}")
    print(f"Rows processed  : {total_rows}")
    print(f"Chunks created  : {len(all_chunks)}")
    print(f"Output file     : {COMBINED_OUTPUT_FILE}")
    print("-" * 80)


if __name__ == "__main__":
    main()
