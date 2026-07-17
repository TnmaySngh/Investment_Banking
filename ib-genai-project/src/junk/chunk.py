import os
import re
from typing import Any

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_EXCEL_FILE = (
    r"C:\Projects\project\Investment_Banking"
    r"\ib-genai-project\data\excel\balance_sheet\cleaned\*.xlsx"
)

OUTPUT_TEXT_FILE = (
    r"C:\Projects\project\Investment_Banking"
    r"\ib-genai-project\data\chunks\balance_sheet"
)


# =========================================================
# FINANCIAL COLUMN GROUPS
# =========================================================

FINANCIAL_COLUMN_GROUPS = {
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
    """Return True when a value is empty."""

    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def make_readable_column_name(column_name: str) -> str:
    """
    Convert camelCase names into readable financial labels.

    Example:
        totalCurrentAssets -> Total Current Assets
    """

    text = str(column_name).strip()

    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text
    )

    text = text.replace("_", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.title()


def format_financial_value(value: Any) -> str:
    """Format dates, numbers and text for LLM-readable output."""

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
    """Clean the generated text using regular expressions."""

    # Remove control characters.
    text = re.sub(
        r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]",
        "",
        text
    )

    # Remove trailing spaces.
    text = re.sub(
        r"[ \t]+$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Reduce excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# METADATA
# =========================================================

def create_metadata_text(row: pd.Series) -> str:
    """Create metadata that will be repeated in every chunk."""

    metadata_lines = []

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

        label = metadata_labels.get(
            column,
            make_readable_column_name(column)
        )

        metadata_lines.append(
            f"{label}: {format_financial_value(value)}"
        )

    return "\n".join(metadata_lines)


# =========================================================
# FINANCIAL CHUNK GENERATION
# =========================================================

def create_financial_section_chunk(
    row: pd.Series,
    section_name: str,
    section_columns: list[str],
    row_number: int
) -> str | None:
    """
    Create one financial-semantic chunk for one fiscal period.
    """

    symbol = format_financial_value(
        row.get("symbol", "Unknown")
    )

    fiscal_year = format_financial_value(
        row.get("fiscalYear", "Unknown")
    )

    period = format_financial_value(
        row.get("period", "Unknown")
    )

    chunk_id = (
        f"{symbol}_"
        f"{fiscal_year}_"
        f"{period}_"
        f"{section_name.replace(' ', '_').lower()}"
    )

    metric_lines = []

    for column in section_columns:
        if column not in row.index:
            continue

        value = row[column]

        if is_empty(value):
            continue

        label = make_readable_column_name(column)
        formatted_value = format_financial_value(value)

        metric_lines.append(
            f"- {label}: {formatted_value}"
        )

    if not metric_lines:
        return None

    chunk_lines = [
        "=" * 80,
        f"CHUNK ID: {chunk_id}",
        f"SOURCE ROW: {row_number}",
        f"FINANCIAL SECTION: {section_name}",
        "=" * 80,
        "",
        create_metadata_text(row),
        "",
        f"{section_name} Metrics:",
        *metric_lines,
        "",
    ]

    return clean_generated_text(
        "\n".join(chunk_lines)
    )


def create_column_based_chunks(
    dataframe: pd.DataFrame
) -> list[str]:
    """
    Generate financial chunks based on related Excel columns.
    """

    chunks = []

    for dataframe_index, row in dataframe.iterrows():

        # Add 2 because Excel starts at row 1 and row 1 is the header.
        excel_row_number = dataframe_index + 2

        for section_name, columns in FINANCIAL_COLUMN_GROUPS.items():

            chunk = create_financial_section_chunk(
                row=row,
                section_name=section_name,
                section_columns=columns,
                row_number=excel_row_number
            )

            if chunk:
                chunks.append(chunk)

    return chunks


# =========================================================
# SAVE TEXT OUTPUT
# =========================================================

def save_chunks_to_text(
    chunks: list[str],
    output_text_file: str
) -> None:
    """Save all chunks into one text file."""

    output_folder = os.path.dirname(output_text_file)

    if output_folder:
        os.makedirs(
            output_folder,
            exist_ok=True
        )

    with open(
        output_text_file,
        "w",
        encoding="utf-8"
    ) as file:

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):
            file.write(
                f"CHUNK NUMBER: {chunk_number}\n"
            )

            file.write(chunk)

            file.write(
                "\n\n"
            )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Read the Excel workbook and generate financial chunks."""

    if not os.path.isfile(INPUT_EXCEL_FILE):
        raise FileNotFoundError(
            f"Input Excel file was not found:\n"
            f"{INPUT_EXCEL_FILE}"
        )

    dataframe = pd.read_excel(
        INPUT_EXCEL_FILE,
        engine="openpyxl"
    )

    if dataframe.empty:
        raise ValueError(
            "The Excel file does not contain any rows."
        )

    # Clean column names.
    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    chunks = create_column_based_chunks(
        dataframe
    )

    if not chunks:
        raise ValueError(
            "No financial chunks were generated. "
            "Verify that the Excel column names match the configured groups."
        )

    save_chunks_to_text(
        chunks=chunks,
        output_text_file=OUTPUT_TEXT_FILE
    )

    print("-" * 70)
    print("Financial column-based chunking completed.")
    print(f"Rows processed : {len(dataframe)}")
    print(f"Chunks created : {len(chunks)}")
    print(f"Output file    : {OUTPUT_TEXT_FILE}")
    print("-" * 70)


if __name__ == "__main__":
    main()