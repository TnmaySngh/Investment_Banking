from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================
# One config block per statement type. Each statement type has its own
# input folder (where the cleaned .xlsx files live), its own output
# folder/file, and its own column groupings (since growth / key metrics /
# ratios each expose a completely different set of columns).

BASE_DATA_FOLDER = Path(
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data"
    r"\excel\AAPL\metrics_ratio_growth\cleaned"
)

# growth_clean.xlsx, key_metrics_clean.xlsx, and ratios_clean.xlsx all live
# together in BASE_DATA_FOLDER, so every statement type below reads from
# this same input folder and instead filters by filename pattern to pick
# out only the file(s) that belong to it (see "file_pattern" in
# STATEMENT_TYPE_CONFIGS further down).
OUTPUT_BASE_FOLDER = Path(
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data"
    r"\chunks\AAPL"
)

# =========================================================
# FINANCIAL COLUMN GROUPS — per statement type
# =========================================================

GROWTH_COLUMN_GROUPS: dict[str, list[str]] = {
    "Income Statement Growth": [
        "revenueGrowth",
        "grossProfitGrowth",
        "ebitgrowth",
        "ebitdaGrowth",
        "operatingIncomeGrowth",
        "netIncomeGrowth",
        "epsgrowth",
        "epsdilutedGrowth",
        "weightedAverageSharesGrowth",
        "weightedAverageSharesDilutedGrowth",
    ],
    "Cash Flow & Capex Growth": [
        "operatingCashFlowGrowth",
        "freeCashFlowGrowth",
        "growthCapitalExpenditure",
    ],
    "Balance Sheet Growth": [
        "receivablesGrowth",
        "inventoryGrowth",
        "assetGrowth",
        "debtGrowth",
        "bookValueperShareGrowth",
    ],
    "Expense Growth": [
        "rdexpenseGrowth",
        "sgaexpensesGrowth",
    ],
    "Dividend Growth": [
        "dividendsPerShareGrowth",
    ],
    "Long-Term Revenue & Cash Flow Growth (Per Share)": [
        "tenYRevenueGrowthPerShare",
        "fiveYRevenueGrowthPerShare",
        "threeYRevenueGrowthPerShare",
        "tenYOperatingCFGrowthPerShare",
        "fiveYOperatingCFGrowthPerShare",
        "threeYOperatingCFGrowthPerShare",
    ],
    "Long-Term Net Income Growth (Per Share)": [
        "tenYNetIncomeGrowthPerShare",
        "fiveYNetIncomeGrowthPerShare",
        "threeYNetIncomeGrowthPerShare",
        "tenYBottomLineNetIncomeGrowthPerShare",
        "fiveYBottomLineNetIncomeGrowthPerShare",
        "threeYBottomLineNetIncomeGrowthPerShare",
    ],
    "Long-Term Equity & Dividend Growth (Per Share)": [
        "tenYShareholdersEquityGrowthPerShare",
        "fiveYShareholdersEquityGrowthPerShare",
        "threeYShareholdersEquityGrowthPerShare",
        "tenYDividendperShareGrowthPerShare",
        "fiveYDividendperShareGrowthPerShare",
        "threeYDividendperShareGrowthPerShare",
    ],
}

KEY_METRICS_COLUMN_GROUPS: dict[str, list[str]] = {
    "Valuation Metrics": [
        "marketCap",
        "enterpriseValue",
        "evToSales",
        "evToOperatingCashFlow",
        "evToFreeCashFlow",
        "evToEBITDA",
        "netDebtToEBITDA",
        "grahamNumber",
        "grahamNetNet",
        "earningsYield",
        "freeCashFlowYield",
        "tangibleAssetValue",
        "netCurrentAssetValue",
    ],
    "Profitability & Returns": [
        "incomeQuality",
        "taxBurden",
        "interestBurden",
        "returnOnAssets",
        "operatingReturnOnAssets",
        "returnOnTangibleAssets",
        "returnOnEquity",
        "returnOnInvestedCapital",
        "returnOnCapitalEmployed",
    ],
    "Liquidity & Working Capital": [
        "currentRatio",
        "workingCapital",
        "investedCapital",
    ],
    "Capital Expenditure Metrics": [
        "capexToOperatingCashFlow",
        "capexToDepreciation",
        "capexToRevenue",
    ],
    "Expense Ratios": [
        "salesGeneralAndAdministrativeToRevenue",
        "researchAndDevelopementToRevenue",
        "stockBasedCompensationToRevenue",
        "intangiblesToTotalAssets",
    ],
    "Efficiency & Cash Conversion Cycle": [
        "averageReceivables",
        "averagePayables",
        "averageInventory",
        "daysOfSalesOutstanding",
        "daysOfPayablesOutstanding",
        "daysOfInventoryOutstanding",
        "operatingCycle",
        "cashConversionCycle",
    ],
    "Free Cash Flow Metrics": [
        "freeCashFlowToEquity",
        "freeCashFlowToFirm",
    ],
}

RATIOS_COLUMN_GROUPS: dict[str, list[str]] = {
    "Profitability Margins": [
        "grossProfitMargin",
        "ebitMargin",
        "ebitdaMargin",
        "operatingProfitMargin",
        "pretaxProfitMargin",
        "continuousOperationsProfitMargin",
        "netProfitMargin",
        "bottomLineProfitMargin",
    ],
    "Efficiency & Turnover Ratios": [
        "receivablesTurnover",
        "payablesTurnover",
        "inventoryTurnover",
        "fixedAssetTurnover",
        "assetTurnover",
        "workingCapitalTurnoverRatio",
    ],
    "Liquidity Ratios": [
        "currentRatio",
        "quickRatio",
        "solvencyRatio",
        "cashRatio",
    ],
    "Valuation Ratios": [
        "priceToEarningsRatio",
        "priceToEarningsGrowthRatio",
        "forwardPriceToEarningsGrowthRatio",
        "priceToBookRatio",
        "priceToSalesRatio",
        "priceToFreeCashFlowRatio",
        "priceToOperatingCashFlowRatio",
        "priceToFairValue",
        "enterpriseValueMultiple",
        "debtToMarketCap",
    ],
    "Leverage & Solvency Ratios": [
        "debtToAssetsRatio",
        "debtToEquityRatio",
        "debtToCapitalRatio",
        "longTermDebtToCapitalRatio",
        "financialLeverageRatio",
    ],
    "Cash Flow Coverage Ratios": [
        "operatingCashFlowRatio",
        "operatingCashFlowSalesRatio",
        "freeCashFlowOperatingCashFlowRatio",
        "debtServiceCoverageRatio",
        "interestCoverageRatio",
        "shortTermOperatingCashFlowCoverageRatio",
        "operatingCashFlowCoverageRatio",
        "capitalExpenditureCoverageRatio",
        "dividendPaidAndCapexCoverageRatio",
    ],
    "Dividend Ratios": [
        "dividendPayoutRatio",
        "dividendYield",
        "dividendYieldPercentage",
        "dividendPerShare",
    ],
    "Per Share Metrics": [
        "revenuePerShare",
        "netIncomePerShare",
        "interestDebtPerShare",
        "cashPerShare",
        "bookValuePerShare",
        "tangibleBookValuePerShare",
        "shareholdersEquityPerShare",
        "operatingCashFlowPerShare",
        "capexPerShare",
        "freeCashFlowPerShare",
    ],
    "Tax & Other Ratios": [
        "netIncomePerEBT",
        "ebtPerEbit",
        "effectiveTaxRate",
    ],
}

# Some columns don't split cleanly with the generic camelCase regex
# (e.g. "ebitgrowth" has no internal capital, and the tenY/fiveY/threeY
# per-share growth columns mix a leading acronym with camelCase in a way
# the regex mangles). Give these an explicit, readable label instead.
COLUMN_LABEL_OVERRIDES: dict[str, str] = {
    "ebitgrowth": "EBIT Growth",
    "epsgrowth": "EPS Growth",
    "epsdilutedGrowth": "EPS Diluted Growth",
    "rdexpenseGrowth": "R&D Expense Growth",
    "sgaexpensesGrowth": "SG&A Expenses Growth",
    "bookValueperShareGrowth": "Book Value Per Share Growth",
    "evToSales": "EV To Sales",
    "evToOperatingCashFlow": "EV To Operating Cash Flow",
    "evToFreeCashFlow": "EV To Free Cash Flow",
    "evToEBITDA": "EV To EBITDA",
    "netDebtToEBITDA": "Net Debt To EBITDA",
    "tenYRevenueGrowthPerShare": "10Y Revenue Growth Per Share",
    "fiveYRevenueGrowthPerShare": "5Y Revenue Growth Per Share",
    "threeYRevenueGrowthPerShare": "3Y Revenue Growth Per Share",
    "tenYOperatingCFGrowthPerShare": "10Y Operating Cash Flow Growth Per Share",
    "fiveYOperatingCFGrowthPerShare": "5Y Operating Cash Flow Growth Per Share",
    "threeYOperatingCFGrowthPerShare": "3Y Operating Cash Flow Growth Per Share",
    "tenYNetIncomeGrowthPerShare": "10Y Net Income Growth Per Share",
    "fiveYNetIncomeGrowthPerShare": "5Y Net Income Growth Per Share",
    "threeYNetIncomeGrowthPerShare": "3Y Net Income Growth Per Share",
    "tenYShareholdersEquityGrowthPerShare": "10Y Shareholders Equity Growth Per Share",
    "fiveYShareholdersEquityGrowthPerShare": "5Y Shareholders Equity Growth Per Share",
    "threeYShareholdersEquityGrowthPerShare": "3Y Shareholders Equity Growth Per Share",
    "tenYDividendperShareGrowthPerShare": "10Y Dividend Per Share Growth Per Share",
    "fiveYDividendperShareGrowthPerShare": "5Y Dividend Per Share Growth Per Share",
    "threeYDividendperShareGrowthPerShare": "3Y Dividend Per Share Growth Per Share",
    "tenYBottomLineNetIncomeGrowthPerShare": "10Y Bottom Line Net Income Growth Per Share",
    "fiveYBottomLineNetIncomeGrowthPerShare": "5Y Bottom Line Net Income Growth Per Share",
    "threeYBottomLineNetIncomeGrowthPerShare": "3Y Bottom Line Net Income Growth Per Share",
    "netIncomePerEBT": "Net Income Per EBT",
    "ebtPerEbit": "EBT Per EBIT",
    "researchAndDevelopementToRevenue": "Research And Development To Revenue",
}

# Columns that should be printed as plain integers (no thousands
# separators), even though format_financial_value would otherwise add
# commas to any numeric value.
PLAIN_INTEGER_COLUMNS = {"fiscalYear"}

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

# Statement-type registry. "key" is used for output folder names, output
# filenames, and log messages; "file_pattern" selects which file(s) in the
# shared input folder belong to this statement type; "column_groups"
# selects which grouping dict above to use.
STATEMENT_TYPE_CONFIGS: dict[str, dict[str, Any]] = {
    "growth": {
        "input_folder": BASE_DATA_FOLDER,
        "file_pattern": "*growth*.xlsx",
        "output_folder": OUTPUT_BASE_FOLDER / "growth",
        "combined_output_filename": "growth_all_chunks.txt",
        "column_groups": GROWTH_COLUMN_GROUPS,
    },
    "key_metrics": {
        "input_folder": BASE_DATA_FOLDER,
        "file_pattern": "*key_metrics*.xlsx",
        "output_folder": OUTPUT_BASE_FOLDER / "key_metrics",
        "combined_output_filename": "key_metrics_all_chunks.txt",
        "column_groups": KEY_METRICS_COLUMN_GROUPS,
    },
    "ratios": {
        "input_folder": BASE_DATA_FOLDER,
        "file_pattern": "*ratios*.xlsx",
        "output_folder": OUTPUT_BASE_FOLDER / "ratios",
        "combined_output_filename": "ratios_all_chunks.txt",
        "column_groups": RATIOS_COLUMN_GROUPS,
    },
}


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
    key = str(column_name).strip()

    if key in COLUMN_LABEL_OVERRIDES:
        return COLUMN_LABEL_OVERRIDES[key]

    text = key
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.title()


def format_financial_value(value: Any, column_name: str | None = None) -> str:
    if is_empty(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, (int, float)):
        number = float(value)
        if column_name in PLAIN_INTEGER_COLUMNS and number.is_integer():
            return str(int(number))
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


def detect_statement_period(excel_file: Path, row: pd.Series) -> str:
    """Infer annual, quarterly, or ttm from the row's `period` value first
    (e.g. "FY", "Q1", "TTM"), falling back to the filename if the column
    is missing or empty."""

    period_value = row.get("period")
    if not is_empty(period_value):
        period_text = str(period_value).strip().upper()
        if period_text == "FY":
            return "annual"
        if period_text.startswith("Q"):
            return "quarterly"
        if period_text == "TTM":
            return "ttm"

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
    statement_period: str,
) -> str:
    metadata_lines = [
        f"Source File: {source_file}",
        f"Statement Type: {statement_type}",
        f"Statement Period: {statement_period}",
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
        metadata_lines.append(f"{label}: {format_financial_value(value, column)}")

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
    statement_period: str,
) -> str | None:
    symbol = format_financial_value(row.get("symbol", "Unknown"), "symbol")
    fiscal_year = format_financial_value(row.get("fiscalYear", "Unknown"), "fiscalYear")
    period = format_financial_value(row.get("period", "Unknown"), "period")
    report_date = format_financial_value(row.get("date", "Unknown"), "date")

    # Include statement type/period and report date to prevent duplicate
    # IDs across annual, quarterly, and TTM files.
    chunk_id = "_".join(
        [
            normalize_identifier(symbol),
            normalize_identifier(statement_type),
            normalize_identifier(statement_period),
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
        formatted_value = format_financial_value(value, column)
        metric_lines.append(f"- {label}: {formatted_value}")

    if not metric_lines:
        return None

    chunk_lines = [
        "=" * 80,
        f"CHUNK ID: {chunk_id}",
        f"SOURCE FILE: {source_file}",
        f"SOURCE ROW: {excel_row_number}",
        f"STATEMENT TYPE: {statement_type}",
        f"STATEMENT PERIOD: {statement_period}",
        f"FINANCIAL SECTION: {section_name}",
        "=" * 80,
        "",
        create_metadata_text(
            row=row,
            source_file=source_file,
            statement_type=statement_type,
            statement_period=statement_period,
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
    column_groups: dict[str, list[str]],
    excel_file: Path,
) -> list[str]:
    chunks: list[str] = []

    for dataframe_index, row in dataframe.iterrows():
        excel_row_number = dataframe_index + 2
        statement_period = detect_statement_period(excel_file, row)

        for section_name, columns in column_groups.items():
            chunk = create_financial_section_chunk(
                row=row,
                section_name=section_name,
                section_columns=columns,
                excel_row_number=excel_row_number,
                source_file=source_file,
                statement_type=statement_type,
                statement_period=statement_period,
            )

            if chunk:
                chunks.append(chunk)

    return chunks


# =========================================================
# FILE PROCESSING
# =========================================================

def read_and_chunk_excel_file(
    excel_file: Path,
    statement_type: str,
    column_groups: dict[str, list[str]],
) -> list[str]:
    dataframe = pd.read_excel(excel_file, engine="openpyxl")

    if dataframe.empty:
        print(f"Skipped empty file: {excel_file.name}")
        return []

    dataframe.columns = dataframe.columns.astype(str).str.strip()

    return create_chunks_for_dataframe(
        dataframe=dataframe,
        source_file=excel_file.name,
        statement_type=statement_type,
        column_groups=column_groups,
        excel_file=excel_file,
    )


def save_chunks_to_text(chunks: list[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for chunk_number, chunk in enumerate(chunks, start=1):
            file.write(f"CHUNK NUMBER: {chunk_number}\n")
            file.write(chunk)
            file.write("\n\n")


def process_statement_type(statement_type: str, config: dict[str, Any]) -> None:
    """Read every Excel file for one statement type and write one
    combined chunk file for it."""

    input_folder: Path = config["input_folder"]
    file_pattern: str = config["file_pattern"]
    output_folder: Path = config["output_folder"]
    combined_output_file = output_folder / config["combined_output_filename"]
    column_groups: dict[str, list[str]] = config["column_groups"]

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist for '{statement_type}':\n{input_folder}"
        )

    if not input_folder.is_dir():
        raise NotADirectoryError(
            f"The configured input path for '{statement_type}' is not a folder:\n"
            f"{input_folder}"
        )

    excel_files = sorted(
        file
        for file in input_folder.glob(file_pattern)
        if not file.name.startswith("~$")
    )

    if not excel_files:
        raise FileNotFoundError(
            f"No .xlsx files matching '{file_pattern}' were found for "
            f"'{statement_type}' in:\n{input_folder}"
        )

    all_chunks: list[str] = []
    total_rows = 0
    processed_files = 0
    failed_files = 0

    print("-" * 80)
    print(f"Statement type    : {statement_type}")
    print(f"Input folder      : {input_folder}")
    print(f"Excel files found : {len(excel_files)}")
    print("-" * 80)

    for excel_file in excel_files:
        try:
            print(f"Processing: {excel_file.name}")

            chunks = read_and_chunk_excel_file(
                excel_file,
                statement_type=statement_type,
                column_groups=column_groups,
            )

            dataframe = pd.read_excel(excel_file, engine="openpyxl")
            row_count = len(dataframe)

            all_chunks.extend(chunks)
            total_rows += row_count
            processed_files += 1

            print(f"  Rows: {row_count} | Chunks: {len(chunks)}")

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
            f"No chunks were generated for '{statement_type}'. "
            "Verify the Excel files and configured column names."
        )

    save_chunks_to_text(
        chunks=all_chunks,
        output_file=combined_output_file,
    )

    print("-" * 80)
    print(f"Chunking completed successfully for '{statement_type}'.")
    print(f"Files processed : {processed_files}")
    print(f"Files failed    : {failed_files}")
    print(f"Rows processed  : {total_rows}")
    print(f"Chunks created  : {len(all_chunks)}")
    print(f"Output file     : {combined_output_file}")
    print("-" * 80)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    """Process growth, key_metrics, and ratios Excel files, writing one
    combined chunk file per statement type."""

    for statement_type, config in STATEMENT_TYPE_CONFIGS.items():
        process_statement_type(statement_type, config)


if __name__ == "__main__":
    main()
