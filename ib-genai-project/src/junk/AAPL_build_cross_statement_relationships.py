from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# =========================================================
# CONFIGURATION
# =========================================================
# Input: the six *_all_chunks.txt files already produced by the earlier
# chunking scripts (balance sheet, income statement, cash flow, growth,
# ratios, key metrics). This script does NOT re-read the original Excel
# files — it parses the chunk text that was already generated, joins
# records across statement types for the same company/period, and
# produces a NEW set of "relationship" chunks describing how the
# statements reconcile with each other.

INPUT_FOLDER = Path(r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\chunks\AAPL")

INPUT_FILES: dict[str, str] = {
    "balance_sheet": "balance_sheet/balance_sheet_all_chunks.txt",
    "income_statement": "income_statement/income_statement_all_chunks.txt",
    "cash_flow": "cash_flow/cash_flow_all_chunks.txt",
    "growth": "growth/growth_all_chunks.txt",
    "ratios": "ratios/ratios_all_chunks.txt",
    "key_metrics": "key_metrics/key_metrics_all_chunks.txt",
}

OUTPUT_FOLDER = Path(r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\chunks\AAPL\relationships")
OUTPUT_FILE = OUTPUT_FOLDER / "relationships_all_chunks.txt"

# Relative tolerance used to decide MATCH vs CLOSE vs MISMATCH when
# comparing a computed value to a reported one.
CLOSE_TOLERANCE = 0.02      # within 2% -> CLOSE (rounding-level difference)
MATCH_TOLERANCE = 0.001     # within 0.1% -> MATCH


# =========================================================
# PARSING — turn a *_all_chunks.txt file back into structured records
# =========================================================

@dataclass
class ChunkRecord:
    category: str                 # balance_sheet / income_statement / cash_flow / growth / ratios / key_metrics
    statement_type: str           # annual / quarterly / ttm / growth / ratios / key_metrics
    group: str                    # normalized join group: annual / quarterly / ttm
    section: str                  # e.g. "Current Assets", "Valuation Metrics"
    symbol: str
    fiscal_year: int
    period: str                   # FY / Q1 / Q2 / Q3 / Q4 / TTM
    report_date: str
    metrics: dict[str, float] = field(default_factory=dict)


def _to_number(raw: str) -> Optional[float]:
    raw = raw.strip()
    if not raw:
        return None
    cleaned = raw.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _derive_category(financial_statement: str, fallback_source_file: str) -> str:
    text = (financial_statement or fallback_source_file or "").lower()
    for suffix in ("_annual", "_quarterly", "_ttm"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    text = text.replace("_clean.xlsx", "").replace(".xlsx", "")
    return text.strip("_") or "unknown"


def _normalize_group(statement_type: str) -> str:
    if statement_type in ("annual", "growth", "ratios", "key_metrics"):
        return "annual"
    if statement_type in ("quarterly", "ttm"):
        return statement_type
    return "unknown"


def parse_chunk_file(path: Path) -> list[ChunkRecord]:
    if not path.exists():
        print(f"  (skipping, not found: {path})")
        return []

    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    blocks = re.split(r"CHUNK NUMBER: \d+\n", text)

    records: list[ChunkRecord] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        meta: dict[str, str] = {}
        metrics: dict[str, float] = {}
        section_name: Optional[str] = None
        in_metrics = False

        for line in block.split("\n"):
            line = line.rstrip()
            if not line:
                continue
            if set(line) == {"="}:
                continue

            if line.endswith("Metrics:"):
                section_name = line[: -len("Metrics:")].strip()
                in_metrics = True
                continue

            if in_metrics:
                m = re.match(r"^-\s*(.+?):\s*(.*)$", line)
                if m:
                    label, value = m.groups()
                    number = _to_number(value)
                    if number is not None:
                        metrics[label.strip()] = number
                continue

            m = re.match(r"^([A-Za-z][A-Za-z0-9 &/]*):\s*(.*)$", line)
            if m:
                key, value = m.groups()
                meta[key.strip()] = value.strip()

        if section_name is None or "Fiscal Year" not in meta or "Period" not in meta:
            continue

        fiscal_year_raw = meta.get("Fiscal Year", "").replace(",", "")
        try:
            fiscal_year = int(float(fiscal_year_raw))
        except ValueError:
            continue

        statement_type = meta.get("Statement Type", "unknown")
        category = _derive_category(
            meta.get("Financial Statement", ""),
            meta.get("Source File", ""),
        )

        records.append(
            ChunkRecord(
                category=category,
                statement_type=statement_type,
                group=_normalize_group(statement_type),
                section=section_name,
                symbol=meta.get("Company Symbol", "Unknown"),
                fiscal_year=fiscal_year,
                period=meta.get("Period", "Unknown"),
                report_date=meta.get("Report Date", "Unknown"),
                metrics=metrics,
            )
        )

    return records


# =========================================================
# JOINING — group parsed records by (symbol, fiscal_year, period, group)
# =========================================================

# PeriodData maps category -> section -> metric_label -> value
PeriodData = dict[str, dict[str, dict[str, float]]]
PeriodKey = tuple[str, int, str, str]  # symbol, fiscal_year, period, group


def build_period_index(all_records: list[ChunkRecord]) -> dict[PeriodKey, PeriodData]:
    index: dict[PeriodKey, PeriodData] = {}

    for record in all_records:
        key: PeriodKey = (record.symbol, record.fiscal_year, record.period, record.group)
        period_data = index.setdefault(key, {})
        category_data = period_data.setdefault(record.category, {})
        category_data[record.section] = record.metrics

    return index


def get_metric(
    period_data: PeriodData,
    category: str,
    section: str,
    metric: str,
) -> Optional[float]:
    return period_data.get(category, {}).get(section, {}).get(metric)


# =========================================================
# RELATIONSHIP RULES
# =========================================================
# Each rule pulls one or more raw metrics from the joined period data,
# computes a derived value, compares it to a reported value from a
# (possibly different) statement, and reports MATCH / CLOSE / MISMATCH.

@dataclass
class RelationshipResult:
    name: str
    formula_text: str
    computed_value: float
    reported_value: float
    reported_source: str
    verdict: str
    diff_pct: float


RuleFn = Callable[[PeriodData], Optional[RelationshipResult]]


def _classify(computed: float, reported: float) -> tuple[str, float]:
    if reported == 0:
        diff_pct = 0.0 if computed == 0 else float("inf")
    else:
        diff_pct = abs(computed - reported) / abs(reported)

    if diff_pct <= MATCH_TOLERANCE:
        verdict = "MATCH"
    elif diff_pct <= CLOSE_TOLERANCE:
        verdict = "CLOSE"
    else:
        verdict = "MISMATCH"
    return verdict, diff_pct


def rule_total_assets_from_current_noncurrent(pd_: PeriodData) -> Optional[RelationshipResult]:
    tca = get_metric(pd_, "balance_sheet", "Current Assets", "Total Current Assets")
    tnca = get_metric(pd_, "balance_sheet", "Non-Current Assets", "Total Non Current Assets")
    total_assets = get_metric(pd_, "balance_sheet", "Non-Current Assets", "Total Assets")
    if None in (tca, tnca, total_assets):
        return None

    computed = tca + tnca
    verdict, diff_pct = _classify(computed, total_assets)
    return RelationshipResult(
        name="Total Assets = Total Current Assets + Total Non-Current Assets",
        formula_text=f"{tca:,.0f} + {tnca:,.0f} = {computed:,.0f}",
        computed_value=computed,
        reported_value=total_assets,
        reported_source="balance_sheet / Non-Current Assets / Total Assets",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_assets_equal_liabilities_plus_equity(pd_: PeriodData) -> Optional[RelationshipResult]:
    total_assets = get_metric(pd_, "balance_sheet", "Non-Current Assets", "Total Assets")
    total_liabilities = get_metric(pd_, "balance_sheet", "Non-Current Liabilities", "Total Liabilities")
    total_equity = get_metric(pd_, "balance_sheet", "Shareholders Equity", "Total Equity")
    if None in (total_assets, total_liabilities, total_equity):
        return None

    computed = total_liabilities + total_equity
    verdict, diff_pct = _classify(computed, total_assets)
    return RelationshipResult(
        name="Total Assets = Total Liabilities + Total Equity",
        formula_text=f"{total_liabilities:,.0f} + {total_equity:,.0f} = {computed:,.0f}",
        computed_value=computed,
        reported_value=total_assets,
        reported_source="balance_sheet / Non-Current Assets / Total Assets",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_current_ratio(pd_: PeriodData) -> Optional[RelationshipResult]:
    tca = get_metric(pd_, "balance_sheet", "Current Assets", "Total Current Assets")
    tcl = get_metric(pd_, "balance_sheet", "Current Liabilities", "Total Current Liabilities")
    reported = get_metric(pd_, "ratios", "Liquidity Ratios", "Current Ratio")
    if None in (tca, tcl, reported) or tcl == 0:
        return None

    computed = tca / tcl
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Current Ratio = Total Current Assets / Total Current Liabilities",
        formula_text=f"{tca:,.0f} / {tcl:,.0f} = {computed:.4f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="ratios / Liquidity Ratios / Current Ratio",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_debt_to_equity(pd_: PeriodData) -> Optional[RelationshipResult]:
    total_liabilities = get_metric(pd_, "balance_sheet", "Non-Current Liabilities", "Total Liabilities")
    total_equity = get_metric(pd_, "balance_sheet", "Shareholders Equity", "Total Equity")
    reported = get_metric(pd_, "ratios", "Leverage & Solvency Ratios", "Debt To Equity Ratio")
    if None in (total_liabilities, total_equity, reported) or total_equity == 0:
        return None

    computed = total_liabilities / total_equity
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Debt-to-Equity = Total Liabilities / Total Equity",
        formula_text=f"{total_liabilities:,.0f} / {total_equity:,.0f} = {computed:.4f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="ratios / Leverage & Solvency Ratios / Debt To Equity Ratio",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_gross_profit_margin(pd_: PeriodData) -> Optional[RelationshipResult]:
    revenue = get_metric(pd_, "income_statement", "Revenue and Gross Profit", "Revenue")
    gross_profit = get_metric(pd_, "income_statement", "Revenue and Gross Profit", "Gross Profit")
    reported = get_metric(pd_, "ratios", "Profitability Margins", "Gross Profit Margin")
    if None in (revenue, gross_profit, reported) or revenue == 0:
        return None

    computed = gross_profit / revenue
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Gross Profit Margin = Gross Profit / Revenue",
        formula_text=f"{gross_profit:,.0f} / {revenue:,.0f} = {computed:.4f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="ratios / Profitability Margins / Gross Profit Margin",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_net_profit_margin(pd_: PeriodData) -> Optional[RelationshipResult]:
    revenue = get_metric(pd_, "income_statement", "Revenue and Gross Profit", "Revenue")
    net_income = get_metric(pd_, "income_statement", "Tax and Net Income", "Net Income")
    reported = get_metric(pd_, "ratios", "Profitability Margins", "Net Profit Margin")
    if None in (revenue, net_income, reported) or revenue == 0:
        return None

    computed = net_income / revenue
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Net Profit Margin = Net Income / Revenue",
        formula_text=f"{net_income:,.0f} / {revenue:,.0f} = {computed:.4f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="ratios / Profitability Margins / Net Profit Margin",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_return_on_equity(pd_: PeriodData) -> Optional[RelationshipResult]:
    net_income = get_metric(pd_, "income_statement", "Tax and Net Income", "Net Income")
    total_equity = get_metric(pd_, "balance_sheet", "Shareholders Equity", "Total Equity")
    reported = get_metric(pd_, "key_metrics", "Profitability & Returns", "Return On Equity")
    if None in (net_income, total_equity, reported) or total_equity == 0:
        return None

    computed = net_income / total_equity
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Return on Equity = Net Income / Total Equity",
        formula_text=f"{net_income:,.0f} / {total_equity:,.0f} = {computed:.4f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="key_metrics / Profitability & Returns / Return On Equity",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_enterprise_value(pd_: PeriodData) -> Optional[RelationshipResult]:
    market_cap = get_metric(pd_, "key_metrics", "Valuation Metrics", "Market Cap")
    net_debt = get_metric(pd_, "balance_sheet", "Debt and Investment Summary", "Net Debt")
    reported = get_metric(pd_, "key_metrics", "Valuation Metrics", "Enterprise Value")
    if None in (market_cap, net_debt, reported):
        return None

    computed = market_cap + net_debt
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Enterprise Value = Market Cap + Net Debt",
        formula_text=f"{market_cap:,.0f} + {net_debt:,.0f} = {computed:,.0f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="key_metrics / Valuation Metrics / Enterprise Value",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_net_debt_to_ebitda(pd_: PeriodData) -> Optional[RelationshipResult]:
    net_debt = get_metric(pd_, "balance_sheet", "Debt and Investment Summary", "Net Debt")
    ebitda = get_metric(pd_, "income_statement", "Operating Profitability", "Ebitda")
    reported = get_metric(pd_, "key_metrics", "Valuation Metrics", "Net Debt To EBITDA")
    if None in (net_debt, ebitda, reported) or ebitda == 0:
        return None

    computed = net_debt / ebitda
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Net Debt / EBITDA",
        formula_text=f"{net_debt:,.0f} / {ebitda:,.0f} = {computed:.4f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="key_metrics / Valuation Metrics / Net Debt To EBITDA",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_tangible_asset_value_vs_equity(pd_: PeriodData) -> Optional[RelationshipResult]:
    total_equity = get_metric(pd_, "balance_sheet", "Shareholders Equity", "Total Equity")
    goodwill = get_metric(pd_, "balance_sheet", "Non-Current Assets", "Goodwill")
    intangibles = get_metric(pd_, "balance_sheet", "Non-Current Assets", "Intangible Assets")
    reported = get_metric(pd_, "key_metrics", "Valuation Metrics", "Tangible Asset Value")
    if None in (total_equity, goodwill, intangibles, reported):
        return None

    computed = total_equity - goodwill - intangibles
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Tangible Asset Value = Total Equity - Goodwill - Intangible Assets",
        formula_text=f"{total_equity:,.0f} - {goodwill:,.0f} - {intangibles:,.0f} = {computed:,.0f}",
        computed_value=computed,
        reported_value=reported,
        reported_source="key_metrics / Valuation Metrics / Tangible Asset Value",
        verdict=verdict,
        diff_pct=diff_pct,
    )


def rule_revenue_growth_yoy(
    pd_: PeriodData,
    prior_year_index: dict[PeriodKey, PeriodData],
    key: PeriodKey,
) -> Optional[RelationshipResult]:
    symbol, fiscal_year, period, group = key
    prior_key = (symbol, fiscal_year - 1, period, group)
    prior_data = prior_year_index.get(prior_key)
    if prior_data is None:
        return None

    revenue_this_year = get_metric(pd_, "income_statement", "Revenue and Gross Profit", "Revenue")
    revenue_prior_year = get_metric(prior_data, "income_statement", "Revenue and Gross Profit", "Revenue")
    reported = get_metric(pd_, "growth", "Income Statement Growth", "Revenue Growth")
    if None in (revenue_this_year, revenue_prior_year, reported) or revenue_prior_year == 0:
        return None

    computed = (revenue_this_year - revenue_prior_year) / revenue_prior_year
    verdict, diff_pct = _classify(computed, reported)
    return RelationshipResult(
        name="Revenue Growth = (Revenue this year - Revenue prior year) / Revenue prior year",
        formula_text=(
            f"({revenue_this_year:,.0f} - {revenue_prior_year:,.0f}) / "
            f"{revenue_prior_year:,.0f} = {computed:.4f}"
        ),
        computed_value=computed,
        reported_value=reported,
        reported_source="growth / Income Statement Growth / Revenue Growth",
        verdict=verdict,
        diff_pct=diff_pct,
    )


# Rules that only need the current period's joined data.
SINGLE_PERIOD_RULES: list[RuleFn] = [
    rule_total_assets_from_current_noncurrent,
    rule_assets_equal_liabilities_plus_equity,
    rule_current_ratio,
    rule_debt_to_equity,
    rule_gross_profit_margin,
    rule_net_profit_margin,
    rule_return_on_equity,
    rule_enterprise_value,
    rule_net_debt_to_ebitda,
    rule_tangible_asset_value_vs_equity,
]


# =========================================================
# CHUNK GENERATION
# =========================================================

def normalize_identifier(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def format_result_chunk(
    symbol: str,
    fiscal_year: int,
    period: str,
    group: str,
    result: RelationshipResult,
) -> str:
    chunk_id = "_".join(
        [
            normalize_identifier(symbol),
            "relationship",
            group,
            str(fiscal_year),
            normalize_identifier(period),
            normalize_identifier(result.name),
        ]
    )

    lines = [
        "=" * 80,
        f"CHUNK ID: {chunk_id}",
        "RELATIONSHIP TYPE: Cross-Statement Consistency Check",
        f"COMPANY SYMBOL: {symbol}",
        f"FISCAL YEAR: {fiscal_year}",
        f"PERIOD: {period}",
        f"STATEMENT GROUP: {group}",
        "=" * 80,
        "",
        f"Relationship: {result.name}",
        f"Formula applied: {result.formula_text}",
        f"Reported value (source: {result.reported_source}): {result.reported_value:,.4f}",
        f"Computed value: {result.computed_value:,.4f}",
        f"Difference: {result.diff_pct * 100:.2f}%",
        f"Verdict: {result.verdict}",
    ]

    return "\n".join(lines).strip()


def build_relationship_chunks(period_index: dict[PeriodKey, PeriodData]) -> list[str]:
    chunks: list[str] = []

    for key in sorted(period_index.keys()):
        symbol, fiscal_year, period, group = key
        period_data = period_index[key]

        for rule in SINGLE_PERIOD_RULES:
            result = rule(period_data)
            if result is not None:
                chunks.append(
                    format_result_chunk(symbol, fiscal_year, period, group, result)
                )

        yoy_result = rule_revenue_growth_yoy(period_data, period_index, key)
        if yoy_result is not None:
            chunks.append(
                format_result_chunk(symbol, fiscal_year, period, group, yoy_result)
            )

    return chunks


def save_chunks(chunks: list[str], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for number, chunk in enumerate(chunks, start=1):
            file.write(f"CHUNK NUMBER: {number}\n")
            file.write(chunk)
            file.write("\n\n")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    all_records: list[ChunkRecord] = []

    print("-" * 80)
    print("Parsing per-statement chunk files")
    print("-" * 80)

    # Pre-flight check: verify every expected input file actually exists
    # before parsing anything. This turns a confusing downstream
    # "No chunk records were parsed" error into a precise list of exactly
    # which file(s) are missing and where the script looked for them.
    missing: list[Path] = []
    for category, filename in INPUT_FILES.items():
        path = INPUT_FOLDER / filename
        if not path.exists():
            missing.append(path)

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "The following input file(s) were not found:\n"
            f"{missing_list}\n\n"
            f"INPUT_FOLDER is currently set to:\n  {INPUT_FOLDER}\n\n"
            "Update INPUT_FOLDER (and/or INPUT_FILES) near the top of this "
            "script to point at the folder that actually contains your "
            "*_all_chunks.txt files."
        )

    for category, filename in INPUT_FILES.items():
        path = INPUT_FOLDER / filename
        print(f"Reading {category}: {path}")
        records = parse_chunk_file(path)
        print(f"  Parsed {len(records)} chunk records")
        all_records.extend(records)

        if not records:
            print(
                f"  WARNING: 0 records parsed from {path.name}. The file "
                "exists but its format didn't match what the parser "
                "expects (see parse_chunk_file). Check for unexpected "
                "line endings, encoding, or a changed chunk format."
            )

    if not all_records:
        raise ValueError(
            "No chunk records were parsed. All input files were found, "
            "but none of them produced parseable chunks — see the "
            "per-file WARNING messages above for which file(s) failed "
            "and check the chunk format."
        )

    period_index = build_period_index(all_records)
    print("-" * 80)
    print(f"Joined into {len(period_index)} (symbol, fiscal_year, period, group) periods")

    relationship_chunks = build_relationship_chunks(period_index)

    if not relationship_chunks:
        raise ValueError(
            "No relationship chunks were generated. Verify that the required "
            "categories/sections/metrics are present across the input files."
        )

    save_chunks(relationship_chunks, OUTPUT_FILE)

    print("-" * 80)
    print("Relationship chunking completed.")
    print(f"Relationship chunks created : {len(relationship_chunks)}")
    print(f"Output file                 : {OUTPUT_FILE}")
    print("-" * 80)


if __name__ == "__main__":
    import sys

    # Optional overrides so you don't have to edit the script every time:
    #   python AAPL_build_cross_statement_relationships.py <input_folder> [output_folder]
    if len(sys.argv) >= 2:
        INPUT_FOLDER = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        OUTPUT_FOLDER = Path(sys.argv[2])
        OUTPUT_FILE = OUTPUT_FOLDER / "relationships_all_chunks.txt"

    main()
