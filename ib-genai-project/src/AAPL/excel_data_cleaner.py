r"""
excel_data_cleaner.py
============================
Cleans the Excel files produced by json_to_excel_converter.py:
- Strips HTML tags, line breaks, tabs, and extra whitespace from text
- Drops fully-empty rows/columns and duplicate rows
- Saves each cleaned file into a "cleaned" subfolder next to the input

All settings (paths, companies, statement types, cleaning options,
etc.) live in config.txt, which by default is read from:
    C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\config\config.txt
Nothing is hardcoded here. To onboard a new company or statement type,
just add a line to config.txt; you never need to touch this file.

Usage:
    python excel_data_cleaner.py
    python excel_data_cleaner.py --config path\to\other_config.txt
"""

from __future__ import annotations

import os
import re
import glob
import argparse
import configparser
from pathlib import Path

import pandas as pd


# -----------------------------
# CONFIG LOADING
# -----------------------------

def load_config(config_path: str) -> dict:
    """Read config.txt and return a plain dict of settings ready to use."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path)

    settings = {
        "base_excel_dir": parser.get("paths", "base_excel_dir"),
        "excel_engine": parser.get("settings", "excel_engine", fallback="openpyxl"),
        "output_subfolder": parser.get("cleaning", "output_subfolder", fallback="cleaned"),
        "excel_glob_pattern": parser.get("cleaning", "excel_glob_pattern", fallback="*.xlsx"),
        "cleaned_suffix": parser.get("cleaning", "cleaned_suffix", fallback="_clean"),
    }

    # Build a flat list of (company, statement_type) jobs from [companies]
    jobs = []
    if parser.has_section("companies"):
        for company, statement_list in parser.items("companies"):
            statement_types = [s.strip() for s in statement_list.split(",") if s.strip()]
            for statement_type in statement_types:
                jobs.append((company.upper(), statement_type))
    settings["jobs"] = jobs

    return settings


# -----------------------------
# CLEANING FUNCTIONS
# -----------------------------

def clean_text(value):
    """
    Clean a text value.

    Operations:
    - Preserve missing values
    - Remove HTML tags
    - Replace line breaks and tabs with spaces
    - Collapse multiple spaces
    - Remove leading and trailing spaces
    """

    if pd.isna(value):
        return value

    text = str(value)

    # Remove HTML tags such as <p>, <br>, <div>.
    text = re.sub(r"<[^>]+>", "", text)

    # Replace line breaks and tabs with a single space.
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Replace multiple spaces with one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleanup operations to an Excel DataFrame."""

    # Clean column names.
    df.columns = [clean_text(column) for column in df.columns]

    # Clean only text/object columns.
    text_columns = df.select_dtypes(include=["object", "string"]).columns

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    # Remove completely empty rows.
    df = df.dropna(how="all")

    # Remove completely empty columns.
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows.
    df = df.drop_duplicates()

    # Reset row numbers after cleanup.
    df = df.reset_index(drop=True)

    return df


# -----------------------------
# FILE PROCESSING
# -----------------------------

def clean_excel_file(
    input_file: Path,
    output_folder: Path,
    cleaned_suffix: str,
    excel_engine: str,
) -> dict:
    """Read, clean, and save one Excel file."""

    print("\n" + "=" * 80)
    print(f"Processing: {input_file.name}")

    # Read the first worksheet.
    original_df = pd.read_excel(input_file, engine=excel_engine)

    original_rows = len(original_df)
    original_columns = len(original_df.columns)

    cleaned_df = clean_dataframe(original_df.copy())

    cleaned_rows = len(cleaned_df)
    cleaned_columns = len(cleaned_df.columns)

    output_file = output_folder / f"{input_file.stem}{cleaned_suffix}.xlsx"

    cleaned_df.to_excel(output_file, index=False, engine=excel_engine)

    print(f"Original rows: {original_rows}")
    print(f"Cleaned rows:  {cleaned_rows}")
    print(f"Original columns: {original_columns}")
    print(f"Cleaned columns:  {cleaned_columns}")
    print(f"Rows removed: {original_rows - cleaned_rows}")
    print(f"Saved to: {output_file}")

    return {
        "input_file": input_file.name,
        "output_file": output_file.name,
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "rows_removed": original_rows - cleaned_rows,
        "original_columns": original_columns,
        "cleaned_columns": cleaned_columns,
        "status": "Success",
    }


def process_excel_folder(
    input_folder: Path,
    output_subfolder: str,
    excel_glob_pattern: str,
    cleaned_suffix: str,
    excel_engine: str,
) -> None:
    """Clean every Excel file in one company/statement_type folder."""

    if not input_folder.exists():
        print(f"Input folder does not exist:\n{input_folder}")
        return

    if not input_folder.is_dir():
        print(f"The configured input path is not a folder:\n{input_folder}")
        return

    output_folder = input_folder / output_subfolder
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find Excel files only in the input folder.
    # This does not search the cleaned output subfolder.
    excel_files = sorted(
        file
        for file in input_folder.glob(excel_glob_pattern)
        if not file.name.startswith("~$")
        and not file.stem.endswith(cleaned_suffix)
    )

    if not excel_files:
        print(f"No Excel files were found in:\n{input_folder}")
        return

    print(f"Excel files found: {len(excel_files)}")
    print(f"Output folder: {output_folder}")

    successful_files = 0
    failed_files = 0

    for input_file in excel_files:
        try:
            clean_excel_file(
                input_file=input_file,
                output_folder=output_folder,
                cleaned_suffix=cleaned_suffix,
                excel_engine=excel_engine,
            )
            successful_files += 1

        except PermissionError as error:
            failed_files += 1
            print(
                f"Permission denied: {input_file.name}\n"
                f"Close the file in Excel and run the script again. ({error})"
            )

        except Exception as error:
            failed_files += 1
            print(f"Failed to process {input_file.name}: {error}")

    print("\n" + "-" * 40)
    print("Cleanup completed")
    print(f"Successful files: {successful_files}")
    print(f"Failed files: {failed_files}")
    print(f"Output folder: {output_folder}")
    print("-" * 40 + "\n")


# -----------------------------
# RUN THE PROGRAM
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean Excel financial data files")
    parser.add_argument(
        "--config",
        default=r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\config\config.txt",
        help="Path to config.txt (default: fixed config folder path)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not cfg["jobs"]:
        print("No [companies] entries found in config.txt - nothing to do.")
        return

    print(f"Loaded {len(cfg['jobs'])} company/statement job(s) from {args.config}\n")

    for company, statement_type in cfg["jobs"]:
        excel_folder = Path(cfg["base_excel_dir"]) / company / statement_type

        print("========================================")
        print(f"Company: {company} | Statement type: {statement_type}")
        print("========================================")

        process_excel_folder(
            input_folder=excel_folder,
            output_subfolder=cfg["output_subfolder"],
            excel_glob_pattern=cfg["excel_glob_pattern"],
            cleaned_suffix=cfg["cleaned_suffix"],
            excel_engine=cfg["excel_engine"],
        )


if __name__ == "__main__":
    main()
