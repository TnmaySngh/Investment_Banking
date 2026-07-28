from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FOLDER = Path(
    r"C:\AZ_DEVOPS_PYTHON\Investment_Banking"
    r"\ib-genai-project\data\excel\AAPL\balance_sheet"
)

OUTPUT_FOLDER = INPUT_FOLDER / "cleaned"

# =========================================================
# CLEANING FUNCTIONS
# =========================================================

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
    df.columns = [
        clean_text(column)
        for column in df.columns
    ]

    # Clean only text/object columns.
    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    # Remove completely empty rows.
    df = df.dropna(how="all")

    # Remove completely empty columns.
    df = df.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows.
    df = df.drop_duplicates()

    # Reset row numbers after cleanup.
    df = df.reset_index(drop=True)

    return df


# =========================================================
# FILE PROCESSING
# =========================================================

def clean_excel_file(
    input_file: Path,
    output_folder: Path
) -> dict:
    """Read, clean, and save one Excel file."""

    print("\n" + "=" * 80)
    print(f"Processing: {input_file.name}")

    # Read the first worksheet.
    original_df = pd.read_excel(
        input_file,
        engine="openpyxl"
    )

    original_rows = len(original_df)
    original_columns = len(original_df.columns)

    cleaned_df = clean_dataframe(
        original_df.copy()
    )

    cleaned_rows = len(cleaned_df)
    cleaned_columns = len(cleaned_df.columns)

    output_file = (
        output_folder
        / f"{input_file.stem}_clean.xlsx"
    )

    cleaned_df.to_excel(
        output_file,
        index=False,
        engine="openpyxl"
    )

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


def process_all_excel_files() -> None:
    """Clean every Excel file in the configured folder."""

    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            "Input folder does not exist:\n"
            f"{INPUT_FOLDER}"
        )

    if not INPUT_FOLDER.is_dir():
        raise NotADirectoryError(
            "The configured input path is not a folder:\n"
            f"{INPUT_FOLDER}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find XLSX files only in the input folder.
    # This does not search the cleaned output subfolder.
    excel_files = sorted(
        file
        for file in INPUT_FOLDER.glob("*.xlsx")
        if not file.name.startswith("~$")
        and not file.stem.endswith("_clean")
    )

    if not excel_files:
        print(
            "No .xlsx files were found in:\n"
            f"{INPUT_FOLDER}"
        )
        return

    print(f"Excel files found: {len(excel_files)}")
    print(f"Output folder: {OUTPUT_FOLDER}")

    processing_results = []

    for input_file in excel_files:
        try:
            result = clean_excel_file(
                input_file=input_file,
                output_folder=OUTPUT_FOLDER
            )

        except PermissionError as error:
            print(
                f"Permission denied: {input_file.name}\n"
                "Close the file in Excel and run the script again."
            )

            result = {
                "input_file": input_file.name,
                "output_file": "",
                "original_rows": "",
                "cleaned_rows": "",
                "rows_removed": "",
                "original_columns": "",
                "cleaned_columns": "",
                "status": f"Permission error: {error}",
            }

        except Exception as error:
            print(
                f"Failed to process {input_file.name}: "
                f"{error}"
            )

            result = {
                "input_file": input_file.name,
                "output_file": "",
                "original_rows": "",
                "cleaned_rows": "",
                "rows_removed": "",
                "original_columns": "",
                "cleaned_columns": "",
                "status": f"Failed: {error}",
            }

        processing_results.append(result)

    """ # Create a processing summary.
    summary_df = pd.DataFrame(
        processing_results
    )

    summary_file = (
        OUTPUT_FOLDER
        / "cleanup_summary.xlsx"
    )

    summary_df.to_excel(
        summary_file,
        index=False,
        engine="openpyxl"
    )

    successful_files = sum(
        result["status"] == "Success"
        for result in processing_results
    )

    print("\n" + "=" * 80)
    print("CLEANUP COMPLETED")
    print("=" * 80)
    print(f"Files found:      {len(excel_files)}")
    print(f"Files successful: {successful_files}")
    print(
        f"Files failed:     "
        f"{len(excel_files) - successful_files}"
    )
    print(f"Summary file:     {summary_file}") """


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    process_all_excel_files()