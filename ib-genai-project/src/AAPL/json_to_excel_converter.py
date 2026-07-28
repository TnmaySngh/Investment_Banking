r"""
json_to_excel_converter.py
============================
Converts JSON files into Excel files, expanding a nested dict column
(e.g. "records") into separate columns.

All settings (paths, companies, statement types, target column name,
etc.) live in config.txt, which by default is read from:
    C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\config\config.txt
Nothing is hardcoded here. To onboard a new company or statement type,
just add a line to config.txt; you never need to touch this file.

Usage:
    python json_to_excel_converter.py
    python json_to_excel_converter.py --config path\to\other_config.txt
"""

import os
import glob
import argparse
import configparser
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
        "base_json_dir": parser.get("paths", "base_json_dir"),
        "base_excel_dir": parser.get("paths", "base_excel_dir"),
        "target_column_name": parser.get("settings", "target_column_name", fallback="records"),
        "save_raw": parser.getboolean("settings", "save_raw", fallback=False),
        "rsuffix": parser.get("settings", "rsuffix", fallback="_nested"),
        "excel_engine": parser.get("settings", "excel_engine", fallback="openpyxl"),
        "json_glob_pattern": parser.get("settings", "json_glob_pattern", fallback="*.json"),
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
# CONVERSION FUNCTION
# -----------------------------

def convert_and_split_json(
    json_folder: str,
    excel_folder: str,
    target_column_name: str,
    save_raw: bool = False,
    rsuffix: str = "_nested",
    excel_engine: str = "openpyxl",
    json_glob_pattern: str = "*.json",
) -> None:
    """
    Convert every JSON file in json_folder into Excel file(s):

    1. filename_raw.xlsx (only if save_raw=True)
       Contains the original JSON data before expanding the target column.

    2. filename.xlsx
       Contains the processed data after expanding the target dictionary column.
    """

    # Validate input folder
    if not os.path.isdir(json_folder):
        print(f"JSON folder does not exist: {json_folder}")
        return

    # Create Excel output folder if it does not already exist
    os.makedirs(excel_folder, exist_ok=True)

    # Find all JSON files
    search_path = os.path.join(json_folder, json_glob_pattern)
    json_files = glob.glob(search_path)

    if not json_files:
        print(f"No JSON files found in: {json_folder}")
        return

    print(f"Found {len(json_files)} JSON file(s).")
    print(f"Excel output folder: {excel_folder}\n")

    successful_files = 0
    failed_files = 0

    for file_path in json_files:
        json_filename = os.path.basename(file_path)
        base_name = os.path.splitext(json_filename)[0]

        print(f"Processing: {json_filename}")

        try:
            # Read the JSON file
            df = pd.read_json(file_path)

            # ---------------------------------
            # Save the original/raw Excel file (optional, config-driven)
            # ---------------------------------
            if save_raw:
                raw_excel_path = os.path.join(excel_folder, f"{base_name}_raw.xlsx")
                df.to_excel(raw_excel_path, index=False, engine=excel_engine)
                print(f"  Raw file saved: {raw_excel_path}")

            # ---------------------------------
            # Expand the target column
            # ---------------------------------
            if target_column_name in df.columns:
                print(f"  Expanding column: {target_column_name}")

                # Convert dictionaries in the target column into columns
                new_columns = df[target_column_name].apply(
                    lambda value: pd.Series(value)
                    if isinstance(value, dict)
                    else pd.Series(dtype="object")
                )

                # Remove the original dictionary column
                df_without_target = df.drop(columns=[target_column_name])

                # Add the generated columns
                final_df = df_without_target.join(new_columns, rsuffix=rsuffix)

            else:
                print(
                    f"  Column '{target_column_name}' was not found. "
                    "The processed file will contain the original data."
                )
                final_df = df.copy()

            # ---------------------------------
            # Save the processed Excel file
            # ---------------------------------
            processed_excel_path = os.path.join(excel_folder, f"{base_name}.xlsx")
            final_df.to_excel(processed_excel_path, index=False, engine=excel_engine)

            print(f"  Processed file saved: {processed_excel_path}\n")

            successful_files += 1

        except ValueError as error:
            failed_files += 1
            print(f"  Invalid JSON structure: {error}\n")

        except PermissionError:
            failed_files += 1
            print(
                "  Permission error. Close the Excel file if it is currently "
                "open, and verify that you can write to the output folder.\n"
            )

        except Exception as error:
            failed_files += 1
            print(f"  Failed to process the file: {error}\n")

    # -----------------------------
    # SUMMARY
    # -----------------------------
    print("----------------------------------------")
    print("Conversion completed")
    print(f"Successful JSON files: {successful_files}")
    print(f"Failed JSON files: {failed_files}")
    print(f"Output folder: {excel_folder}")
    print("----------------------------------------\n")


# -----------------------------
# RUN THE PROGRAM
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert JSON financial data files to Excel")
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
        json_folder = os.path.join(cfg["base_json_dir"], company, statement_type)
        excel_folder = os.path.join(cfg["base_excel_dir"], company, statement_type)

        print("========================================")
        print(f"Company: {company} | Statement type: {statement_type}")
        print("========================================")

        convert_and_split_json(
            json_folder=json_folder,
            excel_folder=excel_folder,
            target_column_name=cfg["target_column_name"],
            save_raw=cfg["save_raw"],
            rsuffix=cfg["rsuffix"],
            excel_engine=cfg["excel_engine"],
            json_glob_pattern=cfg["json_glob_pattern"],
        )


if __name__ == "__main__":
    main()
