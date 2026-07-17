import os
import glob
import pandas as pd


# -----------------------------
# CONFIGURATION
# -----------------------------

# Folder containing JSON files
json_folder = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\json\AAPL\cash_flow"

# Folder where Excel files will be created
excel_folder = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\excel\AAPL\cash_flow"

# Dictionary column to expand
target_column_name = "records"

# -----------------------------
# CONVERSION FUNCTION
# -----------------------------

def convert_and_split_json(
    json_folder: str,
    excel_folder: str,
    target_column_name: str
) -> None:
    """
    Convert every JSON file in json_folder into two Excel files:

    1. filename_raw.xlsx
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
    search_path = os.path.join(json_folder, "*.json")
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
            # Save the original/raw Excel file
            # ---------------------------------
            """ raw_excel_path = os.path.join(
                excel_folder,
                f"{base_name}_raw.xlsx"
            )

            df.to_excel(
                raw_excel_path,
                index=False,
                engine="openpyxl"
            )

            print(f"  Raw file saved: {raw_excel_path}") """

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
                df_without_target = df.drop(
                    columns=[target_column_name]
                )

                # Add the generated columns
                final_df = df_without_target.join(
                    new_columns,
                    rsuffix="_nested"
                )

            else:
                print(
                    f"  Column '{target_column_name}' was not found. "
                    "The processed file will contain the original data."
                )

                final_df = df.copy()

            # ---------------------------------
            # Save the processed Excel file
            # ---------------------------------
            processed_excel_path = os.path.join(
                excel_folder,
                f"{base_name}.xlsx"
            )

            final_df.to_excel(
                processed_excel_path,
                index=False,
                engine="openpyxl"
            )

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
    print("----------------------------------------")


# -----------------------------
# RUN THE PROGRAM
# -----------------------------

if __name__ == "__main__":
    convert_and_split_json(
        json_folder=json_folder,
        excel_folder=excel_folder,
        target_column_name=target_column_name
    )