from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FOLDER = Path(
    r"C:\Projects\project\Investment_Banking"
    r"\ib-genai-project\data\excel\balance_sheet\cleaned"
)

OUTPUT_FOLDER = Path(
    r"C:\Projects\project\Investment_Banking"
    r"\ib-genai-project\data\excel\balance_sheet\cleaned\text"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# EXCEL TO TEXT
# =========================================================

def dataframe_row_to_text(
    row: pd.Series,
    row_number: int
) -> str:

    lines = []

    lines.append("=" * 80)
    lines.append(f"RECORD NUMBER: {row_number}")
    lines.append("=" * 80)

    for column_name, value in row.items():

        if pd.isna(value):
            continue

        lines.append(
            f"{column_name}: {value}"
        )

    return "\n".join(lines)


def excel_to_text(
    excel_file: Path
):

    print(
        f"\nProcessing: "
        f"{excel_file.name}"
    )

    df = pd.read_excel(
        excel_file,
        engine="openpyxl"
    )

    text_blocks = []

    for index, row in df.iterrows():

        block = dataframe_row_to_text(
            row=row,
            row_number=index + 1
        )

        text_blocks.append(block)

    output_file = (
        OUTPUT_FOLDER
        / f"{excel_file.stem}.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n\n".join(text_blocks)
        )

    print(
        f"Created: "
        f"{output_file}"
    )

    print(
        f"Rows exported: "
        f"{len(df)}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    excel_files = sorted(
        INPUT_FOLDER.glob("*.xlsx")
    )

    if not excel_files:
        print(
            "No Excel files found."
        )
        return

    print(
        f"Files found: "
        f"{len(excel_files)}"
    )

    for excel_file in excel_files:

        try:

            excel_to_text(
                excel_file
            )

        except Exception as error:

            print(
                f"Failed: "
                f"{excel_file.name}"
            )

            print(error)

    print(
        "\nAll text files generated successfully."
    )


if __name__ == "__main__":
    main()