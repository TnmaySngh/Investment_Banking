import re
import pandas as pd


def clean_text(value):
    if pd.isna(value):
        return value

    text = str(value)

    text = re.sub(r"<[^>]+>", "", text)          # Remove HTML tags
    text = re.sub(r"[\r\n\t]+", " ", text)       # Remove line breaks/tabs
    text = re.sub(r"\s+", " ", text)             # Collapse spaces
    text = text.strip()

    return text

input_file = r"C:\Projects\project\Investment_Banking\ib-genai-project\data\excel\balance_sheet"
# output_file = r"C:\Projects\project\Investment_Banking\ib-genai-project\data\excel\balance_sheet_annual_clean.xlsx"
output_file = r"C:\Projects\project\Investment_Banking\ib-genai-project\data\excel\balance_sheet_clean"

# Read Excel
df = pd.read_excel(input_file, engine="openpyxl")

# Clean string columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(clean_text)

# Remove duplicates
df = df.drop_duplicates()

# Remove completely empty rows
df = df.dropna(how="all")

# Save cleaned data to Excel
df.to_excel(
    output_file,
    index=False,
    engine="openpyxl"
)

print(f"Cleaned Excel file saved successfully:\n{output_file}")