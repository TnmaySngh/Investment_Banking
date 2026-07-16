import json
import pandas as pd

json_path = r"C:\Projects\project\Investment_Banking\ib-genai-project\data\json\balance_sheet_annual.json"
excel_path = r"C:\Projects\project\Investment_Banking\ib-genai-project\data\excel\balance_sheet_annual.xlsx"

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)

df.to_excel(excel_path, index=False)

print(f"Excel file created successfully: {excel_path}")
