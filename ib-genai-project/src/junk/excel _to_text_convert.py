import os
import glob
import pandas as pd

excel_folder = r"C:\Projects\project\Investment_Banking\ib-genai-project\data\excel"
text_folder = r"C:\Projects\project\Investment_Banking\ib-genai-project\data\text"

os.makedirs(text_folder, exist_ok=True)

for excel_file in glob.glob(os.path.join(excel_folder, "*.xlsx")):

    df = pd.read_excel(excel_file)

    base_name = os.path.splitext(os.path.basename(excel_file))[0]
    text_file = os.path.join(text_folder, f"{base_name}.txt")

    with open(text_file, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    print(f"Created: {text_file}")

print("All text files generated successfully.")