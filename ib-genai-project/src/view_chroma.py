import chromadb
import pandas as pd

client = chromadb.PersistentClient(
    path=r"C:/Projects/project/Investment_Banking/ib-genai-project/data/vector_db/chroma_balance_sheet_annual_db"
)


collection = client.get_collection(
        name="balance_sheet"
    )

print("Documents:", collection.count())

data = collection.get(
    include=["documents", "metadatas"]
)

rows = []

for i in range(len(data["ids"])):
    rows.append({
        "id": data["ids"][i],
        "metadata": str(data["metadatas"][i]),
        "document": data["documents"][i][:200]
    })

df = pd.DataFrame(rows)

print(df.head(10))