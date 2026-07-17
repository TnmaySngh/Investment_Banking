import chromadb
import pandas as pd

client = chromadb.PersistentClient(
    path=r"C:/AZ_DEVOPS_PYTHON/Investment_Banking/ib-genai-project/data/vector_db/AAPL/chroma_balance_sheet_db"
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