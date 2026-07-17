import chromadb

client = chromadb.PersistentClient(
    path=r"C:\Projects\project\Investment_Banking\ib-genai-project\data\vector_db\chroma_balance_sheet_annual_db"
)

client.delete_collection("balance_sheet")

print("Collection deleted")