import chromadb

BALANCE_DB = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\vector_db\AAPL\chroma_balance_sheet_db"
CASHFLOW_DB = r"C:\AZ_DEVOPS_PYTHON\Investment_Banking\ib-genai-project\data\vector_db\AAPL\chroma_cash_flow_db"

balance_client = chromadb.PersistentClient(path=BALANCE_DB)
cashflow_client = chromadb.PersistentClient(path=CASHFLOW_DB)

print(balance_client.list_collections())
print(cashflow_client.list_collections())