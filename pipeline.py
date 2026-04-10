import shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================
# Load CSV
# =========================
df = pd.read_csv("banking_dataset.csv", encoding="utf-8")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Query", "Response"])
df["Query"]    = df["Query"].str.strip()
df["Response"] = df["Response"].str.strip()

print(f"Loaded {len(df)} Q&A pairs")


# =========================
# Build Documents
# =========================
documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row["Query"],
        metadata={"answer": row["Response"]}
    )
    documents.append(doc)


# =========================
# Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# =========================
# ChromaDB
# =========================
shutil.rmtree("./chroma_db", ignore_errors=True)

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"VectorDB ready — {vectordb._collection.count()} documents indexed")
print("Pipeline setup complete. Now run chat.py to start chatting!")
