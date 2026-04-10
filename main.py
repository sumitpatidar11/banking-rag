# main.py

# ========================= PIPELINE =========================
import shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from mistralai.client import MistralClient

# Load CSV
df = pd.read_csv("banking_dataset.csv", encoding="utf-8")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["Query", "Response"])
df["Query"]    = df["Query"].str.strip()
df["Response"] = df["Response"].str.strip()
print(f"Loaded {len(df)} Q&A pairs")

# Build Documents
documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row["Query"],
        metadata={"answer": row["Response"]}
    )
    documents.append(doc)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ChromaDB
shutil.rmtree("./chroma_db", ignore_errors=True)
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"VectorDB ready — {vectordb._collection.count()} documents indexed")


# ========================= CHAT LOOP =========================
client = MistralClient(api_key="8FNCdcoivawgRhA33mVgLz8RfxY7ytVn")

print("\n" + "="*50)
print("   XYZ Bank — Virtual Assistant (Alice)")
print("   Type 'exit' or 'bye' to end the chat")
print("="*50 + "\n")

while True:

    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit", "bye", "goodbye"}:
        print("Alice: Thank you for banking with XYZ Bank. Have a great day!")
        break

    results = vectordb.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.metadata["answer"] for doc in results])

    prompt = f"""You are Alice, a professional banking assistant at XYZ Bank.
Answer the customer's question using only the context below.
Keep your answer short and clear.
If answer is not in context say: "I'm sorry, I don't have that information. Please contact our customer care at 1800-XXX-XXXX."

Context:
{context}

Customer Question: {user_input}

Answer:"""

    response = client.chat(
        model="mistral-small-latest",
        temperature=0.2,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content.strip()
    print(f"Alice: {reply}\n")