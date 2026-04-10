from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from mistralai.client import MistralClient

# =========================
# Load ChromaDB
# =========================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print(f"Loaded VectorDB — {vectordb._collection.count()} documents")


# =========================
# Mistral Setup
# =========================
client = MistralClient(api_key="8FNCdcoivawgRhA33mVgLz8RfxY7ytVn")

# =========================
# Chat Loop
# =========================
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

    # Search similar questions from dataset
    results = vectordb.similarity_search(user_input, k=3)

    # Build context from matched answers
    context = "\n\n".join([doc.metadata["answer"] for doc in results])

    # Build prompt
    prompt = f"""You are Alice, a professional banking assistant at XYZ Bank.
Answer the customer's question using only the context below.
Keep your answer short and clear.
If answer is not in context say: "I'm sorry, I don't have that information. Please contact our customer care at 1800-XXX-XXXX."

Context:
{context}

Customer Question: {user_input}

Answer:"""

    # Call Mistral
    response = client.chat(
        model="mistral-small-latest",
        temperature=0.2,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content.strip()
    print(f"Alice: {reply}\n")