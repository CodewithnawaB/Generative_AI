from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")

documents = [
    "Python is programming language",
    "AI is artificial intelligence",
    "LangChain builds AI applications",
    "RAG searches documents for answers",
    "Agents use tools to solve problems",
]

# ── FAISS Vector Store ────────────────
print("Creating FAISS Vector Store...")
faiss_store = FAISS.from_texts(
    documents,
    embeddings
)

# Search similar documents
results = faiss_store.similarity_search(
    "What is Python?",
    k = 2
)
print("\nFAISS Search Results:")
for doc in results:
    print(f"├── {doc.page_content}")

# Save and Load FAISS
faiss_store.save_local("my_faiss_store")
print("\n✅ FAISS saved!")

loaded_store = FAISS.load_local(
    "my_faiss_store",
    embeddings,
    allow_dangerous_deserialization=True
)
print("✅ FAISS loaded!")

# Search loaded store
results2 = loaded_store.similarity_search(
    "Tell me about agents",
    k = 2
)
print("\nLoaded Store Search:")
for doc in results2:
    print(f"├── {doc.page_content}")