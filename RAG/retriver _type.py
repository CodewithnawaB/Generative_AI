from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.retrievers.multi_query import (
    MultiQueryRetriever
)

embeddings = OllamaEmbeddings(model="llama3")
model      = ChatOllama(model="llama3")

docs = [
    "Python is easy programming language",
    "Python used in AI and data science",
    "LangChain framework for AI apps",
    "RAG helps AI search documents",
]

vectorstore = FAISS.from_texts(docs, embeddings)

# ── Type 1: Simple Retriever ──────────
simple_retriever = vectorstore.as_retriever(
    search_kwargs = {"k": 2}
)
results = simple_retriever.invoke("Python")
print("Simple Retriever:")
for r in results:
    print(f"├── {r.page_content}")

# ── Type 2: MultiQuery Retriever ──────
multi_retriever = MultiQueryRetriever.from_llm(
    retriever = simple_retriever,
    llm       = model
)
results2 = multi_retriever.invoke("Python AI")
print("\nMultiQuery Retriever:")
for r in results2:
    print(f"├── {r.page_content}")