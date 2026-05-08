from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ═══════════════════════════════════════
# PHASE 1 → INDEXING
# ═══════════════════════════════════════

documents = [
    """Python is a high level programming language.
    It was created by Guido van Rossum in 1991.
    Python is known for simple and readable syntax.
    It is used in AI, web development, and data science.""",

    """Machine Learning is a subset of AI.
    It allows computers to learn from data.
    Types include supervised and unsupervised learning.
    Popular ML libraries are scikit-learn and TensorFlow.""",

    """LangChain is a framework for building AI apps.
    It connects LLMs with tools and databases.
    LangChain supports chains, agents, and memory.
    It works with OpenAI, Ollama, and other models.""",

    """RAG stands for Retrieval Augmented Generation.
    It helps AI search your own documents.
    RAG gives accurate and up to date answers.
    It uses vector databases to store knowledge.""",

    """Neural Networks are inspired by human brain.
    They have layers of connected neurons.
    Deep Learning uses many layers of neurons.
    GPT and Claude are based on neural networks."""
]

print("=" * 50)
print("PHASE 1 → INDEXING")
print("=" * 50)

# Step 1 → Split Documents
print("\n📄 Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.create_documents(documents)

print(f"✅ Total Chunks: {len(chunks)}")

# Step 2 → Embeddings
print("\n🔢 Creating embeddings...")

embeddings = OllamaEmbeddings(
    model="llama3"
)

print("✅ Embeddings Ready!")

# Step 3 → Vector Store
print("\n🗄️ Creating FAISS Vector Store...")

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("✅ Vector Store Ready!")

# Step 4 → Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

print("✅ Retriever Ready!")

# ═══════════════════════════════════════
# PHASE 2 → GENERATION
# ═══════════════════════════════════════

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use ONLY the provided context.

If answer does not exist in context say:
"I don't have that information."

Context:
{context}

Question:
{question}

Answer:
""")

# LLM
llm = ChatOllama(model="llama3")

parser = StrOutputParser()

# Helper Function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# ═══════════════════════════════════════
# TESTING
# ═══════════════════════════════════════

print("\n" + "=" * 50)
print("TESTING RAG SYSTEM")
print("=" * 50)

questions = [
    "Who created Python?",
    "What is Machine Learning?",
    "What is LangChain used for?",
    "What does RAG stand for?",
    "What is the capital of France?"
]

for question in questions:

    print(f"\n❓ Question: {question}")
    print("-" * 40)

    response = rag_chain.invoke(question)

    print(f"✅ Answer: {response}")