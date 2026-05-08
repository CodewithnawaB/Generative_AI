from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# Step 1 → Create knowledge base
documents = [
    "Python was created by Guido van Rossum in 1991.",
    "LangChain helps build AI applications easily.",
    "LLM stands for Large Language Model.",
    "Ollama runs AI models locally on your computer.",
    "Embeddings convert text into number vectors.",
]

# Step 2 → Create embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Step 3 → Store in vector database
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever()

# Step 4 → Create RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
Use ONLY this context to answer:
Context: {context}
Question: {question}
If answer not in context say: I dont know
""")

# Step 5 → Create RAG chain
model = ChatOllama(model="llama3")
parser = StrOutputParser()

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | model
    | parser
)

# Step 6 → Ask questions
questions = [
    "Who created Python?",
    "What is LangChain?",
    "What are embeddings?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")
    print("-" * 40)