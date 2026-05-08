from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")
parser = StrOutputParser()

chain = (
    ChatPromptTemplate.from_template(
        "In one line explain: {topic}"
    ) | model | parser
)

# ✅ Method 1 → invoke() → One input
print("=== INVOKE ===")
result = chain.invoke({"topic": "Python"})
print(result)

# ✅ Method 2 → batch() → Many inputs at once
print("\n=== BATCH ===")
results = chain.batch([
    {"topic": "Python"},
    {"topic": "AI"},
    {"topic": "LangChain"},
])
for r in results:
    print(r)

# ✅ Method 3 → stream() → Word by word
print("\n=== STREAM ===")
for chunk in chain.stream({"topic": "Machine Learning"}):
    print(chunk, end="", flush=True)
print()