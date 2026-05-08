from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")
parser = StrOutputParser()

# Step 1 → Your custom functions
def clean_input(text):
    return text.strip().lower()

def add_excitement(text):
    return text + " 🎉🚀✅"

def count_words(text):
    words = len(text.split())
    return f"{text}\n\n[Word Count: {words}]"

# Step 2 → Convert to Runnable
clean_runnable     = RunnableLambda(clean_input)
exciting_runnable  = RunnableLambda(add_excitement)
wordcount_runnable = RunnableLambda(count_words)

# Step 3 → Create prompt
prompt = ChatPromptTemplate.from_template(
    "Explain this topic simply: {topic}"
)

# Step 4 → Connect everything in chain
chain = (
    clean_runnable          # Clean input first
    | (lambda x: {"topic": x})
    | prompt                # Fill prompt
    | model                 # Send to LLM
    | parser                # Parse output
    | exciting_runnable     # Add excitement
    | wordcount_runnable    # Count words
)

# Step 5 → Run
result = chain.invoke("  PYTHON PROGRAMMING  ")
print(result)