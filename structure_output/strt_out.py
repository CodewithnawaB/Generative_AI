# pip install langchain langchain-ollama

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Step 1 → Create model
model = ChatOllama(
    model="llama3",
    format="json"    # ← tells AI return JSON!
)

# Step 2 → Create prompt
prompt = ChatPromptTemplate.from_template("""
Extract information and return ONLY JSON:
{{
  "name": "person name",
  "age": age as number,
  "city": "city name"
}}

Text: {text}
""")

# Step 3 → Create chain
chain = prompt | model

# Step 4 → Run
result = chain.invoke({
    "text": "John is 25 years old and lives in London"
})

print(result.content)
# Output:
# {
#   "name": "John",
#   "age": 25,
#   "city": "London"
# }