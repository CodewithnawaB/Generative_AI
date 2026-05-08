# pip install langchain langchain-ollama

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Step 1 → Create Model
model = ChatOllama(model="llama3")

# Step 2 → Create Prompt
prompt = ChatPromptTemplate.from_template("""
You are an expert teacher.
Explain this topic simply: {topic}
""")

# Step 3 → Create Chain using | pipe
chain = prompt | model

# Step 4 → Run Chain
result = chain.invoke({"topic": "Python lists"})
print(result.content)