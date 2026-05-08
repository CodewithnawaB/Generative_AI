from langchain_core.runnables import RunnableBranch
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")
parser = StrOutputParser()

# Step 1 → Create different chains
easy_chain = (
    ChatPromptTemplate.from_template(
        "Explain for a 10 year old child: {question}"
    ) | model | parser
)

medium_chain = (
    ChatPromptTemplate.from_template(
        "Explain for a college student: {question}"
    ) | model | parser
)

expert_chain = (
    ChatPromptTemplate.from_template(
        "Explain with technical depth: {question}"
    ) | model | parser
)

# Step 2 → Create branch
branch = RunnableBranch(
    # Condition 1 → beginner
    (
        lambda x: x["level"] == "beginner",
        easy_chain
    ),
    # Condition 2 → intermediate
    (
        lambda x: x["level"] == "intermediate",
        medium_chain
    ),
    # Default → expert
    expert_chain
)

# Step 3 → Test all levels
inputs = [
    {"question": "What is AI?", "level": "beginner"},
    {"question": "What is AI?", "level": "intermediate"},
    {"question": "What is AI?", "level": "expert"},
]

for inp in inputs:
    print(f"\nLevel: {inp['level'].upper()}")
    print("-" * 30)
    result = branch.invoke(inp)
    print(result[:150] + "...")