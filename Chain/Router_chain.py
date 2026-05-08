from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")
parser = StrOutputParser()

# Step 1 → Create different expert chains
math_chain = (
    ChatPromptTemplate.from_template(
        "You are a math expert. Answer: {question}"
    ) | model | parser
)

code_chain = (
    ChatPromptTemplate.from_template(
        "You are a Python expert. Answer: {question}"
    ) | model | parser
)

general_chain = (
    ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {question}"
    ) | model | parser
)

# Step 2 → Router function
def route_question(input):
    question = input["question"].lower()
    
    if any(word in question for word in 
           ["math", "calculate", "equation", "number"]):
        print("→ Routing to MATH Chain 🔢")
        return math_chain
    
    elif any(word in question for word in 
             ["python", "code", "programming", "error"]):
        print("→ Routing to CODE Chain 💻")
        return code_chain
    
    else:
        print("→ Routing to GENERAL Chain 🌐")
        return general_chain

# Step 3 → Create router chain
from langchain_core.runnables import RunnableLambda

router_chain = RunnableLambda(
    lambda x: route_question(x).invoke(x)
)

# Step 4 → Test routing
questions = [
    {"question": "What is 25 multiplied by 4?"},
    {"question": "How do I fix Python list error?"},
    {"question": "What is the capital of France?"},
]

for q in questions:
    print(f"\nQuestion: {q['question']}")
    result = router_chain.invoke(q)
    print(f"Answer: {result[:100]}...")
    print("-" * 40)