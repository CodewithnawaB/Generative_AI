from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")
parser = StrOutputParser()

# Step 1 → Create 3 different chains
explain_chain = (
    ChatPromptTemplate.from_template(
        "Explain {topic} in simple words"
    ) | model | parser
)

example_chain = (
    ChatPromptTemplate.from_template(
        "Give 3 real world examples of {topic}"
    ) | model | parser
)

quiz_chain = (
    ChatPromptTemplate.from_template(
        "Give 2 quiz questions about {topic}"
    ) | model | parser
)

# Step 2 → Run ALL at same time!
parallel_chain = RunnableParallel(
    explanation = explain_chain,
    examples    = example_chain,
    quiz        = quiz_chain
)

# Step 3 → Run once → Get 3 results!
result = parallel_chain.invoke({"topic": "Machine Learning"})

print("=" * 40)
print("EXPLANATION:")
print(result["explanation"])

print("\n" + "=" * 40)
print("EXAMPLES:")
print(result["examples"])

print("\n" + "=" * 40)
print("QUIZ:")
print(result["quiz"])