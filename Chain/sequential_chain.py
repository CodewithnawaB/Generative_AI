from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")
parser = StrOutputParser()

# Step 1 → First Chain → Explain topic
explain_prompt = ChatPromptTemplate.from_template(
    "Explain this topic in detail: {topic}"
)
explain_chain = explain_prompt | model | parser

# Step 2 → Second Chain → Summarize explanation
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this in 3 bullet points: {explanation}"
)
summary_chain = summary_prompt | model | parser

# Step 3 → Third Chain → Translate summary
translate_prompt = ChatPromptTemplate.from_template(
    "Translate this to Urdu: {summary}"
)
translate_chain = translate_prompt | model | parser

# Step 4 → Connect ALL chains together
full_chain = (
    explain_chain
    | (lambda x: {"explanation": x})
    | summary_chain
    | (lambda x: {"summary": x})
    | translate_chain
)

# Step 5 → Run
result = full_chain.invoke({"topic": "Artificial Intelligence"})
print(result)