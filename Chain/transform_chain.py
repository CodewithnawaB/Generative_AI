from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

model = ChatOllama(model="llama3")
parser = StrOutputParser()

# Step 1 → Transform function
def clean_text(input):
    text = input["raw_text"]
    # Clean the text
    text = text.strip()
    text = text.lower()
    text = " ".join(text.split())
    return {"cleaned_text": text}

# Step 2 → Transform chain
transform_chain = RunnableLambda(clean_text)

# Step 3 → LLM chain
llm_chain = (
    ChatPromptTemplate.from_template(
        "Summarize this text: {cleaned_text}"
    ) | model | parser
)

# Step 4 → Connect transform + LLM
full_chain = transform_chain | llm_chain

# Step 5 → Run with messy input
result = full_chain.invoke({
    "raw_text": "  PYTHON   is   a    GREAT   language!  "
})
print(result)