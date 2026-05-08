from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Step 1 → Define your structure
class PersonInfo(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(description="Age of person")
    city: str = Field(description="City of person")
    job: str = Field(description="Job of person")

# Step 2 → Create model
model = ChatOllama(model="llama3")

# Step 3 → Attach structure to model
structured_model = model.with_structured_output(PersonInfo)

# Step 4 → Create prompt
prompt = ChatPromptTemplate.from_template("""
Extract all information from this text:
{text}
""")

# Step 5 → Create chain
chain = prompt | structured_model

# Step 6 → Run
result = chain.invoke({
    "text": "Ali is a 30 year old doctor living in Karachi"
})

# Step 7 → Access data easily!
print(f"Name: {result.name}")
print(f"Age:  {result.age}")
print(f"City: {result.city}")
print(f"Job:  {result.job}")

# Output:
# Name: Ali
# Age:  30
# City: Karachi
# Job:  Doctor