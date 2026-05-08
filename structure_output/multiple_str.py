from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Step 1 → Define structures
class ProductInfo(BaseModel):
    product_name: str = Field(description="Name of product")
    price: float = Field(description="Price of product")
    category: str = Field(description="Category of product")
    in_stock: bool = Field(description="Is product in stock")

class ReviewInfo(BaseModel):
    rating: int = Field(description="Rating 1 to 5")
    sentiment: str = Field(description="positive/negative/neutral")
    summary: str = Field(description="Short summary of review")

# Step 2 → Create model
model = ChatOllama(model="llama3")

# Step 3 → Product extraction chain
product_chain = (
    ChatPromptTemplate.from_template(
        "Extract product info from: {text}"
    )
    | model.with_structured_output(ProductInfo)
)

# Step 4 → Review extraction chain
review_chain = (
    ChatPromptTemplate.from_template(
        "Extract review info from: {text}"
    )
    | model.with_structured_output(ReviewInfo)
)

# Step 5 → Run product extraction
product = product_chain.invoke({
    "text": "iPhone 15 costs $999, it is a smartphone, available now"
})
print(f"Product: {product.product_name}")
print(f"Price:   ${product.price}")
print(f"Category:{product.category}")
print(f"In Stock:{product.in_stock}")

print("\n" + "="*40 + "\n")

# Step 6 → Run review extraction
review = review_chain.invoke({
    "text": "This phone is amazing! Best purchase ever. 5 stars!"
})
print(f"Rating:    {review.rating}/5")
print(f"Sentiment: {review.sentiment}")
print(f"Summary:   {review.summary}")