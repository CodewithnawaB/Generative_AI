from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="llama3")
parser = StrOutputParser()

# Step 1 → Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember our conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Step 2 → Create chain
chain = prompt | model | parser

# Step 3 → Store history
chat_history = []

print("Conversation Chain Ready! 💬")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break
    
    # Step 4 → Run with history
    response = chain.invoke({
        "history": chat_history,
        "question": user_input
    })
    
    # Step 5 → Save to history
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    
    print(f"Bot: {response}\n")