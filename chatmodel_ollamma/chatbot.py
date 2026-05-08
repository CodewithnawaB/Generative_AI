import ollama

history = []

print("Open Source Chatbot - LLaMA3 🦙")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    history.append({
        "role": "user",
        "content": user_input
    })

    response = ollama.chat(
        model="llama3",
        messages=history
    )
    
    reply = response["message"]["content"]
    
    history.append({
        "role": "assistant",
        "content": reply
    })
    
    print(f"\nLLaMA: {reply}\n")