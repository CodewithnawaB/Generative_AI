from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3")

# ── Type 1: Buffer Memory ─────────────
# Remember ALL conversation
buffer_memory = ConversationBufferMemory(
    return_messages = True,
    memory_key      = "chat_history"
)

buffer_memory.save_context(
    {"input": "My name is Ali"},
    {"output": "Hello Ali!"}
)
buffer_memory.save_context(
    {"input": "I love Python"},
    {"output": "Python is great!"}
)

print("Buffer Memory:")
print(buffer_memory.load_memory_variables({}))

# ── Type 2: Window Memory ─────────────
# Remember only LAST 2 messages
window_memory = ConversationBufferWindowMemory(
    k               = 2,
    return_messages = True
)

window_memory.save_context(
    {"input": "Message 1"}, {"output": "Reply 1"}
)
window_memory.save_context(
    {"input": "Message 2"}, {"output": "Reply 2"}
)
window_memory.save_context(
    {"input": "Message 3"}, {"output": "Reply 3"}
)

print("\nWindow Memory (last 2 only):")
print(window_memory.load_memory_variables({}))

# ── Type 3: Summary Memory ────────────
# Summarize old conversations
summary_memory = ConversationSummaryMemory(
    llm             = model,
    return_messages = True
)

summary_memory.save_context(
    {"input": "I am a student studying AI"},
    {"output": "Great! AI is an exciting field"}
)

print("\nSummary Memory:")
print(summary_memory.load_memory_variables({}))

# ── Memory in Chatbot ─────────────────
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system",  "You are helpful assistant"),
    MessagesPlaceholder("chat_history"),
    ("human",   "{question}")
])

chain  = prompt | model | StrOutputParser()
memory = ConversationBufferMemory(
    return_messages = True,
    memory_key      = "chat_history"
)

print("\n💬 Chatbot with Memory:")
questions = [
    "My name is Ahmed",
    "What is my name?",
    "I love playing cricket",
    "What sport do I love?",
]

for q in questions:
    history  = memory.load_memory_variables({})
    response = chain.invoke({
        "chat_history": history["chat_history"],
        "question"    : q
    })
    memory.save_context(
        {"input"  : q},
        {"output" : response}
    )
    print(f"You: {q}")
    print(f"Bot: {response}\n")