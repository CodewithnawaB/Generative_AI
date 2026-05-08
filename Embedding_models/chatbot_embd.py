from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama

# ── Load embedding model ──────────────────────────
print("Loading Embedding Model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding Model Ready! ✅\n")

# ── Knowledge Base ────────────────────────────────
knowledge_base = [
    "Python is a popular programming language.",
    "AI stands for Artificial Intelligence.",
    "LLM means Large Language Model.",
    "Ollama runs AI models locally on your computer.",
    "Embeddings convert text into numbers for AI.",
]

# ── Convert knowledge to embeddings ──────────────
print("Processing Knowledge Base...")
kb_embeddings = embed_model.encode(knowledge_base)
print("Knowledge Base Ready! ✅\n")

# ── Function to find best match ───────────────────
def find_best_match(user_question):
    # Convert question to embedding
    question_embedding = embed_model.encode([user_question])
    
    # Compare with knowledge base
    similarities = cosine_similarity(
        question_embedding, 
        kb_embeddings
    )[0]
    
    # Get best match index
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    best_match = knowledge_base[best_index]
    
    return best_match, best_score

# ── Chat history ──────────────────────────────────
history = []

print("=" * 40)
print("  Embedding Chatbot Ready! 🤖")
print("  Type 'quit' to exit")
print("=" * 40 + "\n")

# ── Main chat loop ────────────────────────────────
while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Goodbye! 👋")
        break
    
    # Step 1 → Find relevant knowledge
    best_match, score = find_best_match(user_input)
    print(f"\n[Embedding Match]: {best_match}")
    print(f"[Similarity Score]: {score:.2f}\n")
    
    # Step 2 → Build prompt with context
    prompt = f"""
    Use this context to answer the question:
    Context: {best_match}
    Question: {user_input}
    Answer clearly and simply.
    """
    
    # Step 3 → Add to history
    history.append({
        "role": "user",
        "content": prompt
    })
    
    # Step 4 → Send to LLaMA
    response = ollama.chat(
        model="llama3",
        messages=history
    )
    
    reply = response["message"]["content"]
    
    # Step 5 → Save reply to history
    history.append({
        "role": "assistant",
        "content": reply
    })
    
    print(f"Bot: {reply}\n")
    print("-" * 40 + "\n")