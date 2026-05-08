import ollama

# ✅ Good Prompt with all components
system_prompt = """
You are an expert Python tutor.
- Always explain in simple words
- Give code examples
- Keep answers short and clear
"""

user_prompt = """
I am a beginner.
Explain what is a list in Python.
Give me one simple example.
Format: 
1. Definition
2. Example code
3. Output
"""

response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]
)

print(response["message"]["content"])