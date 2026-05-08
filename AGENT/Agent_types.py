from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import numexpr

# ─────────────────────────────────────
# LOCAL OPEN-SOURCE MODEL
# ─────────────────────────────────────

llm = ChatOllama(
    model="mistral",
    temperature=0
)

# ─────────────────────────────────────
# TOOLS
# ─────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions"""

    try:
        result = numexpr.evaluate(expression)
        return str(result)

    except Exception as e:
        return f"Calculation Error: {str(e)}"


@tool
def search_info(query: str) -> str:
    """Search simple information"""

    db = {
        "python": "Python is a programming language.",
        "ai": "AI means Artificial Intelligence.",
        "langchain": "LangChain is used for AI applications.",
    }

    query = query.lower()

    for key, value in db.items():
        if key in query:
            return value

    return "No information found."


# ─────────────────────────────────────
# CREATE AGENT
# ─────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=[calculator, search_info]
)

# ─────────────────────────────────────
# RUN AGENT
# ─────────────────────────────────────

response = agent.invoke({
    "messages": [
        (
            "human",
            "What is Python and calculate 15 * 3?"
        )
    ]
})

# ─────────────────────────────────────
# FINAL OUTPUT
# ─────────────────────────────────────

print("\n🧠 FINAL RESPONSE:\n")

print(response["messages"][-1].content)s