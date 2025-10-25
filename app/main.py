from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


load_dotenv()


class State(TypedDict):
    user_input: str
    response: str


def call_llm(state: State):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    user_input = state["user_input"]
    result = llm.invoke(user_input)
    content = None
    try:
        if result is None:
            content = ""
        elif hasattr(result, "content"):
            content = result.content
        elif hasattr(result, "text"):
            content = result.text
        elif hasattr(result, "generations"):
            # Some APIs return generations: list[list[Generation]]
            try:
                gen = result.generations[0][0]
                content = getattr(gen, "text", getattr(gen, "content", str(gen)))
            except Exception:
                content = str(result)
        else:
            content = str(result)
    except Exception:
        # Ensure we never crash here; use a fallback string
        content = ""

    state["response"] = content

    return state


graph = StateGraph(State)
graph.add_node("llm", call_llm)

graph.add_edge(START, "llm")
graph.add_edge("llm", END)

app = graph.compile()

if __name__ == "__main__":
    user_input = input("Enter your input: ")
    result = app.invoke({"user_input": user_input})
    print("LLM Response:", result["response"])
