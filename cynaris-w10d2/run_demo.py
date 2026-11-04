
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from src.graph import build_graph

EXAMPLES = [
    "My Python code has an error.",
    "I cannot log in to my account.",
    "Explain what an AI agent is.",
    "How do I install LangGraph?",
    "I have a question about learning paths.",
]

def main():
    app = build_graph(checkpointer=MemorySaver())
    for index, user_input in enumerate(EXAMPLES, start=1):
        config = {"configurable": {"thread_id": f"w10d2-demo-{index}"}}
        print(f"\\n--- Example {index}: {user_input} ---")
        paused = app.invoke({"user_input": user_input}, config=config)
        if paused.get("__interrupt__"):
            payload = paused["__interrupt__"][0].value
            print("\\n[PAUSED FOR HUMAN REVIEW]")
            print("Category:", payload["category"])
            print("Draft:", payload["draft"])
            feedback = input("Enter approval or feedback to resume: ").strip()
            if not feedback:
                feedback = "Approved after human review."
            completed = app.invoke(Command(resume=feedback), config=config)
        else:
            completed = paused
        print("\\n[FINAL RESPONSE]")
        print(completed.get("final_response", completed))

if __name__ == "__main__":
    main()
