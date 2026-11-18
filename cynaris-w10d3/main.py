"""Run five sample conversations and demonstrate checkpointed memory."""
from langgraph.checkpoint.memory import MemorySaver

from src.graph import build_graph, default_config, resume_after_review

SAMPLE_INPUTS = [
    "My Python code is showing an error.",
    "I cannot log in to my account.",
    "Can you explain what an AI agent is?",
    "The application is crashing when I open it.",
    "I need help resetting my password.",
]


def main():
    graph = build_graph(checkpointer=MemorySaver())
    config = default_config("w10d3-live-demo")

    print("W10D3 — LangGraph Persistent Conversations")
    print("All five inputs use the same thread ID to retain conversation state.\n")

    for index, user_input in enumerate(SAMPLE_INPUTS, start=1):
        print(f"\n--- Example {index}/5 ---")
        print(f"User: {user_input}")

        result = graph.invoke(
            {"user_input": user_input, "conversation_history": []},
            config=config,
        )

        while result.get("__interrupt__"):
            interrupt_value = result["__interrupt__"][0].value
            print("\n[PAUSED FOR HUMAN REVIEW]")
            print(interrupt_value["draft_response"])
            feedback = input("Enter approval or feedback to resume: ").strip()
            if not feedback:
                feedback = "Approved"
            result = resume_after_review(graph, feedback, config)

        print(f"Category: {result.get('category')}")
        print(f"Route: {result.get('route', 'general_support')}")
        print(f"Assistant: {result.get('final_response')}")
        print(f"Conversation entries saved: {len(result.get('conversation_history', []))}")

    print("\nDemo complete. Checkpointed state is retained for this thread")
    print("while this Python process and its in-memory checkpointer remain active.")


if __name__ == "__main__":
    main()
