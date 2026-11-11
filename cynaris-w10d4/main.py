"""Run five sample requests through the W10D4 human-review workflow."""
from src.graph import build_graph, default_config, resume_after_review

SAMPLE_INPUTS = [
    "My Python code is showing an error.",
    "I cannot log in to my account.",
    "Can you explain what LangGraph is?",
    "The application crashes when I install the package.",
    "I need help updating my billing details.",
]

def main():
    graph = build_graph()
    print("W10D4 — Human-in-the-Loop with LangGraph")
    print("Three nodes: classify -> route -> respond")
    for index, message in enumerate(SAMPLE_INPUTS, start=1):
        config = default_config(f"w10d4-example-{index}")
        print(f"\\n--- Example {index}/{len(SAMPLE_INPUTS)} ---")
        print(f"User: {message}")
        result = graph.invoke({"user_input": message, "history": []}, config=config)
        if result.get("__interrupt__"):
            review = result["__interrupt__"][0].value
            print("\\n[PAUSED FOR HUMAN REVIEW]")
            print(f"Category: {review['category']}")
            print(f"Route: {review['route']}")
            print(f"Review request: {review['message']}")
            feedback = input("Enter approval or feedback to resume: ").strip()
            result = resume_after_review(graph, feedback, config)
        print(f"Response: {result.get('response', 'No response generated.')}")
        print(f"Conversation entries saved: {len(result.get('history', []))}")

if __name__ == "__main__":
    main()
