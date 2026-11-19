
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from src.graph import build_graph, classify_node, route_by_category

def test_five_inputs_classify_and_route():
    examples = [
        ("My Python code has an error", "technical", "respond_technical"),
        ("I cannot log in to my account", "account", "respond_account"),
        ("Explain AI agents", "general", "respond_general"),
        ("How do I install LangGraph?", "technical", "respond_technical"),
        ("I have a question about learning paths", "general", "respond_general"),
    ]
    for message, category, route in examples:
        state = classify_node({"user_input": message})
        assert state["category"] == category
        assert route_by_category(state) == route

def test_human_interrupt_and_resume():
    app = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "w10d2-hitl-test"}}
    paused = app.invoke({"user_input": "My Python code has an error"}, config=config)
    assert paused.get("__interrupt__")
    assert paused["category"] == "technical"
    completed = app.invoke(Command(resume="Approved after review."), config=config)
    assert completed["human_feedback"] == "Approved after review."
    assert "Human review: Approved after review." in completed["final_response"]
