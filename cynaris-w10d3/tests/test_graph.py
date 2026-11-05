from langgraph.checkpoint.memory import MemorySaver

from src.graph import build_graph, default_config, resume_after_review


def run_one_turn(graph, config, message, feedback="Approved by test"):
    paused = graph.invoke(
        {"user_input": message, "conversation_history": []},
        config=config,
    )
    assert paused.get("__interrupt__"), "Expected human-review interrupt"
    return resume_after_review(graph, feedback, config)


def test_five_inputs_are_classified_and_routed():
    graph = build_graph(checkpointer=MemorySaver())
    cases = [
        ("Python code has an error", "technical", "technical_support"),
        ("I cannot log in to my account", "account", "account_support"),
        ("Explain machine learning", "general", "general_support"),
        ("The app keeps crashing", "technical", "technical_support"),
        ("I forgot my password", "account", "account_support"),
    ]

    for index, (message, expected_category, expected_route) in enumerate(cases):
        config = default_config(f"routing-{index}")
        result = run_one_turn(graph, config, message)
        assert result["category"] == expected_category
        assert result["route"] == expected_route
        assert "Approved by test" in result["final_response"]


def test_human_interrupt_can_be_resumed():
    graph = build_graph(checkpointer=MemorySaver())
    config = default_config("human-review")

    paused = graph.invoke(
        {"user_input": "I cannot sign in", "conversation_history": []},
        config=config,
    )
    assert paused.get("__interrupt__")

    resumed = resume_after_review(
        graph,
        "Please include the password reset page",
        config,
    )
    assert "Please include the password reset page" in resumed["final_response"]


def test_same_thread_keeps_conversation_history_between_turns():
    graph = build_graph(checkpointer=MemorySaver())
    config = default_config("persistent-conversation")

    first = run_one_turn(graph, config, "Hello, I need general help.")
    assert len(first["conversation_history"]) == 1

    second = run_one_turn(graph, config, "My Python code has an error.")
    assert len(second["conversation_history"]) == 2
    assert "Hello, I need general help." in second["conversation_history"][0]
    assert "My Python code has an error." in second["conversation_history"][1]
