from src.graph import build_graph, default_config, resume_after_review

def test_classification_and_routes_for_five_inputs():
    graph = build_graph()
    cases = [
        ("My Python code has an error.", "technical", "technical_support"),
        ("I cannot log in to my account.", "account", "account_support"),
        ("Explain what an AI agent is.", "general", "general_support"),
        ("The app crashes during installation.", "technical", "technical_support"),
        ("I have a billing payment question.", "account", "account_support"),
    ]
    for index, (message, category, expected_route) in enumerate(cases):
        config = default_config(f"route-test-{index}")
        paused = graph.invoke({"user_input": message, "history": []}, config=config)
        assert paused["category"] == category
        assert paused["route"] == expected_route
        assert "__interrupt__" in paused

def test_human_review_pauses_and_resumes():
    graph = build_graph()
    config = default_config("human-review-test")
    paused = graph.invoke(
        {"user_input": "My Python code has an error.", "history": []}, config=config
    )
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["route"] == "technical_support"
    resumed = resume_after_review(graph, "Approved", config)
    assert resumed["review_feedback"] == "Approved"
    assert "Human review approved" in resumed["response"]
    assert len(resumed["history"]) == 1

def test_non_approval_feedback_is_recorded():
    graph = build_graph()
    config = default_config("human-feedback-test")
    graph.invoke({"user_input": "Please explain this topic.", "history": []}, config=config)
    resumed = resume_after_review(graph, "Please revise the wording.", config)
    assert resumed["review_feedback"] == "Please revise the wording."
    assert "Please revise the wording." in resumed["response"]
