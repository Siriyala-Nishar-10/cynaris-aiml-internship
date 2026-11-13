import pytest

from triage_agent.classify import LLMClassifier, RuleClassifier

rules = RuleClassifier()

# The five inputs from the lesson, with the label each should get.
FIVE_INPUTS = [
    ("I was charged twice on my last invoice", "billing"),
    ("The app crashes every time I open it", "technical"),
    ("I want a refund, this product is broken", "refund"),
    ("Hello, what are your opening hours?", "general"),
    ("asdf qwerty", "unknown"),
]


@pytest.mark.parametrize("message,expected", FIVE_INPUTS)
def test_five_lesson_inputs_get_expected_label(message, expected):
    assert rules.classify(message).label == expected


def test_matching_is_case_insensitive_and_ignores_punctuation():
    assert rules.classify("REFUND!!! now.").label == "refund"


def test_keywords_match_whole_words_only():
    # 'hi' sits inside 'this' and 'which'; it must not count as a greeting.
    assert rules.classify("this which").label == "unknown"


def test_empty_message_is_unknown_not_a_guess():
    result = rules.classify("")
    assert result.label == "unknown" and result.confidence == 0.0


def test_tie_between_labels_gives_half_confidence_and_priority_winner():
    result = rules.classify("I was charged twice and the app crashes")
    assert result.label == "billing"          # billing outranks technical
    assert result.confidence == 0.5


def test_refund_outranks_billing_on_a_tie():
    assert rules.classify("refund my invoice").label == "refund"


# ---- LLM classifier (chat function faked: no model needed) ----

def test_llm_valid_label_is_used_with_no_invented_confidence():
    result = LLMClassifier(lambda prompt: "billing", rules).classify("anything")
    assert (result.label, result.confidence, result.method) == ("billing", None, "llm")


def test_llm_label_is_cleaned_of_case_and_punctuation():
    assert LLMClassifier(lambda p: "  Technical.\n", rules).classify("x").label == "technical"


def test_llm_rambling_reply_falls_back_to_rules():
    chat = lambda p: "I think this is probably a billing thing, maybe refund"
    result = LLMClassifier(chat, rules).classify("The app crashes every time")
    assert result.label == "technical" and result.method == "rules(llm-fallback)"


def test_llm_prompt_contains_the_customer_message():
    seen = []
    LLMClassifier(lambda p: seen.append(p) or "general", rules).classify("my special message")
    assert "my special message" in seen[0]
