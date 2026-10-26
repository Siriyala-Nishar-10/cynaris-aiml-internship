"""Unit tests for the pure helpers. Run from the project root:  python -m pytest -q"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ANSWER_KEY, DESCRIPTIONS, DOC_KEYWORDS, EXPECTED_DOC, doc_key, keyword_hit


def test_every_question_has_a_doc_and_key():
    assert set(EXPECTED_DOC) == set(ANSWER_KEY) == set(range(1, 11))
    assert set(EXPECTED_DOC.values()) <= set(DOC_KEYWORDS) == set(DESCRIPTIONS)


def test_doc_key_recognises_all_five_papers():
    names = {"GrowWise_SME_An_AI-Driven_Financial_Advisory.txt": "growwise",
             "Mushroom_Species_Classification_CNN.txt": "mushroom",
             "Rag-Based_Framework_for_Intelligent_Banking_Assistants.txt": "banking",
             "The_Rise_of_Agentic_AI_in_Finance.txt": "agentic",
             "When_the_Dashboard_Knows_More_Than_the_Report.txt": "dashboard"}
    for filename, key in names.items():
        assert doc_key(filename) == key


def test_doc_key_unknown_file():
    assert doc_key("notes.txt") is None


def test_keyword_hit_true_and_false():
    assert keyword_hit(1, "It reached a precision of 87.5% versus 62.3%.")
    assert not keyword_hit(1, "It reached a precision of 90%.")
    assert keyword_hit(3, "103 species with 96.70% test accuracy")
    assert not keyword_hit(3, "103 species")          # needs both facts
    assert keyword_hit(8, "from nearly two weeks to just one day")
