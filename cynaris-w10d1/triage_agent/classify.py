"""Turn a customer message into a label.

Two interchangeable classifiers share one interface:
  RuleClassifier  deterministic keywords: fast, free, testable, always available
  LLMClassifier   asks a local model, and falls back to the rules if the reply
                  is not exactly one known label (small models ramble)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

LABELS = ("billing", "technical", "refund", "general")
PRIORITY = ("refund", "billing", "technical", "general")  # tie-break order: money first

KEYWORDS = {
    "refund": ["refund", "money back", "reimburse", "chargeback"],
    "billing": ["invoice", "billing", "charged", "charge", "payment", "credit card",
                "subscription", "receipt"],
    "technical": ["error", "crash", "crashes", "crashing", "bug", "freezes",
                  "not working", "broken", "slow", "login", "install"],
    "general": ["hello", "hi", "hey", "thanks", "thank you", "hours", "contact"],
}


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: Optional[float]
    method: str


class Classifier(Protocol):
    def classify(self, message: str) -> Classification: ...


class RuleClassifier:
    def classify(self, message: str) -> Classification:
        text = message.lower()
        scores = {
            label: sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", text))
            for label, kws in KEYWORDS.items()
        }
        total = sum(scores.values())
        if total == 0:
            return Classification("unknown", 0.0, "rules")  # nothing matched: don't guess
        best = max(PRIORITY, key=lambda label: (scores[label], -PRIORITY.index(label)))
        # Confidence = the winner's share of all keyword evidence. A tie between
        # two labels therefore scores 0.5, which the router treats as "unsure".
        return Classification(best, scores[best] / total, "rules")


class LLMClassifier:
    PROMPT = ("Classify this customer message into exactly one category: "
              "billing, technical, refund, or general.\n"
              "Reply with ONLY the single category word.\n\nMessage: {message}")

    def __init__(self, chat_fn: Callable[[str], str], fallback: Classifier):
        self.chat_fn = chat_fn
        self.fallback = fallback

    def classify(self, message: str) -> Classification:
        raw = self.chat_fn(self.PROMPT.format(message=message))
        cleaned = raw.strip().strip(".!\"' \n").lower()
        if cleaned in LABELS:  # strict on purpose: "it's probably billing" is rejected
            # LLMs give no calibrated confidence, so we report None rather than
            # inventing a number.
            return Classification(cleaned, None, "llm")
        fallback = self.fallback.classify(message)
        return Classification(fallback.label, fallback.confidence, "rules(llm-fallback)")
