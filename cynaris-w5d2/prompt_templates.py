"""
W5D2: Prompt templates (dependency-free, LangChain-style interface)
------------------------------------------------------------------------
LangChain's `langchain` / `langchain-ollama` packages could not be installed
in this sandboxed environment (no internet access to PyPI -- verified with
`pip install langchain langchain-ollama`, same failure mode as aif360/shap in
W4D6). The core idea LangChain's PromptTemplate provides -- a reusable string
with named `{placeholders}` filled in at call time -- doesn't need a library,
so it's implemented directly here.

If you have internet access, the real equivalent is:

    from langchain_core.prompts import PromptTemplate
    template = PromptTemplate.from_template("Explain {topic} to a {audience}.")
    template.format(topic="overfitting", audience="beginner")

which does exactly what PromptTemplate.format() below does.
"""

import re
from dataclasses import dataclass, field


class PromptTemplate:
    """A reusable prompt with named {placeholders}, filled in at call time.

    Equivalent to langchain_core.prompts.PromptTemplate for the plain string
    case (no partial variables, no output parsers).
    """

    def __init__(self, template: str):
        self.template = template
        self.input_variables = sorted(set(re.findall(r"\{(\w+)\}", template)))

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        return cls(template)

    def format(self, **kwargs) -> str:
        missing = set(self.input_variables) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variable(s): {sorted(missing)}")
        return self.template.format(**kwargs)

    def __repr__(self):
        return f"PromptTemplate(input_variables={self.input_variables})"


@dataclass
class FewShotPromptTemplate:
    """Builds a few-shot prompt: instruction + N worked examples + the new
    input, so the model can pattern-match the examples' format rather than
    guess it from the instruction alone.

    Equivalent to langchain_core.prompts.FewShotPromptTemplate for the
    simple text-join case.
    """
    prefix: str
    example_template: PromptTemplate
    examples: list = field(default_factory=list)  # list of dicts matching example_template's variables
    suffix: str = ""
    example_separator: str = "\n\n"

    def format(self, **kwargs) -> str:
        formatted_examples = [self.example_template.format(**ex) for ex in self.examples]
        blocks = [self.prefix] + formatted_examples + [self.suffix.format(**kwargs) if kwargs else self.suffix]
        return self.example_separator.join(b for b in blocks if b)


def chain_of_thought(question: str) -> str:
    """Appends the classic 'Let's think step by step' CoT trigger phrase
    (Kojima et al., 2022) to a zero-shot question."""
    return f"{question}\nLet's think step by step."
