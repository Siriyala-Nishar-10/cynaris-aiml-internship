"""
W5D2 Task B: Prompting Technique Comparison
-------------------------------------------------
Compares 3 prompting techniques on the SAME underlying task (sentiment
classification with a deliberately tricky, sarcastic example), using the
SAME model and system prompt, so technique is the only variable:

  1. Zero-shot        -- just ask
  2. Few-shot         -- show 3 worked examples first (via FewShotPromptTemplate)
  3. Chain-of-thought -- append "Let's think step by step." (Kojima et al., 2022)

Usage:
    python prompting_techniques.py
    python prompting_techniques.py --model qwen2.5:3b
"""

import argparse
import json
import re
import sys

from ollama_client import OllamaConnectionError, OllamaError, chat, list_models
from prompt_templates import FewShotPromptTemplate, PromptTemplate, chain_of_thought

SYSTEM_PROMPT = "You are a sentiment classifier."
OPTIONS = {"temperature": 0.2, "seed": 42, "num_predict": 200}

# Deliberately sarcastic -- a model that only pattern-matches positive words
# ("great", "love") will get this wrong; the true label is NEGATIVE.
TARGET_TEXT = "Oh great, my flight got delayed for the third time. Love this airline."
TRUE_LABEL = "NEGATIVE"

FEW_SHOT_EXAMPLES = [
    {"text": "This phone's battery dies in 2 hours. Fantastic job, really.", "label": "NEGATIVE"},
    {"text": "The staff were kind and the room was spotless.", "label": "POSITIVE"},
    {"text": "Sure, take my money and give me a broken product. Perfect.", "label": "NEGATIVE"},
]

zero_shot_template = PromptTemplate.from_template(
    "Classify the sentiment of this text as POSITIVE or NEGATIVE:\n\"{text}\"\nAnswer with one word."
)

few_shot_template = FewShotPromptTemplate(
    prefix="Classify the sentiment of each text as POSITIVE or NEGATIVE. Watch out for sarcasm.",
    example_template=PromptTemplate.from_template('Text: "{text}"\nSentiment: {label}'),
    examples=FEW_SHOT_EXAMPLES,
    suffix='Text: "{text}"\nSentiment:',
)


def extract_label(text: str) -> str:
    """Pull POSITIVE or NEGATIVE out of a (possibly rambling) response."""
    match = re.search(r"\b(POSITIVE|NEGATIVE)\b", text.upper())
    return match.group(1) if match else "UNCLEAR"


def build_prompts(text: str) -> dict:
    return {
        "zero_shot": zero_shot_template.format(text=text),
        "few_shot": few_shot_template.format(text=text),
        "chain_of_thought": chain_of_thought(zero_shot_template.format(text=text)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare zero-shot / few-shot / CoT prompting.")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--output", default="prompting_technique_results.json")
    args = parser.parse_args()

    try:
        installed = list_models()
    except OllamaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1
    if not any(n == args.model or n.startswith(args.model + ":") for n in installed):
        print(f"ERROR: model '{args.model}' isn't installed. Run: ollama pull {args.model}")
        return 1

    prompts = build_prompts(TARGET_TEXT)
    print(f"Model: {args.model}")
    print(f"Target text (sarcastic; true label = {TRUE_LABEL}): {TARGET_TEXT}\n")

    results = []
    for technique, prompt in prompts.items():
        print(f"--- {technique} ---")
        print(f"Prompt sent:\n{prompt}\n")
        try:
            result = chat(args.model, prompt, system_prompt=SYSTEM_PROMPT, options=OPTIONS)
        except OllamaError as exc:
            print(f"ERROR: {exc}\n")
            results.append({"technique": technique, "prompt": prompt, "error": str(exc)})
            continue

        predicted = extract_label(result.content)
        correct = predicted == TRUE_LABEL
        print(f"Response: {result.content}")
        print(f"Extracted label: {predicted} | Correct: {correct}\n")

        results.append({
            "technique": technique,
            "prompt": prompt,
            "response": result.content,
            "predicted_label": predicted,
            "correct": correct,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "target_text": TARGET_TEXT, "true_label": TRUE_LABEL,
                   "results": results}, f, indent=2)
    print(f"Saved {args.output}")

    print("\n=== Summary ===")
    for r in results:
        if "predicted_label" in r:
            print(f"{r['technique']:16s}: predicted={r['predicted_label']:8s} "
                  f"correct={r['correct']}")
    print("\nNote: this is ONE example with ONE model and ONE seed -- a single data")
    print("point, not proof that any technique is universally better. Re-run with")
    print("different examples/seeds before drawing a general conclusion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
