"""
W5D1 Task 2: Call the Ollama API with a custom system prompt, test 5 prompts
------------------------------------------------------------------------------
Usage:
    python run_prompts.py
    python run_prompts.py --model qwen2.5:3b --output qwen_results.json

Prerequisites: Ollama is running and the model has been pulled
(`ollama pull llama3.2:3b`).
"""

import argparse
import json
import sys

from ollama_client import (
    OllamaConnectionError, OllamaError, chat, list_models,
)

# The system prompt shapes the model's persona and output rules for every turn.
SYSTEM_PROMPT = (
    "You are a concise technical mentor for AI/ML interns. "
    "Answer in at most 4 sentences, in plain language, and include one concrete "
    "example when it helps. If you are not sure about something, say so instead "
    "of guessing."
)

# Five prompts drawn from what the course has covered so far.
PROMPTS = [
    "What is overfitting, and how can I spot it in a learning curve?",
    "Explain the difference between precision and recall.",
    "When would I pick a Random Forest over Logistic Regression?",
    "What does it mean to quantise a language model?",
    "Write a one-line Python expression that computes accuracy from two lists y_true and y_pred.",
]

# Low temperature + fixed seed keeps runs (mostly) repeatable, which matters
# when you want to compare models or re-run for your evidence screenshot.
OPTIONS = {"temperature": 0.2, "seed": 42, "num_predict": 250}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--output", default="run_prompts_results.json")
    args = parser.parse_args()

    try:
        installed = list_models()
    except OllamaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not any(name == args.model or name.startswith(args.model + ":") for name in installed):
        print(f"ERROR: model '{args.model}' isn't installed. Installed: {installed or 'none'}")
        print(f"Run: ollama pull {args.model}")
        return 1

    print(f"Model: {args.model}")
    print(f"System prompt: {SYSTEM_PROMPT}\n")

    results = []
    for i, prompt in enumerate(PROMPTS, start=1):
        print(f"--- Prompt {i}/{len(PROMPTS)} ---")
        print(f"Q: {prompt}")
        try:
            result = chat(args.model, prompt, system_prompt=SYSTEM_PROMPT, options=OPTIONS)
        except OllamaError as exc:
            print(f"ERROR: {exc}\n")
            results.append({"prompt": prompt, "error": str(exc)})
            continue
        print(f"A: {result.content}")
        print(f"   [{result.completion_tokens} tokens, {result.tokens_per_sec:.1f} tok/s, "
              f"{result.latency_s:.1f}s total, load {result.load_s:.1f}s]\n")
        results.append({
            "prompt": prompt,
            "response": result.content,
            "completion_tokens": result.completion_tokens,
            "tokens_per_sec": round(result.tokens_per_sec, 2),
            "latency_s": round(result.latency_s, 2),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "system_prompt": SYSTEM_PROMPT,
                   "options": OPTIONS, "results": results}, f, indent=2)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
