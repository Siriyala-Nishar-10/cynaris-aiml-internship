"""
W5D2 Task A: System Prompt Experiment
------------------------------------------
Sends the SAME user question through 4 different system prompts (personas)
to the same model, so the only variable that changes is the system prompt.
This isolates what a system prompt actually controls: tone, length, format
and stance -- not facts the model doesn't know.

Usage:
    python system_prompt_experiment.py
    python system_prompt_experiment.py --model qwen2.5:3b
"""

import argparse
import json
import sys

from ollama_client import OllamaConnectionError, OllamaError, chat, list_models

QUESTION = "Should I use a Random Forest or Logistic Regression for my dataset?"

PERSONAS = {
    "terse_expert": (
        "You are a senior ML engineer. Answer in at most 2 sentences. "
        "No pleasantries, no hedging, just the direct technical answer."
    ),
    "friendly_teacher": (
        "You are a friendly, encouraging teacher explaining machine learning to "
        "a beginner. Use a simple analogy and keep the tone warm and patient."
    ),
    "socratic_tutor": (
        "You are a Socratic tutor. Never give a direct answer. Instead, respond "
        "only with 2-3 guiding questions that help the person reason it out themselves."
    ),
    "json_only": (
        "You are an API. Respond with ONLY a valid JSON object of the shape "
        '{"recommendation": string, "reason": string}. No other text, no markdown fences.'
    ),
}

OPTIONS = {"temperature": 0.2, "seed": 42, "num_predict": 300}


def looks_like_json(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare system prompts on the same question.")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--output", default="system_prompt_results.json")
    args = parser.parse_args()

    try:
        installed = list_models()
    except OllamaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1
    if not any(n == args.model or n.startswith(args.model + ":") for n in installed):
        print(f"ERROR: model '{args.model}' isn't installed. Run: ollama pull {args.model}")
        return 1

    print(f"Model: {args.model}")
    print(f"Question (same for every persona): {QUESTION}\n")

    results = []
    for name, system_prompt in PERSONAS.items():
        print(f"--- Persona: {name} ---")
        print(f"System prompt: {system_prompt}")
        try:
            result = chat(args.model, QUESTION, system_prompt=system_prompt, options=OPTIONS)
        except OllamaError as exc:
            print(f"ERROR: {exc}\n")
            results.append({"persona": name, "system_prompt": system_prompt, "error": str(exc)})
            continue

        word_count = len(result.content.split())
        is_json = looks_like_json(result.content) if name == "json_only" else None
        print(f"Response ({word_count} words): {result.content}")
        if name == "json_only":
            print(f"Valid JSON: {is_json}")
        print()

        results.append({
            "persona": name,
            "system_prompt": system_prompt,
            "response": result.content,
            "word_count": word_count,
            "valid_json": is_json,
            "tokens_per_sec": round(result.tokens_per_sec, 2),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "question": QUESTION, "results": results}, f, indent=2)
    print(f"Saved {args.output}")

    word_counts = {r["persona"]: r.get("word_count") for r in results if "word_count" in r}
    print(f"\nWord count by persona: {word_counts}")
    print("The question and the facts available to the model never changed --")
    print("only the system prompt did. Any difference above is purely the system")
    print("prompt's effect on tone, length, format and stance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
