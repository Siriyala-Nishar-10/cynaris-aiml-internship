"""
W5D1 Task 3: Compare llama3.2:3b vs qwen2.5:3b on the same 3 questions
--------------------------------------------------------------------------
Usage:
    python compare_models.py
    python compare_models.py --models llama3.2:3b qwen2.5:3b

Outputs:
    comparison_results.json   raw data (responses, timings, check results)
    comparison_report.md      readable report with a section for YOUR observations

The three questions are chosen so each tests something different:
  1. Factual recall     -> small models often hallucinate specifics (names/dates)
  2. Arithmetic reasoning -> has a single verifiable answer
  3. Instruction following -> "exactly 3 bullets, each under 20 words" is checkable

The automatic checks are deliberately simple (substring / line counting). They
are a fast objective signal, NOT a full quality judgement -- a correct answer
phrased unusually (e.g. "eighty" instead of "80") can fail a check, and a check
can pass on an answer that's otherwise poor. Read the actual responses.
"""

import argparse
import json
import re
import sys
from datetime import datetime

from ollama_client import OllamaConnectionError, OllamaError, chat, list_models

SYSTEM_PROMPT = "You are a helpful assistant. Be accurate and concise."
OPTIONS = {"temperature": 0.2, "seed": 42, "num_predict": 300}
DEFAULT_MODELS = ["llama3.2:3b", "qwen2.5:3b"]

def contains_all(text: str, needles: list) -> bool:
    """True if every needle appears in text (case-insensitive)."""
    lowered = text.lower()
    return all(n.lower() in lowered for n in needles)


def contains_number(text: str, number: str) -> bool:
    """True if `number` appears as a standalone number, so '80' doesn't match
    '180', '1.80' or '80.5' (but does match 'is 80.' at the end of a sentence)."""
    return re.search(rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)", text) is not None


def extract_bullets(text: str) -> list:
    """Return the text of each bullet line ('-', '*', '•', or '1.' / '1)' style)."""
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:[-*\u2022]|\d+[.)])\s+(.*)$", stripped)
        if match:
            bullets.append(match.group(1))
    return bullets


def exactly_three_short_bullets(text: str) -> bool:
    bullets = extract_bullets(text)
    return len(bullets) == 3 and all(len(b.split()) < 20 for b in bullets)


QUESTIONS = [
    {
        "id": "Q1 factual",
        "prompt": "Who wrote 'The Discovery of India', and in what year was it first published?",
        "check_desc": "mentions 'Nehru' and '1946'",
        "check": lambda t: contains_all(t, ["nehru", "1946"]),
    },
    {
        "id": "Q2 reasoning",
        "prompt": "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
        "check_desc": "answer contains 80",
        "check": lambda t: contains_number(t, "80"),
    },
    {
        "id": "Q3 instruction-following",
        "prompt": "Explain what overfitting is in exactly 3 bullet points, each under 20 words.",
        "check_desc": "exactly 3 bullets, each under 20 words",
        "check": exactly_three_short_bullets,
    },
]

def run_model(model: str) -> list:
    """Warm the model up (so load time doesn't skew the timings), then ask every question."""
    print(f"\n=== {model} ===")
    try:
        chat(model, "Say OK.", options={"num_predict": 5})  # warm-up, result discarded
    except OllamaError as exc:
        print(f"Warm-up failed: {exc}")
        raise

    rows = []
    for q in QUESTIONS:
        result = chat(model, q["prompt"], system_prompt=SYSTEM_PROMPT, options=OPTIONS)
        passed = bool(q["check"](result.content))
        print(f"[{q['id']}] check {'PASS' if passed else 'FAIL'} "
              f"| {result.tokens_per_sec:.1f} tok/s | {result.latency_s:.1f}s")
        rows.append({
            "question_id": q["id"],
            "prompt": q["prompt"],
            "check_desc": q["check_desc"],
            "check_passed": passed,
            "response": result.content,
            "completion_tokens": result.completion_tokens,
            "tokens_per_sec": round(result.tokens_per_sec, 2),
            "latency_s": round(result.latency_s, 2),
            "word_count": len(result.content.split()),
        })
    return rows


def build_report(all_results: dict) -> str:
    models = list(all_results)
    lines = [
        "# W5D1: Local LLM Comparison Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Models: {', '.join(models)}  ",
        f"System prompt: `{SYSTEM_PROMPT}`  ",
        f"Options: `{OPTIONS}`",
        "",
        "> Automatic checks are simple substring/line-count tests -- a fast objective",
        "> signal, not a full quality judgement. See the full responses below.",
        "",
        "## Summary",
        "",
        "| Question | Check | " + " | ".join(f"{m} check | {m} tok/s" for m in models) + " |",
        "|---|---|" + "---|---|" * len(models),
    ]
    for i, q in enumerate(QUESTIONS):
        cells = []
        for m in models:
            row = all_results[m][i]
            cells.append("PASS" if row["check_passed"] else "FAIL")
            cells.append(f"{row['tokens_per_sec']:.1f}")
        lines.append(f"| {q['id']} | {q['check_desc']} | " + " | ".join(cells) + " |")

    lines += ["", "## Full responses", ""]
    for i, q in enumerate(QUESTIONS):
        lines += [f"### {q['id']}", "", f"**Prompt:** {q['prompt']}", ""]
        for m in models:
            row = all_results[m][i]
            lines += [
                f"**{m}** (check: {'PASS' if row['check_passed'] else 'FAIL'}, "
                f"{row['word_count']} words, {row['tokens_per_sec']:.1f} tok/s)",
                "",
                *[f"> {ln}" if ln.strip() else ">" for ln in row["response"].splitlines()],
                "",
            ]

    lines += [
        "## My observations (fill in after reading the responses above)",
        "",
        "- **Accuracy / hallucination:** which model got the facts right? Did either invent details?",
        "- **Reasoning:** did the arithmetic work show correct steps, or just a lucky number?",
        "- **Instruction following:** did each model respect 'exactly 3 bullets' and the word limit?",
        "- **Style and verbosity:** which answers were clearer, and which rambled?",
        "- **Speed:** tokens/sec difference, and whether it mattered in practice.",
        "- **Overall:** which would I use for what, and why?",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local Ollama models on 3 questions.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    try:
        installed = list_models()
    except OllamaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1

    missing = [m for m in args.models if m not in installed]
    if missing:
        print(f"ERROR: not installed: {missing}. Installed: {installed or 'none'}")
        for m in missing:
            print(f"Run: ollama pull {m}")
        return 1

    all_results = {}
    for model in args.models:
        all_results[model] = run_model(model)

    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump({"system_prompt": SYSTEM_PROMPT, "options": OPTIONS, "results": all_results}, f, indent=2)
    with open("comparison_report.md", "w", encoding="utf-8") as f:
        f.write(build_report(all_results))
    print("\nSaved comparison_results.json and comparison_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
