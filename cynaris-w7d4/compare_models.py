"""W7D4 task 3 - llama3.2:3b vs qwen2.5:3b on the same 3 questions, same system prompt, same settings.

Usage:  python compare_models.py
Writes results/model_comparison.csv. Then fill the three rating columns by hand
(1-5) and write your observations in the README.
"""
import csv
from ollama_client import chat, installed_models

MODELS = ["llama3.2:3b", "qwen2.5:3b"]
SYSTEM = "You are a careful assistant. Answer directly and briefly."

QUESTIONS = [
    ("factual",     "What does the acronym RAG stand for in AI, and what problem does it address?"),
    ("reasoning",   "A model answers 17 of 20 test questions correctly. If the next 30 questions have the same accuracy, how many will it get right? Show the calculation."),
    ("instruction", "List exactly three advantages of running an LLM locally. Use a numbered list and no more than 12 words per item."),
]

missing = [m for m in MODELS if m not in installed_models()]
if missing:
    raise SystemExit("Not installed: " + ", ".join(missing) + "   (ollama pull <model>)")

rows = []
for model in MODELS:
    chat(model, "hi", system=SYSTEM, max_tokens=5)            # warm-up so load time does not skew the first question
    for kind, q in QUESTIONS:
        r = chat(model, q, system=SYSTEM, temperature=0.0)     # temperature 0 -> repeatable
        rows.append([model, kind, q, r["text"].replace("\n", " | "), r["wall_s"], r["out_tokens"], r["tokens_per_s"], "", "", ""])
        print(f"\n[{model}] {kind}\n{r['text']}\n   {r['wall_s']}s | {r['out_tokens']} tokens | {r['tokens_per_s']} tok/s")

with open("results/model_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "type", "question", "response", "wall_s", "out_tokens", "tokens_per_s",
                "accuracy_1to5", "instruction_following_1to5", "clarity_1to5"])
    w.writerows(rows)
print("\nSaved results/model_comparison.csv - fill the three rating columns, then document the differences.")
