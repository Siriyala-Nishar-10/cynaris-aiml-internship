"""W7D4 task 2 - call the Ollama API with a custom system prompt and test 5 prompts.

Usage:  python ollama_prompts.py            (default model llama3.2:3b)
        python ollama_prompts.py qwen2.5:3b
Writes results/prompts_<model>.csv
"""
import csv, sys
from ollama_client import chat, installed_models

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"

SYSTEM = ("You are a concise assistant for an AI/ML intern building RAG systems. "
          "Answer in at most 4 sentences. If you are not sure, say so instead of guessing.")

PROMPTS = [
    "Explain retrieval-augmented generation in two sentences.",
    "What is the difference between BM25 and dense retrieval?",
    "Summarise in one sentence: A hybrid retriever merges keyword and embedding results with reciprocal rank fusion.",
    "Extract the numbers as JSON: The system reached 85% correct answers over 50 queries.",
    "What was the exact closing price of Reliance Industries on 3 March 2024?",   # should admit uncertainty
]

if MODEL not in installed_models():
    sys.exit(f"{MODEL} is not installed. Run: ollama pull {MODEL}")

rows = []
for i, p in enumerate(PROMPTS, 1):
    r = chat(MODEL, p, system=SYSTEM)
    rows.append([i, p, r["text"].replace("\n", " "), r["wall_s"], r["out_tokens"], r["tokens_per_s"]])
    print(f"\nP{i}: {p}\n -> {r['text']}\n    {r['wall_s']}s | {r['out_tokens']} tokens | {r['tokens_per_s']} tok/s")

path = f"results/prompts_{MODEL.replace(':', '_')}.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["prompt_id", "prompt", "response", "wall_s", "out_tokens", "tokens_per_s"])
    w.writerows(rows)
print(f"\nSaved {path}")
