"""
W5D6 (Task 1): Token audit of the local Q&A bot's prompts
-----------------------------------------------------------
Measures how many tokens the bot's prompt contains at 5 different sizes, and
what that prompt WOULD cost on a paid API. The local model itself is free per
call, but the token counts show what moving this workload to an API costs.

How the 5 sizes are made: the bot's real prompt (system prompt + retrieved
context + question) is built with 1, 3, 5, 8 and 12 retrieved chunks.

  cost = input_tokens x input_price + output_tokens x output_price

Tokens are counted with tiktoken (GPT-4o's tokenizer, o200k_base). Other
models tokenize differently, so treat the figures as close estimates. With
--ollama the script also asks Ollama for the real prompt token count of the
local model, so the two tokenizers can be compared.

Prices are the ones from the W5D6 lesson table (USD per 1M tokens).

Output:
  token_audit.csv   one row per prompt size, used by build_audit_xlsx.py

Run:  python token_audit.py
      python token_audit.py --ollama        also count with llama3.2:3b (slow on CPU)
Needs: pip install tiktoken chromadb ollama pypdf
       qa_bot.py and a docs/ folder (copy both from your W5D5 folder)
"""

import argparse
import csv

import ollama
import tiktoken

from qa_bot import CHAT_MODEL, SYSTEM_PROMPT, build_index, retrieve

QUESTION = "What does the input data of the RAG system comprise?"
K_VALUES = [1, 3, 5, 8, 12]          # retrieved chunks -> 5 prompt sizes
OUTPUT_TOKENS = 150                  # assumed answer length for the cost estimate
USD_INR = 88.0                       # update to the current exchange rate
OUTPUT_FILE = "token_audit.csv"

# USD per 1M tokens (input, output), from the W5D6 lesson table
PRICES = {
    "GPT-4o": (5.00, 15.00),
    "Claude Sonnet 4": (3.00, 15.00),
    "Gemini 2.0 Flash": (0.10, 0.40),
    "GPT-4o mini": (0.15, 0.60),
}


def build_prompt(hits: list[dict], question: str) -> str:
    """Same prompt layout as qa_bot.answer(): system prompt + context + question."""
    context = "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"


def cost_inr(in_tokens: int, out_tokens: int, model: str) -> float:
    """Per-request cost in INR for one of the priced API models."""
    price_in, price_out = PRICES[model]
    return (in_tokens * price_in + out_tokens * price_out) / 1_000_000 * USD_INR


def ollama_prompt_tokens(prompt: str) -> int | None:
    """Ask Ollama how many prompt tokens the local model counts (generates 1 token)."""
    try:
        reply = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
                            options={"num_predict": 1})
        return getattr(reply, "prompt_eval_count", None)
    except Exception as exc:
        print(f"  ollama count failed: {exc}")
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ollama", action="store_true", help="also count tokens with the local model")
    args = p.parse_args()

    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")

    col = build_index()
    hits_all = retrieve(col, QUESTION, max(K_VALUES))

    rows = []
    for k in K_VALUES:
        prompt = build_prompt(hits_all[:k], QUESTION)
        tokens = len(enc.encode(prompt))
        local = ollama_prompt_tokens(prompt) if args.ollama else None
        rows.append({"label": f"{k} chunks", "k_chunks": k, "chars": len(prompt),
                     "tokens_tiktoken": tokens, "tokens_ollama": local if local is not None else ""})

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nQuestion: {QUESTION}")
    print(f"Assumed output: {OUTPUT_TOKENS} tokens | USD->INR {USD_INR}\n")
    header = f"{'size':<10}{'chars':>7}{'tokens':>8}" + "".join(f"{m:>18}" for m in PRICES)
    print(header + "   (cost per request, INR)")
    for r in rows:
        costs = "".join(f"{cost_inr(r['tokens_tiktoken'], OUTPUT_TOKENS, m):>18.4f}" for m in PRICES)
        print(f"{r['label']:<10}{r['chars']:>7}{r['tokens_tiktoken']:>8}{costs}")
    print(f"\nSaved {OUTPUT_FILE}")
    print("VERIFY: tokens should grow roughly linearly with the number of chunks (about 4 characters per token)")


if __name__ == "__main__":
    main()
