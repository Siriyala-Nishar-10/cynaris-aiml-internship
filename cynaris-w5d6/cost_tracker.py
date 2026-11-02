"""
W5D6 (Task 4): Cost tracker, tokens per request and daily total in INR
------------------------------------------------------------------------
Wraps the local Q&A bot so every request is logged to cost_log.csv with the
real token counts reported by Ollama (prompt_eval_count and eval_count).

The local model costs nothing per call, so each request is also priced as if
it had gone to a paid API (GPT-4o mini and GPT-4o). The daily report shows the
actual spend (INR 0 for local inference) next to what the same traffic would
have cost, which is the number the routing strategy is built on.

  cost = input_tokens x input_price + output_tokens x output_price

Prices are USD per 1M tokens from the W5D6 lesson table. Update USD_INR to the
current exchange rate before you quote any figure.

Caveat: Ollama may reuse a cached prompt prefix, in which case prompt_eval_count
can be lower than the full prompt size.

Output:
  cost_log.csv   one row per request

Run:  python cost_tracker.py demo      run 5 questions through the bot and log them
      python cost_tracker.py report    print the daily totals from cost_log.csv
Needs: pip install ollama chromadb pypdf
       qa_bot.py and a docs/ folder (copy both from your W5D5 folder)
"""

import csv
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ollama

LOG_FILE = "cost_log.csv"
USD_INR = 88.0   # update to the current exchange rate

# USD per 1M tokens (input, output), from the W5D6 lesson table
REFERENCE_PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (5.00, 15.00)}

FIELDS = ["timestamp", "date", "model", "prompt_tokens", "completion_tokens", "seconds",
          "inr_equiv_gpt4o_mini", "inr_equiv_gpt4o"]

DEMO_QUESTIONS = [
    "What is the main topic of this document?",
    "What problem does the study address?",
    "What are the two phases of the research?",
    "What does the input data of the RAG system comprise?",
    "Who won the 2010 FIFA World Cup?",
]


def cost_inr(prompt_tokens: int, completion_tokens: int, api_model: str) -> float:
    """What this request would cost on a paid API, in INR."""
    price_in, price_out = REFERENCE_PRICES[api_model]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000 * USD_INR


def log_request(model: str, prompt_tokens: int, completion_tokens: int, seconds: float) -> dict:
    """Append one request to the CSV log and return the logged row."""
    now = datetime.now()
    row = {
        "timestamp": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "seconds": round(seconds, 2),
        "inr_equiv_gpt4o_mini": round(cost_inr(prompt_tokens, completion_tokens, "gpt-4o-mini"), 6),
        "inr_equiv_gpt4o": round(cost_inr(prompt_tokens, completion_tokens, "gpt-4o"), 6),
    }
    new_file = not Path(LOG_FILE).exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return row


def tracked_chat(model: str, messages: list[dict], options: dict | None = None) -> tuple[str, dict]:
    """ollama.chat() plus logging. Returns (answer text, logged row)."""
    start = time.perf_counter()
    reply = ollama.chat(model=model, messages=messages, options=options or {"temperature": 0})
    seconds = time.perf_counter() - start
    row = log_request(
        model,
        getattr(reply, "prompt_eval_count", 0) or 0,
        getattr(reply, "eval_count", 0) or 0,
        seconds,
    )
    return reply["message"]["content"].strip(), row


def tracked_answer(col, question: str) -> tuple[str, dict]:
    """Same prompt as qa_bot.answer() (top-3 chunks), but logged."""
    from qa_bot import CHAT_MODEL, SYSTEM_PROMPT, retrieve

    hits = retrieve(col, question, 3)
    context = "\n\n".join(f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return tracked_chat(CHAT_MODEL, messages)


def daily_report(path: str = LOG_FILE) -> dict:
    """Print and return daily totals: requests, tokens and INR (actual and API-equivalent)."""
    if not Path(path).exists():
        sys.exit(f"{path} not found. Run 'python cost_tracker.py demo' first.")

    days = defaultdict(lambda: {"requests": 0, "prompt": 0, "completion": 0, "mini": 0.0, "gpt4o": 0.0})
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = days[r["date"]]
            d["requests"] += 1
            d["prompt"] += int(r["prompt_tokens"])
            d["completion"] += int(r["completion_tokens"])
            d["mini"] += float(r["inr_equiv_gpt4o_mini"])
            d["gpt4o"] += float(r["inr_equiv_gpt4o"])

    print(f"\n{'date':<12}{'requests':>9}{'prompt tok':>12}{'output tok':>12}"
          f"{'actual INR':>12}{'GPT-4o mini INR':>17}{'GPT-4o INR':>12}")
    for date_, d in sorted(days.items()):
        print(f"{date_:<12}{d['requests']:>9}{d['prompt']:>12}{d['completion']:>12}"
              f"{0.0:>12.2f}{d['mini']:>17.4f}{d['gpt4o']:>12.4f}")
    print(f"\nActual cost is INR 0 (local inference). The API columns show what the same "
          f"traffic would cost (USD->INR {USD_INR}).")
    return dict(days)


def run_demo() -> None:
    """Send the demo questions through the bot, logging each one."""
    from qa_bot import build_index

    col = build_index()
    for q in DEMO_QUESTIONS:
        _, row = tracked_answer(col, q)
        print(f"{q}\n  prompt={row['prompt_tokens']} out={row['completion_tokens']} "
              f"{row['seconds']}s | API-equiv: mini INR {row['inr_equiv_gpt4o_mini']:.4f}, "
              f"GPT-4o INR {row['inr_equiv_gpt4o']:.4f}")
    daily_report()
    print("VERIFY: check that prompt tokens are in the hundreds to low thousands and that GPT-4o costs far more than GPT-4o mini")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if command == "demo":
        run_demo()
    elif command == "report":
        daily_report()
    else:
        sys.exit("Usage: python cost_tracker.py [demo|report]")
