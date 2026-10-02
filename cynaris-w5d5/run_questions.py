"""
W5D5: Test the Q&A bot with 5 prompts (custom system prompt, one model)
------------------------------------------------------------------------
Runs 5 questions through qa_bot.answer() and saves everything to JSON, so the
results can be checked by hand. The set is chosen to test different things:

  1. Summary question        -> does it capture the main topic?
  2. Problem statement       -> does it pull a specific point from the text?
  3. Two-part factual        -> does it answer BOTH parts correctly?
  4. Detail from a list      -> does it stay close to the source wording?
  5. Out-of-scope question   -> does it refuse instead of hallucinating?

Edit QUESTIONS to match your own documents. After running, check each answer
against the cited page and write VERIFIED / WRONG notes in your README.

Run:  python run_questions.py
      python run_questions.py --model qwen2.5:3b
Needs: qa_bot.py in the same folder, plus a docs/ folder with a PDF
"""

import argparse
import json

from qa_bot import CHAT_MODEL, answer, build_index

OUTPUT_FILE = "run_questions_results.json"

QUESTIONS = [
    "What is the main topic of this document?",
    "What problem does the study address?",
    "What language is the data in, and what are the two phases of the research?",
    "What does the input data of the RAG system comprise?",
    "Who won the 2010 FIFA World Cup?",   # out of scope on purpose
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=CHAT_MODEL)
    args = p.parse_args()

    col = build_index()
    results = []

    for n, q in enumerate(QUESTIONS, start=1):
        r = answer(col, q, args.model)
        results.append(r)
        pages = ", ".join(f"{h['source']} p.{h['page']} ({h['sim']:.2f})" for h in r["sources"])
        print(f"\n[{n}/{len(QUESTIONS)}] {q}")
        print(f"Answer ({r['seconds']}s, LLM called: {r['llm_called']}): {r['answer']}")
        print(f"Retrieved: {pages}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to {OUTPUT_FILE}")
    print("VERIFY: check each answer against the cited PDF page; the last one should be a refusal")


if __name__ == "__main__":
    main()
