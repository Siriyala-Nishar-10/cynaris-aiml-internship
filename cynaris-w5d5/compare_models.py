"""
W5D5: Compare llama3.2:3b vs qwen2.5:3b on the same 3 questions
----------------------------------------------------------------
Both models get EXACTLY the same retrieved context and the same system
prompt, so any difference in the answers comes from the model, not the
retrieval. Retrieval runs once per question and is shared.

The 3 questions test different skills:

  1. Summary          -> synthesis across chunks
  2. Specific detail  -> accuracy on a concrete fact
  3. Out of scope     -> does the model refuse, or hallucinate?

The similarity guardrail is switched OFF here (min_sim=0) so that the
out-of-scope question still reaches the LLM. That is what lets us see how
each model behaves when the context does not contain the answer.

Output:
  comparison_results.json   raw answers, timings, token counts
  comparison_report.md      side-by-side report with an empty rating table
                            to fill in by hand after reading the answers

Run:  python compare_models.py
Needs: ollama pull llama3.2:3b && ollama pull qwen2.5:3b
       qa_bot.py in the same folder, plus a docs/ folder with a PDF
"""

import json

from qa_bot import answer, build_index

MODELS = ["llama3.2:3b", "qwen2.5:3b"]

QUESTIONS = [
    "What is the main topic of this document?",
    "What language is the data in, and what are the two phases of the research?",
    "Who won the 2010 FIFA World Cup?",   # out of scope on purpose
]

JSON_FILE = "comparison_results.json"
REPORT_FILE = "comparison_report.md"


def tokens_per_second(r: dict) -> str:
    """Rough speed figure: generated tokens / total seconds (includes retrieval)."""
    if r["eval_tokens"] and r["seconds"]:
        return f"{r['eval_tokens'] / r['seconds']:.1f}"
    return "n/a"


def write_report(results: dict) -> None:
    """Write a markdown report: a speed table, the answers, and a blank rating table."""
    lines = ["# W5D5: llama3.2:3b vs qwen2.5:3b", "",
             "Same retrieved context and system prompt for both models (min_sim=0).", ""]

    for q in QUESTIONS:
        lines += [f"## Q: {q}", "", "| Model | Seconds | Tokens | Tokens/s |", "|---|---|---|---|"]
        for m in MODELS:
            r = results[m][q]
            lines.append(f"| {m} | {r['seconds']} | {r['eval_tokens']} | {tokens_per_second(r)} |")
        lines.append("")
        for m in MODELS:
            lines += [f"**{m}**", "", "> " + results[m][q]["answer"].replace("\n", "\n> "), ""]

    lines += ["## Manual rating (fill in after checking against the PDF)", "",
              "| Question | Model | Correct? (1-5) | Grounded in context? (1-5) | Concise? (1-5) |",
              "|---|---|---|---|---|"]
    for i, _ in enumerate(QUESTIONS, start=1):
        for m in MODELS:
            lines.append(f"| Q{i} | {m} |  |  |  |")
    lines += ["", "## Observations", "", "- Quality: ", "- Speed: ",
              "- Out-of-scope behaviour: ", "- Which model I would use and why: ", ""]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    col = build_index()
    results = {m: {} for m in MODELS}

    for q in QUESTIONS:
        print(f"\nQ: {q}")
        for m in MODELS:
            r = answer(col, q, model=m, min_sim=0)
            results[m][q] = r
            print(f"  [{m}] {r['seconds']}s | {tokens_per_second(r)} tok/s")
            print(f"    {r['answer'][:200]}")

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    write_report(results)
    print(f"\nSaved {JSON_FILE} and {REPORT_FILE}")
    print("VERIFY: read both answers for each question, check them against the PDF, "
          "then fill in the rating table and observations in the report")


if __name__ == "__main__":
    main()
