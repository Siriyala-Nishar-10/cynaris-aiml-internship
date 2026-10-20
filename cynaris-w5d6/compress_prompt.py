"""
W5D6 (Task 2): LLMLingua prompt compression at rate=0.4, quality vs full
--------------------------------------------------------------------------
Takes a RAG context of about 2,000 tokens, compresses it with LLMLingua to
40% of its tokens (rate=0.4), then asks the SAME local model the SAME
question with the full and the compressed context.

Quality is checked two ways:

  1. Keyword recall: what fraction of the key facts you expect (EXPECTED)
     appear in the answer. This is a rough automatic score.
  2. Manual check: both answers are printed side by side, compare them to the
     PDF yourself. The keyword score alone is not proof of quality.

Uses LLMLingua-2, a small BERT-based compressor that runs on CPU (the first
run downloads the model, a few hundred MB). The lesson's `compress_prompt`
needs a 7B compressor; LLMLingua-2 gives the same API idea at a fraction of
the size.

Output:
  compression_results.json   tokens before/after, answers, scores, summary
                             (read by build_audit_xlsx.py)

Run:  python compress_prompt.py
Needs: pip install llmlingua tiktoken chromadb ollama pypdf
       qa_bot.py and a docs/ folder (copy both from your W5D5 folder)
Note: each question makes 2 LLM calls with a ~2,000 token prompt, which is
      slow on CPU (allow several minutes).
"""

import json
import time

import ollama
import tiktoken
from llmlingua import PromptCompressor

from qa_bot import CHAT_MODEL, SYSTEM_PROMPT, build_index, retrieve

RATE = 0.4                  # keep 40% of the tokens
TARGET_TOKENS = 2000        # size of the uncompressed context
MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
OUTPUT_FILE = "compression_results.json"

# (question, key facts that a correct answer should contain). EDIT for your PDF.
EXPECTED = [
    ("What is the main topic of this document?", ["HR", "banking", "RAG"]),
    ("What are the two phases of the research?", ["Excel", "PDF"]),
    ("What does the input data of the RAG system comprise?", ["queries", "structured"]),
]


def build_context(col, question: str, enc) -> str:
    """Add the most similar chunks until the context reaches about TARGET_TOKENS."""
    parts, total = [], 0
    for h in retrieve(col, question, 20):
        piece = f"[{h['source']} p.{h['page']}] {h['text']}"
        parts.append(piece)
        total += len(enc.encode(piece))
        if total >= TARGET_TOKENS:
            break
    return "\n\n".join(parts)


def ask(context: str, question: str) -> tuple[str, float]:
    """Ask the local model with the bot's system prompt; returns (answer, seconds)."""
    start = time.perf_counter()
    reply = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        options={"temperature": 0},
    )
    return reply["message"]["content"].strip(), round(time.perf_counter() - start, 1)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Fraction of the expected keywords found in the answer (case-insensitive)."""
    return round(sum(k.lower() in answer.lower() for k in keywords) / len(keywords), 2)


def main() -> None:
    enc = tiktoken.get_encoding("o200k_base")
    col = build_index()
    # device_map="cpu": LLMLingua defaults to CUDA, which fails on a CPU-only PyTorch build
    compressor = PromptCompressor(model_name=MODEL_NAME, use_llmlingua2=True, device_map="cpu")

    results = []
    for question, keywords in EXPECTED:
        context = build_context(col, question, enc)
        out = compressor.compress_prompt_llmlingua2(
            context, rate=RATE, force_tokens=["\n", ".", "?", "!"], drop_consecutive=True
        )
        compressed = out["compressed_prompt"]
        before, after = len(enc.encode(context)), len(enc.encode(compressed))

        full_answer, full_s = ask(context, question)
        comp_answer, comp_s = ask(compressed, question)

        r = {
            "question": question,
            "tokens_before": before, "tokens_after": after,
            "kept_fraction": round(after / before, 3),
            "full_answer": full_answer, "full_seconds": full_s,
            "full_recall": keyword_recall(full_answer, keywords),
            "compressed_answer": comp_answer, "compressed_seconds": comp_s,
            "compressed_recall": keyword_recall(comp_answer, keywords),
        }
        results.append(r)

        print(f"\nQ: {question}")
        print(f"  tokens: {before} -> {after} (kept {r['kept_fraction']:.0%})")
        print(f"  FULL       ({full_s}s, recall {r['full_recall']}): {full_answer}")
        print(f"  COMPRESSED ({comp_s}s, recall {r['compressed_recall']}): {comp_answer}")

    summary = {
        "rate": RATE,
        "avg_keep_ratio": round(sum(r["kept_fraction"] for r in results) / len(results), 3),
        "avg_full_recall": round(sum(r["full_recall"] for r in results) / len(results), 2),
        "avg_compressed_recall": round(sum(r["compressed_recall"] for r in results) / len(results), 2),
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"\nSummary: {summary}")
    print(f"Saved {OUTPUT_FILE}")
    print("VERIFY: compare each pair of answers against the PDF; note any fact the compressed answer lost")


if __name__ == "__main__":
    main()