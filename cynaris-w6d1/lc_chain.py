"""
W6D1 (Task 1): LangChain chain, PromptTemplate -> Ollama LLM -> OutputParser
------------------------------------------------------------------------------
A chain is a pipeline of small, reusable steps joined with the `|` operator
(LangChain Expression Language, LCEL):

  PromptTemplate   fills {placeholders} and builds the final prompt text
  OllamaLLM        sends that text to the local model through Ollama
  OutputParser     turns the model's raw text into the type the code needs

Compared with a single LLM call, each step can be swapped, tested and reused
on its own, and the whole chain exposes the same invoke / batch / stream API.

What this script does:

  1. Builds the chain and shows what each step produces for the first input.
  2. Runs the chain on 5 different inputs (topic + audience) and prints the
     answers with timings.
  3. Builds a second chain with a list parser, to show an output parser
     doing real work: the reply becomes a Python list, not a string.

Output:
  chain_results.json   inputs, answers and timings

Run:  python lc_chain.py
      python lc_chain.py --model qwen2.5:3b
Needs: pip install langchain langchain-ollama
       ollama pull llama3.2:3b   (Ollama must be running)
"""

import argparse
import json
import time

from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

MODEL = "llama3.2:3b"
OUTPUT_FILE = "chain_results.json"

TEMPLATE = (
    "You are a clear technical teacher. "
    "Explain {topic} to {audience} in exactly two sentences."
)

INPUTS = [
    {"topic": "a vector database", "audience": "a complete beginner"},
    {"topic": "cosine similarity", "audience": "a high-school student"},
    {"topic": "retrieval-augmented generation", "audience": "a product manager"},
    {"topic": "quantisation of language models", "audience": "a software engineer"},
    {"topic": "semantic caching", "audience": "a finance analyst"},
]


def build_chain(model: str):
    """PromptTemplate -> Ollama LLM -> StrOutputParser, joined with the | operator."""
    prompt = PromptTemplate.from_template(TEMPLATE)
    llm = OllamaLLM(model=model, temperature=0)   # temperature 0: repeatable answers
    return prompt | llm | StrOutputParser()


def build_list_chain(model: str):
    """Same idea with a list parser: the model's reply is parsed into a Python list."""
    parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate(
        template="List exactly 4 key terms related to {topic}.\n{format_instructions}",
        input_variables=["topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | OllamaLLM(model=model, temperature=0) | parser


def show_steps(model: str, sample: dict) -> None:
    """Run the three steps one at a time so each one's output can be seen."""
    prompt = PromptTemplate.from_template(TEMPLATE)
    llm = OllamaLLM(model=model, temperature=0)
    parser = StrOutputParser()

    prompt_value = prompt.invoke(sample)
    print("STEP 1  PromptTemplate output (text sent to the model):")
    print(f"        {prompt_value.to_string()}")
    raw = llm.invoke(prompt_value)
    print("STEP 2  LLM raw output:")
    print(f"        {raw!r}")
    print("STEP 3  OutputParser result:")
    print(f"        {parser.invoke(raw)!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    print(f"Model: {args.model}\n")
    show_steps(args.model, INPUTS[0])

    chain = build_chain(args.model)
    print("\n" + "=" * 70 + f"\nRunning the chain on {len(INPUTS)} inputs\n")

    results = []
    for n, inp in enumerate(INPUTS, start=1):
        start = time.perf_counter()
        answer = chain.invoke(inp)
        seconds = round(time.perf_counter() - start, 2)
        results.append({"input": inp, "answer": answer, "seconds": seconds})
        print(f"[{n}/{len(INPUTS)}] topic={inp['topic']!r}, audience={inp['audience']!r} ({seconds}s)")
        print(f"      {answer}\n")

    print("=" * 70 + "\nList output parser demo\n")
    terms = build_list_chain(args.model).invoke({"topic": INPUTS[0]["topic"]})
    print(f"type={type(terms).__name__}, items={len(terms)}: {terms}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "results": results, "list_parser_demo": terms},
                  f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUTPUT_FILE}")
    print("VERIFY: each answer has two sentences and fits its audience; the list demo prints a Python list")


if __name__ == "__main__":
    main()
