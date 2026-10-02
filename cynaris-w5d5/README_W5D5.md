# W5D5: Week 5 Project, Local Q&A Bot (Ollama + ChromaDB)

A fully local question-answering bot over my own documents. Embeddings, retrieval and generation all run through Ollama and a persistent ChromaDB store, so no data leaves the machine. The bot answers only from retrieved context, cites its sources, and refuses when the documents don't contain the answer. Two models (`llama3.2:3b` and `qwen2.5:3b`) are compared on the same questions.

## What's in this folder

| File                         | Purpose                                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `qa_bot.py`                  | The bot: ingest docs, index in ChromaDB, retrieve, answer with a custom system prompt (interactive or one-shot) |
| `run_questions.py`           | Task 2: runs 5 test prompts through the bot and saves the results                                               |
| `compare_models.py`          | Task 3: runs 3 questions through both models with identical context and writes a report                         |
| `run_questions_results.json` | Raw results of the 5 test prompts                                                                               |
| `comparison_results.json`    | Raw answers, timings and token counts for both models                                                           |
| `comparison_report.md`       | Side-by-side report with a manual rating table                                                                  |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Mac/Linux: source .venv/bin/activate
pip install chromadb ollama pypdf

ollama pull nomic-embed-text     # embedding model
ollama pull llama3.2:3b          # default chat model
ollama pull qwen2.5:3b           # comparison model
```

Put at least one text-based PDF, TXT or MD file in a `docs/` folder. Ollama must be running locally.

## Usage

```bash
python qa_bot.py                          # interactive chat (type exit to quit)
python qa_bot.py -q "your question"       # one-shot answer
python qa_bot.py --model qwen2.5:3b       # use another chat model
python qa_bot.py --rebuild                # re-index the docs folder
python run_questions.py                   # Task 2: 5 test prompts
python compare_models.py                  # Task 3: model comparison
```

## How it works

1. **Ingest:** every PDF, TXT and MD file in `docs/` is split into 800-character chunks with 150 overlap. PDF chunks keep their page number.
2. **Index:** chunks are embedded with `nomic-embed-text` and stored in a cosine ChromaDB collection with `source` and `page` metadata. The index is built once and reused (`--rebuild` refreshes it).
3. **Retrieve:** the question is embedded with the same model and the top 3 chunks are fetched. Similarity = `1 - cosine distance`.
4. **Guardrail:** if no chunk reaches a similarity of 0.40, the bot replies "I don't know based on the documents." without calling the LLM.
5. **Answer:** the chunks and a custom system prompt go to the chat model at temperature 0. The system prompt says: answer only from the context, at most 4 sentences, reply with the exact refusal sentence if the answer isn't there, and cite sources like `[file.pdf p.2]`.

## Task 2: 5 test prompts (llama3.2:3b)

| #   | Question                                                                   | Result                                                                                                         | Retrieved (similarity)             |
| --- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 1   | What is the main topic of this document?                                   | HR workload and intelligent banking assistants (4.26s)                                                         | p.2 (0.54), p.2 (0.51), p.5 (0.49) |
| 2   | What problem does the study address?                                       | Streamlining HR workload with a RAG assistant for internal policy questions (22.85s)                           | p.2 (0.53), p.5 (0.52), p.2 (0.51) |
| 3   | What language is the data in, and what are the two phases of the research? | Two phases correct (Excel Q&A pairs, then unstructured PDFs); language answered as "natural language" (23.27s) | p.6 (0.62), p.5 (0.61), p.2 (0.60) |
| 4   | What does the input data of the RAG system comprise?                       | Employee queries plus structured internal data (19.52s)                                                        | p.2 (0.77), p.3 (0.65), p.2 (0.65) |
| 5   | Who won the 2010 FIFA World Cup? (out of scope)                            | Refused: "I don't know based on the documents." (0.05s, LLM not called)                                        | p.5 (0.40), p.6 (0.40), p.2 (0.37) |

Manual verification against the PDF:

- Q1, Q2, Q4 matched the cited text.
- Q3: the two phases are correct. [Check the PDF: the data is described as Bulgarian, so "natural language" answers only part of the question.]
- Q5: the guardrail blocked it. The out-of-scope scores (about 0.40) sit close to the threshold, while legitimate questions scored 0.49 to 0.77, so the margin is narrow.

## Task 3: llama3.2:3b vs qwen2.5:3b

Same retrieved context and system prompt for both models, with the guardrail off (`min_sim=0`) so the out-of-scope question reaches the LLM.

| Question                      | llama3.2:3b       | qwen2.5:3b        |
| ----------------------------- | ----------------- | ----------------- |
| Main topic                    | 8.51s, 4.0 tok/s  | 44.9s, 1.0 tok/s  |
| Language and two phases       | 28.82s, 1.6 tok/s | 22.63s, 1.6 tok/s |
| Out of scope (2010 World Cup) | 18.71s, 2.2 tok/s | 15.48s, 0.6 tok/s |

Observations:

- **Content:** both models gave essentially the same answer on the two in-document questions, including the same "natural language" reading of the language question, so that miss comes from the retrieved context, not from either model.
- **Out-of-scope behaviour:** qwen2.5:3b followed the instruction exactly ("I don't know based on the documents."). llama3.2:3b refused but then added an explanation with a citation, which broke the "reply exactly" rule.
- **Citations:** llama3.2:3b followed the citation rule in its answers; qwen2.5:3b did not include a citation in its second answer.
- **Speed:** timings are only indicative. Models alternate on every question, so each call can include model loading time, and tokens/s is computed over total time including retrieval. The runs were slow overall because they ran on CPU.
- Full answers and the manual rating table are in `comparison_report.md`.

## Limitations and trade-offs

- **Citations can be wrong.** In Q1, llama3.2:3b cited `[sample.pdf p.1]` although none of the retrieved chunks came from page 1. A 3B model can invent page numbers, so citations should be checked against the retrieval list printed under each answer.
- **Fixed threshold.** 0.40 is specific to `nomic-embed-text` and this document. A different embedding model or corpus needs a new threshold.
- **Chunking by characters** can split sentences. Overlap reduces the problem but doesn't remove it.
- **Latency:** a 3B model on CPU takes 5 to 45 seconds per answer. A GPU or a smaller context would speed it up.
- **Local vs cloud:** local models keep documents private and cost nothing per call, but are slower and less accurate than large hosted models.

## Self-review checklist

- [x] Code runs from a clean environment
- [x] Index builds from `docs/` and is reused on the next run
- [x] 5 test prompts run and checked against the PDF
- [x] Out-of-scope question is refused
- [x] llama3.2:3b and qwen2.5:3b compared on the same 3 questions
- [x] Rating table and observations filled in `comparison_report.md`
- [x] `chroma_db/`, `.venv/` and `docs/` are in `.gitignore`

## Author

Siriyala Nishar
