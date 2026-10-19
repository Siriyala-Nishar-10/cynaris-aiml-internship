# W6D5: Week 6 Project — Document Chatbot with LangChain

A document chatbot that answers questions about a PDF using a LangChain RAG chain, conversation memory and ChromaDB, plus a two-tool agent. Built with LangChain, ChromaDB and local Ollama models (`llama3.2`, `qwen2.5:3b`, `nomic-embed-text`).

Test document: `sample.pdf`, an IEEE paper on a RAG framework for banking assistants.

## Contents

| File               | Task | What it does                                                                                                                                                                |
| ------------------ | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `doc_chatbot.py`   | 1, 2 | PDF -> chunks -> ChromaDB (cosine) -> `PromptTemplate -> OllamaLLM -> StrOutputParser` RAG chain tested on 5 questions, then a memory-enabled chatbot verified over 5 turns |
| `agent_tools.py`   | 3    | Tool-calling agent (`create_agent`) with a web-search stub and a safe calculator, run on 3 tasks with an automatic pass/fail check                                          |
| `requirements.txt` |      | Dependencies                                                                                                                                                                |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2
ollama pull qwen2.5:3b
```

Ollama must be running. Copy `sample.pdf` into this folder.

## Usage

```bash
python doc_chatbot.py sample.pdf
python agent_tools.py
```

Optional environment variables:

```powershell
$env:CHAT_MODEL  = "llama3.2:1b"    # chat model for the chatbot (must be pulled first; default: llama3.2)
$env:AGENT_MODEL = "llama3.2"        # agent model (default: qwen2.5:3b)
```

## How it works

**Ingestion.** `pypdf` extracts text page by page. `RecursiveCharacterTextSplitter` (500 characters, 50 overlap) creates 63 chunks from 6 pages, each tagged with its page number. Chunks are embedded with `nomic-embed-text` and stored in a cosine-space ChromaDB collection.

**Task 1: RAG chain.** The chain retrieves the top-5 chunks for a question, fills a prompt that restricts the model to the retrieved context, calls the LLM and parses the output to a string. If the context lacks the answer, the model must say "I don't know based on the document."

**Task 2: memory.** The chatbot keeps a conversation buffer, a plain list of human and AI messages. This is what the deprecated `ConversationBufferMemory` stored. Before retrieval, follow-up questions such as "How does the system do that?" are rewritten into standalone questions so the search targets the real topic. The rewrite is validated: it must be one line ending in "?" and under 30 words. If the small model answers instead of rewriting, the code falls back to the previous question plus the current one.

**Task 3: agent.** `langchain.agents.create_agent` runs the ReAct loop (reason, act, observe). The calculator parses expressions with `ast` instead of `eval()`. The search tool is a stub with canned answers. A system prompt tells the model to use `web_search` for facts and the calculator only when a calculation is requested.

## Results

### Task 1: RAG chain (5 questions)

| Question                              | Result                                                                                                                                 |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Purpose of the RAG system             | Reasonable, but partly drawn from the conclusion ("minimize risk and maximize organizational value")                                   |
| Input data of the RAG system          | Correct (employee queries plus structured internal data)                                                                               |
| Components of the system architecture | **Wrong.** Listed row labels from a comparison table (Embedding, Splitter, Metadata, Paraphrase) instead of the two-stage architecture |
| Which workload does it reduce         | Correct (HR workload)                                                                                                                  |
| Capital of Australia (off-topic)      | Correct: "I don't know based on the document."                                                                                         |

### Task 2: memory (5 turns)

Memory works. Turn 4 ("What was the first thing I asked you?") correctly recalled turn 1, and turn 5 summarised the conversation. The history assertion passed (10 messages stored).

| Run                     | Observation                                                                                                                                                                                                                                                                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| First run               | Vague questions returned "I don't know". The rewrite step answered questions instead of rewriting them, including inventing an answer unrelated to the PDF in turn 2, and retrieval then ran on that text. Turn 3 wrongly said the document doesn't explain how the system works.                                                                  |
| After fixes (final run) | Specific default questions, stricter rewrite prompt with validation and fallback, top-5 retrieval. Turn 3 now correctly describes the pipeline (embed the prompt, query the vector database, merge prompt, context and instructions, generate a compliance-checked answer). The final run reproduced the earlier output exactly (`temperature=0`). |

### Task 3: agent (`llama3.2`)

First run with the strict check (required tools called, no calculator errors, correct value in the final answer):

| Task                        | Tools called           | Check | Outcome                                                                                                                                                                            |
| --------------------------- | ---------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `23 * 47 + 130`             | calculator             | PASS  | Correct (1211)                                                                                                                                                                     |
| Height of Mount Everest     | web_search             | FAIL  | Right tool and correct observation (8,849 m), but the final answer said "That's a height, not a 'how many times taller' question"                                                  |
| Everest vs a 2-metre person | web_search, calculator | FAIL  | The calculator received the text `(Mount Everest height) / 2` and returned an error. The final answer was raw tool-call JSON containing a wrong number (`8499 / 2`), not an answer |

Results vary between runs even at `temperature=0`. In an earlier run, Task 3 ended with the model computing 8,849 / 2 = 4,424.5 itself after the same calculator error. Before the system prompt was added, the agent skipped the calculator entirely and subtracted instead of dividing. The system prompt improved tool selection, but `llama3.2` (3B) is still unreliable at forming tool arguments and at writing a final answer from tool results.

### Task 3: agent (`qwen2.5:3b`)

Same code and strict check; only `AGENT_MODEL` changed. Two runs gave different results:

| Task                        | Run A                                   | Run B                                                                                                                   |
| --------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `23 * 47 + 130`             | PASS (calculator, 1211)                 | PASS (calculator, 1211)                                                                                                 |
| Height of Mount Everest     | PASS (web_search, 8,849 m)              | **FAIL.** After the search, the model invented a calculation (`8849 / 1.75`) and answered a question that was not asked |
| Everest vs a 2-metre person | PASS (search, then `8849 / 2` = 4424.5) | PASS (same)                                                                                                             |

`qwen2.5:3b` is clearly better than `llama3.2` at forming tool arguments (valid numeric expressions), but it is not deterministic: 3 of 3 in one run, 2 of 3 in the next. Run B's failure pointed at the system prompt, which told the model to "search first, then calculate" and that "how many times taller means divide", nudging it to calculate after every search. The prompt was then refined: the calculator is used only when a calculation is requested, and a plain fact question needs one search and no extra calculations.

**Run C (refined prompt): 3 of 3 PASS.** Task 2 answered with one search and no extra calculation, though its final answer still added filler text. A single run is not proof that the variance is gone, since runs A and B differed with identical code.

`qwen2.5:3b` is the default agent model.

## Limitations

- **Retrieval misses.** Vague questions and some specific ones (architecture components) retrieve the wrong chunks. Fixed-size chunking ignores document structure, and tables get mixed with prose.
- **Footer noise.** Every PDF page carries an IEEE license footer that is embedded into chunks. The source paper itself notes that data cleaning matters for retrieval quality.
- **Small-model errors.** `llama3.2` expanded RAG as "Retrieval and Generation" (it is Retrieval-Augmented Generation), and in earlier W6D3 runs made other factual slips.
- **Agent reliability depends on the model.** `llama3.2` (3B) failed 2 of 3 tasks, while `qwen2.5:3b` scored 3 of 3 and 2 of 3 in two runs. Even the better model is not deterministic, and system-prompt wording can push it into unnecessary tool calls.
- **Fallback retrieval.** For meta questions like "What was the first thing I asked you?", the fallback still searches the document. This is harmless but wasteful.
- **Memory is in-process only** and is lost when the script exits.
- The search tool is a stub with three stored answers.

## Week 6 summary

| Day | Topic                  | What was built                                                                |
| --- | ---------------------- | ----------------------------------------------------------------------------- |
| D1  | LangChain fundamentals | Chains and prompts                                                            |
| D2  | Memory and persistence | Memory types, SQLite-backed chat history                                      |
| D3  | Tools and agents       | Chain with memory, two-tool agent                                             |
| D4  | RAG pipeline           | ChromaDB vector store, cosine search, metadata filtering, PDF RAG with Ollama |
| D5  | Project                | Document chatbot combining retrieval, a RAG chain, memory and an agent        |

Key lessons: retrieval quality decides RAG quality, small local models need validation and guardrails around them, and a ReAct agent is only as reliable as the model driving it.
