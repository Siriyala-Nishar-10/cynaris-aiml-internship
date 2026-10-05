# W6D1: LangChain Fundamentals, Chains, Memory & Agents

Three small LangChain programs running on a local model through Ollama: a prompt-to-parser chain, a chatbot with conversation memory, and an agent that chooses between two tools. All three use `llama3.2:3b`, and the agent is also run with `qwen2.5:3b` for comparison.

## What's in this folder

| File                                                              | Purpose                                                                           |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `lc_chain.py`                                                     | Task 1: `PromptTemplate -> OllamaLLM -> OutputParser` chain, tested with 5 inputs |
| `lc_memory.py`                                                    | Task 2: conversation memory over 5 turns, with a no-memory control                |
| `lc_agent.py`                                                     | Task 3: agent with 2 tools (web search stub + calculator), 3 tasks                |
| `requirements.txt`                                                | `langchain`, `langchain-ollama`                                                   |
| `chain_results.json`, `memory_results.json`                       | Saved outputs of tasks 1 and 2                                                    |
| `agent_results_llama3.2_3b.json`, `agent_results_qwen2.5_3b.json` | Saved outputs of task 3 (final prompt)                                            |
| `agent_results_llama3.2_3b_v1.json`                               | First agent run, with the original prompt (baseline)                              |
| `evidence/`                                                       | Screenshots of outputs                                                            |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Mac/Linux: source .venv/bin/activate
pip install langchain langchain-ollama

ollama pull llama3.2:3b
ollama pull qwen2.5:3b           # only for the agent comparison
```

Ollama must be running locally.

## Usage

```bash
python lc_chain.py
python lc_memory.py
python lc_agent.py                       # llama3.2:3b
python lc_agent.py --model qwen2.5:3b
```

## Task 1: chain (PromptTemplate -> Ollama LLM -> OutputParser)

The chain is built with the `|` operator (LangChain Expression Language):

```python
prompt | OllamaLLM(model="llama3.2:3b", temperature=0) | StrOutputParser()
```

The script first runs the three steps one at a time for the first input, so each step's output is visible: the filled prompt text, the raw model output, and the parsed result. It then runs the chain on 5 inputs (topic and audience).

| #   | Topic                           | Audience              | Seconds |
| --- | ------------------------------- | --------------------- | ------- |
| 1   | a vector database               | a complete beginner   | 8.56    |
| 2   | cosine similarity               | a high-school student | 9.05    |
| 3   | retrieval-augmented generation  | a product manager     | 9.76    |
| 4   | quantisation of language models | a software engineer   | 9.76    |
| 5   | semantic caching                | a finance analyst     | 7.48    |

A second chain with `CommaSeparatedListOutputParser` shows a parser doing real work: the reply was parsed into a Python `list` of 4 items (`query`, `indexing`, `similarity search`, `faceting`).

Manual check of the answers:

- 4 of 5 answers are exactly two sentences. Answer 2 starts with a preamble ("Here's an explanation of cosine similarity in two sentences:") even though the prompt asked for exactly two sentences.
- Answers 1, 3 and 4 are accurate. Answer 2 says the cosine similarity value is between 0 and 1, but the range is -1 to 1.
- Answer 5 describes semantic caching as predicting frequently accessed data, which is wrong: it returns a stored answer when a new question is similar in meaning (see W5D6). A 3B model can give fluent but incorrect definitions, so its output needs checking.

## Task 2: conversation memory (5 turns)

An LLM call is stateless, so the history has to be stored and put back into each prompt. The lesson names `ConversationBufferMemory`. In the installed LangChain (1.6.4) that class is legacy and deprecated. This script uses its current equivalent: an `InMemoryChatMessageHistory` that keeps every message, a `MessagesPlaceholder("history")` in the prompt, and `RunnableWithMessageHistory` wiring them together per session id. It stores the whole conversation unchanged, which is what a buffer memory does.

| Turn | You                                                    | Result                                                                                              |
| ---- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 1    | Hi, my name is Siriyala and I'm learning LangChain.    | Greets by name                                                                                      |
| 2    | I'm running a local model through Ollama on my laptop. | Acknowledges                                                                                        |
| 3    | What is my name?                                       | "Your name is Siriyala." (PASS)                                                                     |
| 4    | What am I learning, and how am I running the model?    | "You're learning LangChain, and you're running a local model through Ollama on your laptop." (PASS) |
| 5    | Summarise our conversation so far in one sentence.     | Correct one-sentence summary                                                                        |

- History stored after 5 turns: **10 messages** (5 turns x 2).
- **Control:** the same "What is my name?" question in a new session with no history got "I don't have any information about your name", which shows the memory is what made turn 3 work.
- The legacy `ConversationBufferMemory` class was not installed (`langchain-classic`), so the script reported that and skipped it.
- Running the script prints two `LangChainDeprecationWarning` messages. Both classes still work in 1.6.4 but are deprecated in favour of LangGraph's built-in persistence.

## Task 3: agent with 2 tools

The agent uses `create_agent` (LangChain 1.x, built on LangGraph) with two tools:

- `web_search`: a stub that looks up a small built-in dictionary (no real internet)
- `calculator`: safe arithmetic with Python's `ast` module (no `eval`, with checks for division by zero and huge exponents)

The printed trace follows the ReAct pattern (Reason, Act, Observe): an Action is a tool call, an Observation is the tool result, and the last message is the Final answer. A task passes only if the answer is correct **and** the required tools were called.

Three runs were made, and all are reported:

| Run                          | Task 1: capital of France                                                                        | Task 2: 1234 x 5678 + 91 | Task 3: Everest in feet                                                                                                                                            | Score                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| llama3.2:3b, original prompt | Searched correctly, then made an unneeded `calculator('1')` call; final answer did not say Paris | Correct (7006743)        | Never called `web_search`; passed words into the calculator (syntax error); final answer printed a fake tool call as text and claimed a search result it never got | 1/3                            |
| llama3.2:3b, stricter prompt | Searched correctly, but the answer was "No calculation needed for this fact." (no Paris)         | Correct                  | Correct number (29,032.15) but `web_search` was never called, the height came from the model's memory                                                              | 1/3 (2/3 on answer text alone) |
| qwen2.5:3b, stricter prompt  | Correct                                                                                          | Correct                  | `web_search` then `calculator`: "8,849 metres tall, approximately 29,032.15 feet" (one extra harmless `calculator('3.28084')` call)                                | **3/3**                        |

Timings of the final re-run (seconds, tasks 1 to 3): llama3.2:3b 7.4, 9.7, 11.1; qwen2.5:3b 3.9, 6.6, 13.7.

What changed between the first and second run: the system prompt now tells the model to look facts up instead of guessing, to put only numbers in the calculator, to call `web_search` first on two-part questions, to skip unneeded tool calls, and to answer in a full sentence. The calculator's error message also explains how to recover. This fixed the bad calculator arguments and the fake tool call, but llama3.2:3b still skipped a tool, so the model matters more than the prompt here.

## Concepts (viva)

- **Chain vs a single LLM call:** a chain joins reusable steps (prompt template, model, parser) so each can be swapped, tested and reused, and the whole pipeline has the same `invoke`, `batch` and `stream` interface. A single call is one fixed prompt in and raw text out.
- **What memory solves:** model calls are stateless. Memory stores earlier messages and adds them to each new prompt, so the model can answer questions about the conversation. The control run shows the model cannot know the name without it.
- **ReAct:** the model alternates reasoning with actions. It decides what it needs, calls a tool, reads the observation, and repeats until it can answer.

## Limitations

- The agent results depend heavily on the model: llama3.2:3b skipped tools or ignored results, while qwen2.5:3b followed the loop. Three tasks is a small sample.
- The pass check is crude: a keyword in the answer plus the tools used. It doesn't verify the whole answer.
- `web_search` is a stub with four built-in facts, so the agent is not tested on real search results.
- Memory is in-process only (lost when the script ends), and a buffer grows with every turn until it no longer fits the context window. Window or summary memory would fix that.
- Both memory classes are deprecated in LangChain 1.6.4. A production version would use LangGraph persistence (a checkpointer).
- A 3B model can give fluent but wrong answers (see chain answers 2 and 5), and each call takes 7 to 13 seconds on CPU.

## Self-review checklist

- [x] Chain runs on 5 inputs and shows each step's output
- [x] Memory keeps 10 messages over 5 turns and the control shows the difference
- [x] Agent runs 3 tasks with both tools; results reported for all three runs
- [x] Rerun of `lc_agent.py` prints `Passed 1/3` for llama3.2:3b and `Passed 3/3` for qwen2.5:3b
- [x] `.venv/` in `.gitignore`
