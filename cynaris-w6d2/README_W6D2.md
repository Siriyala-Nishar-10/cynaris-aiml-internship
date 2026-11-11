# W6D2: LangChain Memory & Conversation History

The three LangChain programs from W6D1 (chain, memory, agent), re-run on a local Ollama model, plus a new script for this day's topic: three ways of keeping conversation history (full buffer, sliding window, running summary), built on LangGraph persistence with a SQLite checkpointer.

## What's in this folder

| File                                                                                             | Purpose                                                                     |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `lc_chain.py`                                                                                    | Task 1: `PromptTemplate -> OllamaLLM -> OutputParser` chain, 5 inputs       |
| `lc_memory.py`                                                                                   | Task 2: conversation memory over 5 turns, with a no-memory control          |
| `lc_agent.py`                                                                                    | Task 3: agent with 2 tools (web search stub + calculator), 3 tasks          |
| `lc_memory_types.py`                                                                             | W6D2 extension: buffer vs window vs summary memory, and SQLite persistence  |
| `requirements.txt`                                                                               | `langchain`, `langchain-ollama`, `langgraph`, `langgraph-checkpoint-sqlite` |
| `chain_results.json`, `memory_results.json`, `agent_results_*.json`, `memory_types_results.json` | Saved outputs                                                               |
| `evidence/`                                                                                      | Screenshots of outputs                                                      |

`chat_memory.sqlite` (created by the persistence demo) holds conversation text, so it is listed in `.gitignore` and not committed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull qwen2.5:3b           # only for the agent comparison
```

## Usage

```bash
python lc_chain.py
python lc_memory.py
python lc_agent.py                       # llama3.2:3b
python lc_agent.py --model qwen2.5:3b
python lc_memory_types.py                # several minutes on CPU
```

## Tasks 1 to 3 (re-run from W6D1)

**Chain.** `prompt | OllamaLLM | StrOutputParser()`, shown step by step for the first input and then run on 5 inputs (6.76 to 8.71 seconds each). A second chain with `CommaSeparatedListOutputParser` returned a real Python list of 4 items. Two limits of the 3B model showed up: answer 2 added a preamble although the prompt said "exactly two sentences", and answer 5 described semantic caching as predicting frequently used data, which is wrong (it returns a stored answer when a new question means the same as an old one).

**Memory.** A 5-turn conversation with a per-session history (`RunnableWithMessageHistory` with `InMemoryChatMessageHistory`, the current equivalent of `ConversationBufferMemory`). Turn 3 answered "Your name is Siriyala.", the history held 10 messages, and a new session without history said "I don't have any information about your name." Both classes print deprecation warnings in LangChain 1.6.4, which is why `lc_memory_types.py` uses LangGraph persistence instead.

**Agent.** `create_agent` with `web_search` (stub) and `calculator` (safe `ast` evaluation). A task passes only if the answer is right and the required tools were called.
| Model | Task 1: capital of France | Task 2: 1234 x 5678 + 91 | Task 3: Everest in feet | Score |
| ----------- | ------------------------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------------------------------ | ------------------------------ |
| llama3.2:3b | Searched, but the answer was "No calculation needed for this fact." (no Paris) | Correct | Right number (29,032.15) but `web_search` was never called | 1/3 |
| qwen2.5:3b | Searched, answered "The capital of France is Paris." | Correct (7006743) | `web_search` then `calculator`: 8,849 m is about 29,032.15 ft (one extra harmless `calculator('3.28084')` call) | 3/3 (result from the W6D1 run) |

## W6D2 extension: memory strategies and persistence

A model call is stateless, so the history must be stored and sent again. The problem is size: every turn adds to the history. `lc_memory_types.py` runs the same 10-turn conversation through three strategies, using a LangGraph `StateGraph` where the graph state holds the messages and an `InMemorySaver` checkpointer saves it per `thread_id`:

- **buffer:** send the whole history every turn
- **window:** send only the last 4 messages
- **summary:** send a running summary of older messages (written by the model) plus the last 4 messages

Turns 1 to 4 state facts (name, project, favourite language, setup), turns 5 to 8 are filler questions, and turn 9 ("What is my name?") and turn 10 ("What am I building, and which language do I like?") test recall.

| Strategy        | Context sent at turn 10 | Turn 9 (name) | Turn 10 (project, language) | Total seconds |
| --------------- | ----------------------- | ------------- | --------------------------- | ------------- |
| buffer          | 2,273 chars             | PASS          | PASS                        | 40.0          |
| window (last 4) | 352 chars               | FORGOT        | FORGOT                      | 55.7          |
| summary         | 1,115 chars             | PASS          | PASS                        | 117.2         |

How the context size changed per turn: the buffer grew from 86 to 2,273 characters (about 250 more per turn), the window levelled off at about 600 to 700 characters and then fell to 352, and the summary stayed between about 700 and 1,100.

Findings:

- **Buffer** gave full recall and was the fastest here, but its size grows every turn and would eventually overflow the model's context window.
- **Window** kept the prompt small and constant, but forgot every fact older than two turns ("I don't have any information about your name.").
- **Summary** recalled the facts with about half the buffer's context at turn 10, but it took about three times as long as the buffer (117.2 s vs 40.0 s), because once the window fills each turn needs an extra model call to update the summary. In a conversation this short the buffer is simply better. The summary would only pay off in a conversation long enough that a buffer no longer fits.
- **The summary absorbed mistakes.** It correctly kept the user's facts (name, bot project, Python, Ollama), but it also recorded wrong claims from the assistant's own earlier answers, for example that the user uses Docker and that RAG is a "Rapid Application Generator". The turn 9 answer was also hedged ("I don't have that information. You are referred to as 'Siriyala'..."), though it did contain the name. A summary is only as accurate as the model that writes it.
- **Timings are noisy.** The window run was slower than the buffer run even though it sent less text, so the character counts are the reliable measure and the seconds are only indicative (CPU load and model loading vary between runs).
- **Persistence worked.** A conversation was saved to `chat_memory.sqlite` (2 messages), a brand-new SQLite connection loaded the same thread (2 messages), and the model answered "Your name is Siriyala, and you like Python." The history survived the simulated restart.

## Concepts (viva)

- **Chain vs a single LLM call:** a chain joins reusable steps (prompt, model, parser) that can be swapped, tested and reused behind the same `invoke`, `batch` and `stream` interface. A single call is one fixed prompt in and raw text out.
- **What memory solves:** model calls are stateless. Memory stores earlier messages and sends them with each new prompt. Without it the model cannot know the user's name (the control run). Unbounded memory has its own problem, which window and summary memory address.
- **Why LangGraph persistence replaced `ConversationBufferMemory`:** the conversation is part of the graph state and a checkpointer saves it per thread id, so it survives restarts, can be swapped for SQLite or Postgres, and can be trimmed or summarised inside a node.
- **ReAct:** the model alternates reasoning with actions. It decides what it needs, calls a tool, reads the observation and repeats until it can answer.

## Limitations

- The strategy comparison is one conversation, one run, with keyword checks for recall. It shows the trade-off but is not a benchmark.
- Context size is counted in characters (about 4 per token), not model tokens.
- The 3B model gave several wrong definitions (RAG was expanded differently in every strategy), so its answers and summaries need checking.
- The agent result depends heavily on the model, and three tasks is a small sample.
- `lc_memory.py` still uses the deprecated memory classes, kept to match the lesson wording. A production version would use only the LangGraph approach.
- The checkpointer stores whole conversations in a plain SQLite file, which is fine for development but would need access control and a server database (Postgres) in production.

## Self-review checklist

- [x] Chain runs on 5 inputs and shows each step's output
- [x] Memory keeps 10 messages over 5 turns and the control shows the difference
- [x] Agent runs 3 tasks with both tools; results reported for both models
- [x] Memory strategies compared on the same conversation; persistence demo passes
- [x] `.venv/`, `chat_memory.sqlite` in `.gitignore`
