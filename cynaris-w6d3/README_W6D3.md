# W6D3: LangChain Tools & Agents

A LangChain chain with conversation memory and a two-tool agent, built with LangChain, LangGraph and local Ollama (`llama3.2`).

## Contents

| File              | Task | What it does                                                                                                                            |
| ----------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `chain_memory.py` | 1, 2 | `PromptTemplate -> OllamaLLM -> StrOutputParser` chain tested on 5 inputs, then a chat chain with message history verified over 5 turns |
| `agent_tools.py`  | 3    | Tool-calling agent (`create_agent`) with a web-search stub and a safe calculator, run on 3 tasks                                        |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (use source .venv/bin/activate on macOS/Linux)
pip install langchain langchain-core langchain-ollama langgraph
ollama pull llama3.2
```

Ollama must be running locally.

## Usage

```bash
python chain_memory.py
python agent_tools.py
```

## Results

**Task 1: chain.** Five inputs (overfitting, embeddings, recursion, REST API, vector database) each returned a coherent explanation.

**Task 2: memory.** Over 5 turns the model recalled the user's name on turn 4 and summarised all stated facts on turn 5. The script asserts that 10 messages (5 human, 5 AI) are stored, and the assertion passes.

**Task 3: agent.**

| Task                        | Tools used                                                                                                                                                                         |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `23 * 47 + 130`             | `calculator` (result 1211)                                                                                                                                                         |
| Height of Mount Everest     | `web_search` (8,849 m)                                                                                                                                                             |
| Everest vs a 2-metre person | First run: `web_search` only, the model did the division itself. After adding a system prompt requiring the calculator: <CONFIRM: web_search + calculator / still web_search only> |

## Design notes

- **Chain:** built with LCEL (`prompt | llm | parser`), which packages formatting, generation and parsing into one reusable pipeline.
- **Memory:** `RunnableWithMessageHistory` with `InMemoryChatMessageHistory` keeps the full message buffer per session. It replaces the older `ConversationBufferMemory`.
- **Agent:** `langchain.agents.create_agent` runs the ReAct loop (reason, act, observe). The script prints each action and observation and a `TOOLS USED` line. I moved from `langgraph.prebuilt.create_react_agent` after it showed a deprecation warning.
- **Calculator:** parses expressions with `ast` instead of `eval()`, so model output cannot execute arbitrary code.
- **Search tool:** a stub with canned answers, so runs are repeatable offline.

## Known limitations

- **Deprecation warnings:** `RunnableWithMessageHistory` and `InMemoryChatMessageHistory` are deprecated in the installed LangChain version. They work, but the recommended replacement is LangGraph's built-in persistence (`InMemorySaver`).
- **Small-model errors:** `llama3.2` made factual mistakes. It expanded "RAG" as "Reasoning About Graphs" and "REST" as "Representational State of Resource". Small local models hallucinate, which is why retrieval-grounded answers matter.
- **Tool choice:** the model skipped the calculator on the multi-step task until the system prompt required it.
- Memory is in-process only and is lost when the script exits.
- The search tool is a stub and only has three stored answers.
