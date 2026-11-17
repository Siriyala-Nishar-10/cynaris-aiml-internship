"""Two-tool ReAct agent using LangChain and Ollama."""
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools import web_search_stub, calculator

MODEL_NAME = "llama3.2"

REACT_TEMPLATE = """Answer the user as helpfully as possible. You can use these tools:

{tools}

Use this exact format:
Question: the input question
Thought: think about what information or calculation is needed
Action: one of [{tool_names}]
Action Input: the input for the action
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Question: {input}
Thought:{agent_scratchpad}"""


def build_agent():
    tools = [web_search_stub, calculator]
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )


def run_agent_tasks():
    executor = build_agent()
    tasks = [
        "Use the web search tool to explain what LangChain is.",
        "Use the calculator tool to calculate (125 * 4) + 75.",
        "Use the web search tool to explain RAG, then calculate 3 * 12.",
    ]
    outputs = []
    for index, task in enumerate(tasks, start=1):
        result = executor.invoke({"input": task})
        answer = result["output"]
        print(f"\nTask {index}: {task}\nAgent: {answer}")
        outputs.append(f"Task {index}: {task}\nAgent: {answer}\n")
    with open("results/agent_outputs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(outputs))
