"""Run the W9D4 chain, memory, and two-tool agent demonstrations."""
from chain import run_chain_tests
from memory import run_memory_demo
from agent import run_agent_tasks


def main():
    print("\n=== 1. LANGCHAIN CHAIN: 5 INPUTS ===")
    run_chain_tests()

    print("\n=== 2. CONVERSATION MEMORY: 5 TURNS ===")
    run_memory_demo()

    print("\n=== 3. TWO-TOOL REACT AGENT: 3 TASKS ===")
    run_agent_tasks()


if __name__ == "__main__":
    main()
