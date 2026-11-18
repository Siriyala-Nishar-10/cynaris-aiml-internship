"""ConversationBufferMemory demonstration over five turns.

This uses the classic LangChain LLMChain + ConversationBufferMemory API
requested by the W9D4 practical. These classic APIs are deprecated in newer
LangChain releases, so requirements.txt pins a compatible 0.3.x stack.
"""
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.2"


def build_memory_chain():
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=(
            "You are a helpful assistant. Use the conversation history when "
            "answering follow-up questions.\n"
            "Conversation history:\n{history}\n"
            "User: {input}\nAssistant:"
        ),
    )
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="input",
        return_messages=False,
    )
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    return LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=False)


def run_memory_demo():
    chain = build_memory_chain()
    turns = [
        "My project is called ShroomTalk.",
        "It helps with financial advisory for rural mushroom farming.",
        "What is the name of my project?",
        "What farming area does it focus on?",
        "Summarize my project in one sentence.",
    ]
    log = []
    for index, message in enumerate(turns, start=1):
        response = chain.invoke({"input": message})
        answer = response["text"]
        print(f"\nTurn {index}\nUser: {message}\nAssistant: {answer}")
        log.append(f"Turn {index}\nUser: {message}\nAssistant: {answer}\n")
    print("\nStored conversation history:")
    print(chain.memory.buffer)
    with open("results/memory_outputs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
        f.write("\n\nStored conversation history:\n")
        f.write(chain.memory.buffer)
