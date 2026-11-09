"""PromptTemplate -> Ollama chat model -> StrOutputParser."""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

MODEL_NAME = "llama3.2"


def build_chain():
    prompt = PromptTemplate.from_template(
        "You are a concise learning assistant. Answer in 2-3 sentences.\n"
        "Question: {question}\n"
        "Answer:"
    )
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    return prompt | llm | StrOutputParser()


def run_chain_tests():
    chain = build_chain()
    questions = [
        "What is an API?",
        "Explain Python lists.",
        "What is retrieval augmented generation?",
        "Why do developers use Git branches?",
        "What is an AI agent?",
    ]
    outputs = []
    for index, question in enumerate(questions, start=1):
        answer = chain.invoke({"question": question})
        print(f"\nInput {index}: {question}\nOutput: {answer}")
        outputs.append(f"Input {index}: {question}\nOutput: {answer}\n")
    with open("results/chain_outputs.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(outputs))
