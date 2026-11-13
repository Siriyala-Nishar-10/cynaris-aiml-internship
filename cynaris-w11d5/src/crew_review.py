
from crewai import Agent, Crew, LLM, Process, Task

from .config import OLLAMA_BASE_URL, CHAT_MODEL


def review_answer(question: str, answer: str, contexts: list[str]) -> str:
    local_llm = LLM(
        model=f"ollama/{CHAT_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        context_window_size=8192,
    )

    reviewer = Agent(
        role="RAG Quality Reviewer",
        goal=(
            "Evaluate whether the answer correctly and sufficiently "
            "addresses the user's question using retrieved context."
        ),
        backstory=(
            "You are a careful RAG evaluator. You distinguish relevant "
            "evidence from unrelated supplementary context."
        ),
        llm=local_llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Evaluate the following RAG response.

QUESTION:
{question}

ANSWER:
{answer}

RETRIEVED CONTEXTS:
{chr(10).join(contexts)}

REVIEW RULES:
1. Return PASS if the answer addresses the question.
2. Check factual claims against the retrieved contexts.
3. Do not require unrelated supplementary context.
4. Do not flag an answer merely because it omits
   information unrelated to the user's question.
5. Flag unsupported or contradictory factual claims.
6. Flag missing information only when it is essential
   to answering the question.

Return exactly:
PASS - followed by a short reason
or
FLAG - followed by a short reason.
""",
        expected_output="PASS or FLAG followed by one concise reason.",
        agent=reviewer,
    )

    crew = Crew(
        agents=[reviewer],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    return str(crew.kickoff()).strip()
