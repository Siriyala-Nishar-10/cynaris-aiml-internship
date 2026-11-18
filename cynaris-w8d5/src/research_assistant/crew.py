"""CrewAI layer: an analyst extracts evidence, a writer drafts the cited answer."""
from __future__ import annotations

from .config import Settings


def run_crew(question: str, contexts: list[str], cfg: Settings) -> str:
    # Imported lazily so the rest of the package is testable without CrewAI.
    from crewai import Agent, Crew, LLM, Process, Task

    llm = LLM(model=f"ollama/{cfg.chat_model}", base_url=cfg.ollama_url, temperature=0.2)
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))

    analyst = Agent(
        role="Research Analyst",
        goal="Pull out only the facts in the provided sources that bear on the question.",
        backstory="A meticulous analyst who never uses knowledge outside the sources.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write a concise answer that cites sources as [n].",
        backstory="A writer who admits when the sources are insufficient.",
        llm=llm, allow_delegation=False, verbose=False,
    )
    extract = Task(
        description=f"Question: {question}\n\nSources:\n{numbered}\n\n"
                    "List the relevant facts, each tagged with its source number.",
        expected_output="A bulleted list of facts with [n] source tags.",
        agent=analyst,
    )
    draft = Task(
        description="Using only the extracted facts, answer the question with [n] citations. "
                    "If the facts are insufficient, say so.",
        expected_output="A short, cited answer.",
        agent=writer, context=[extract],
    )
    crew = Crew(agents=[analyst, writer], tasks=[extract, draft],
                process=Process.sequential, verbose=False)
    return str(crew.kickoff())
