"""CrewAI research crew: researcher gathers notes; writer creates a report."""
import os
from crewai import Agent, Task, Crew, Process, LLM


def build_research_crew():
    """Build a two-agent sequential crew using the configured LLM provider."""
    model = os.getenv("CREWAI_MODEL", "ollama/llama3.2")
    llm = LLM(model=model)

    researcher = Agent(
        role="Research Analyst",
        goal="Collect accurate, relevant research notes about the requested topic.",
        backstory=(
            "You are a careful analyst. Separate established facts from uncertainty "
            "and never invent citations or claim to have browsed when you have not."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
    writer = Agent(
        role="Technical Report Writer",
        goal="Turn research notes into a clear, structured report.",
        backstory=(
            "You write concise reports with an executive summary, key findings, "
            "limitations, and follow-up questions."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Prepare research notes on: {topic}. Provide key concepts, potential "
            "applications, benefits, risks, and open questions. Clearly label any "
            "claims that need independent verification. Do not fabricate sources."
        ),
        expected_output="Structured research notes with claims and uncertainties.",
        agent=researcher,
    )
    report_task = Task(
        description=(
            "Using the research notes, write a report about {topic}. Include: title, "
            "executive summary, background, key findings, applications, limitations, "
            "conclusion, and suggested sources to verify. Do not invent URLs or citations."
        ),
        expected_output="A readable research report in Markdown.",
        agent=writer,
        context=[research_task],
    )
    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, report_task],
        process=Process.sequential,
        verbose=True,
    )


def generate_report(topic: str) -> str:
    crew = build_research_crew()
    result = crew.kickoff(inputs={"topic": topic})
    return str(result)
