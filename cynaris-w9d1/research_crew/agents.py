"""The three agents. Each has a role, a goal and a backstory that shape how it behaves.

{topic} is filled in from crew.kickoff(inputs={"topic": ...}).
"""
from crewai import Agent


def build_agents(llm, researcher_tools=None, max_iter=6):
    """Return (researcher, writer, reviewer). Only the researcher gets tools."""
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Find accurate, current facts about {topic} and say where each fact came from",
        backstory=("You are a careful analyst who only reports what the evidence supports. "
                   "You always name the source of a fact and say so when you cannot verify one."),
        llm=llm,
        tools=researcher_tools or [],
        allow_delegation=False,
        max_iter=max_iter,
        verbose=True,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Turn research notes about {topic} into a clear briefing for a non-technical manager",
        backstory=("You explain technical subjects in plain language. You never add facts that "
                   "are not in the research notes."),
        llm=llm,
        allow_delegation=False,
        max_iter=max_iter,
        verbose=True,
    )
    reviewer = Agent(
        role="Fact-Checking Editor",
        goal="Check the briefing on {topic} against the research notes and fix or flag problems",
        backstory=("You are a strict editor. You remove claims the notes do not support, fix "
                   "unclear wording, and keep the sources."),
        llm=llm,
        allow_delegation=False,
        max_iter=max_iter,
        verbose=True,
    )
    return researcher, writer, reviewer
