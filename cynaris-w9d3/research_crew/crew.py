"""
Three-agent research crew: Researcher -> Writer -> Reviewer.

Process: sequential. Each task's output is passed to later tasks through
`context=[...]`, which is CrewAI's mechanism for agent-to-agent communication in
a sequential crew. No agent delegates (allow_delegation=False) so the flow stays
predictable on a small local model.

`crewai` is imported inside the functions that need it so the rest of the package
(and its tests) work without it installed.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from .config import CrewConfig
from .tools import SearchLog, build_search_tool

# The three stages, in task order. Used to label each task's output.
STAGE_NAMES = ("research", "draft", "final")


@dataclass
class CrewRun:
    """Everything one crew run produced."""
    final_text: str
    stages: dict = field(default_factory=dict)   # stage name -> text
    duration_s: float = 0.0


def build_llm(config: CrewConfig):
    from crewai import LLM
    return LLM(model=config.model, base_url=config.base_url, temperature=config.temperature)


def build_agents(llm, topic: str, tools: list, config: CrewConfig) -> dict:
    from crewai import Agent

    has_search = bool(tools)
    sourcing_rule = (
        "You cite only URLs that appeared in your web search results."
        if has_search else
        "You have no internet access. You never invent URLs, quotes, statistics or "
        "citations, and you say plainly when something comes from general knowledge."
    )
    common = dict(llm=llm, verbose=config.verbose, allow_delegation=False, max_iter=config.max_iter)

    researcher = Agent(
        role="Research Analyst",
        goal=f"Collect accurate, specific findings about: {topic}",
        backstory=("You are a careful analyst who values verifiable facts over fluent "
                   f"guesses. {sourcing_rule}"),
        tools=tools, **common)
    writer = Agent(
        role="Technical Writer",
        goal=f"Turn the research into a clear, well-structured briefing on: {topic}",
        backstory=("You write for engineers who are new to the topic. You use only the "
                   "facts you were given and never add claims the research does not support."),
        tools=[], **common)
    reviewer = Agent(
        role="Fact-Checking Editor",
        goal="Find unsupported claims and weak spots in the draft, then fix them",
        backstory=("You are a sceptical editor. You check every claim in the draft against "
                   "the research notes, remove or flag anything unsupported, and tighten the writing."),
        tools=[], **common)
    return {"researcher": researcher, "writer": writer, "reviewer": reviewer}


def build_tasks(agents: dict, topic: str, with_search: bool) -> list:
    from crewai import Task

    if with_search:
        research_how = ("Use the Web Search tool several times with different queries. "
                        "Each finding must end with the source URL it came from.")
    else:
        research_how = ("Use only what you already know. Do NOT invent URLs or citations. "
                        "End your notes with a line stating that no sources were consulted.")

    research = Task(
        description=f"Research the topic: {topic}\n{research_how}\n"
                    "Produce 6 to 8 specific findings as short bullet points.",
        expected_output="A bulleted list of 6-8 specific findings" +
                        (", each followed by its source URL." if with_search else
                         " and a closing line saying no sources were consulted."),
        agent=agents["researcher"])

    write = Task(
        description=f"Using only the research notes, write a briefing of about 300 words on: {topic}\n"
                    "Structure: a one-sentence summary, three short sections with headings, "
                    "and a 'Sources' list" + (" containing the URLs from the research." if with_search
                                              else " that says no sources were consulted."),
        expected_output="A markdown briefing of roughly 300 words with headings and a Sources section.",
        agent=agents["writer"],
        context=[research])

    review = Task(
        description="Review the draft against the research notes. List any claims the research does not "
                    "support, then output the corrected final briefing. Keep the markdown structure and "
                    "the Sources section; never add a URL that is not in the research notes.",
        expected_output="First a short 'Issues found' list, then the full corrected final briefing in markdown.",
        agent=agents["reviewer"],
        context=[research, write])
    return [research, write, review]


def build_crew(topic: str, with_search: bool, config: CrewConfig, search_log: Optional[SearchLog] = None):
    from crewai import Crew, Process

    llm = build_llm(config)
    tools = [build_search_tool(config.max_search_results, search_log)] if with_search else []
    agents = build_agents(llm, topic, tools, config)
    tasks = build_tasks(agents, topic, with_search)
    return Crew(agents=list(agents.values()), tasks=tasks, process=Process.sequential,
                verbose=config.verbose)


def run_crew(topic: str, with_search: bool, config: CrewConfig = CrewConfig(),
             search_log: Optional[SearchLog] = None) -> CrewRun:
    """Run the crew. Returns the final text, each stage's output, and the wall time.

    Newer CrewAI versions return a CrewOutput with `.raw` and `.tasks_output`; older
    ones return something that str() turns into the final text. If the per-task
    outputs are not available in the expected shape, only the final text is kept."""
    crew = build_crew(topic, with_search, config, search_log)
    started = time.perf_counter()
    result = crew.kickoff()
    duration = time.perf_counter() - started

    final_text = getattr(result, "raw", None) or str(result)
    raws = [getattr(o, "raw", None) or str(o) for o in (getattr(result, "tasks_output", None) or [])]
    if len(raws) == len(STAGE_NAMES):
        stages = dict(zip(STAGE_NAMES, raws))
        final_text = raws[-1] or final_text
    else:
        stages = {"final": final_text}
    return CrewRun(final_text=final_text, stages=stages, duration_s=duration)
