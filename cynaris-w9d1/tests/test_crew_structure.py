"""Checks that the agents and tasks are wired correctly. Skipped if CrewAI is not installed."""
import pytest

pytest.importorskip("crewai")

from crewai import LLM  # noqa: E402

from research_crew.agents import build_agents  # noqa: E402
from research_crew.tasks import build_tasks  # noqa: E402


@pytest.fixture(scope="module")
def agents():
    llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")
    return build_agents(llm)


def test_three_distinct_agents(agents):
    roles = [a.role for a in agents]
    assert len(set(roles)) == 3
    assert roles == ["Senior Research Analyst", "Technical Writer", "Fact-Checking Editor"]


def test_each_task_belongs_to_the_right_agent_and_passes_context_on(agents):
    researcher, writer, reviewer = agents
    research, writing, review = build_tasks(researcher, writer, reviewer)
    assert (research.agent, writing.agent, review.agent) == (researcher, writer, reviewer)
    assert writing.context == [research]
    assert review.context == [research, writing]


def test_prefetched_web_results_reach_the_research_task(agents):
    research, _, _ = build_tasks(*agents, web_results="[1] Example {page}\nURL: https://x.test")
    assert "https://x.test" in research.description
    assert "{page}" not in research.description         # braces removed so templating is safe


def test_tool_mode_tells_the_researcher_to_use_the_tool(agents):
    research, _, _ = build_tasks(*agents, use_tool=True)
    assert "web_search" in research.description
