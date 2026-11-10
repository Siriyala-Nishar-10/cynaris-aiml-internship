"""
These tests use a STUB `crewai` module that records how it is called. They check
OUR logic: three agents with roles/goals/backstories, tasks wired in order with the
right context, a sequential process, and the search tool given to the Researcher
only when requested. They do NOT prove the real CrewAI accepts these arguments;
that is confirmed only by running `python run_crew.py` with CrewAI installed.
"""

import sys
import types
import unittest
from unittest import mock

from research_crew.config import CrewConfig
from research_crew.crew import build_crew, run_crew


def make_stub():
    crewai = types.ModuleType("crewai")
    crewai_tools = types.ModuleType("crewai.tools")

    class Recorder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    class LLM(Recorder):
        pass

    class Agent(Recorder):
        pass

    class Task(Recorder):
        pass

    class Process:
        sequential = "sequential"
        hierarchical = "hierarchical"

    class Crew(Recorder):
        result_text = "FINAL TEXT"

        def kickoff(self):
            return types.SimpleNamespace(raw=Crew.result_text)

    def tool(name):
        def decorate(fn):
            fn.tool_name = name
            return fn
        return decorate

    crewai.LLM, crewai.Agent, crewai.Task, crewai.Crew, crewai.Process = LLM, Agent, Task, Crew, Process
    crewai_tools.tool = tool
    return {"crewai": crewai, "crewai.tools": crewai_tools}


class TestWiring(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, make_stub())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = CrewConfig(verbose=False)

    def test_three_agents_with_role_goal_and_backstory(self):
        crew = build_crew("RAG", with_search=False, config=self.config)
        self.assertEqual(len(crew.agents), 3)
        for agent in crew.agents:
            for field in ("role", "goal", "backstory"):
                self.assertTrue(getattr(agent, field).strip(), f"{agent.role} has empty {field}")
        self.assertEqual([a.role for a in crew.agents],
                         ["Research Analyst", "Technical Writer", "Fact-Checking Editor"])

    def test_process_is_sequential_and_tasks_are_ordered(self):
        crew = build_crew("RAG", with_search=True, config=self.config)
        self.assertEqual(crew.process, "sequential")
        self.assertEqual([t.agent.role for t in crew.tasks],
                         ["Research Analyst", "Technical Writer", "Fact-Checking Editor"])

    def test_each_task_has_description_and_expected_output(self):
        crew = build_crew("RAG", with_search=True, config=self.config)
        for task in crew.tasks:
            self.assertTrue(task.description.strip())
            self.assertTrue(task.expected_output.strip())

    def test_context_passes_outputs_down_the_chain(self):
        research, write, review = build_crew("RAG", True, self.config).tasks
        self.assertFalse(hasattr(research, "context"))
        self.assertEqual(write.context, [research])
        self.assertEqual(review.context, [research, write])

    def test_search_tool_only_on_researcher_and_only_when_requested(self):
        with_search = build_crew("RAG", True, self.config)
        researcher, writer, reviewer = with_search.agents
        self.assertEqual(len(researcher.tools), 1)
        self.assertEqual(researcher.tools[0].tool_name, "Web Search")
        self.assertEqual(writer.tools, [])
        self.assertEqual(reviewer.tools, [])

        without = build_crew("RAG", False, self.config)
        self.assertEqual(without.agents[0].tools, [])

    def test_no_search_run_forbids_invented_citations(self):
        research = build_crew("RAG", False, self.config).tasks[0]
        self.assertIn("Do NOT invent URLs", research.description)
        self.assertIn("never invent URLs", build_crew("RAG", False, self.config).agents[0].backstory)

    def test_search_run_requires_sources_from_results(self):
        crew = build_crew("RAG", True, self.config)
        self.assertIn("source URL", crew.tasks[0].description)
        self.assertIn("only URLs that appeared", crew.agents[0].backstory)

    def test_topic_appears_in_goals_and_tasks(self):
        crew = build_crew("quantum widgets", True, self.config)
        self.assertIn("quantum widgets", crew.agents[0].goal)
        self.assertIn("quantum widgets", crew.tasks[0].description)

    def test_delegation_off_and_iteration_capped(self):
        for agent in build_crew("RAG", True, self.config).agents:
            self.assertFalse(agent.allow_delegation)
            self.assertEqual(agent.max_iter, self.config.max_iter)

    def test_llm_receives_model_and_base_url(self):
        agent = build_crew("RAG", False, self.config).agents[0]
        self.assertEqual(agent.llm.model, "ollama/llama3.2:3b")
        self.assertEqual(agent.llm.base_url, "http://localhost:11434")

    def test_run_crew_returns_raw_text(self):
        self.assertEqual(run_crew("RAG", False, self.config), "FINAL TEXT")

    def test_run_crew_falls_back_to_str_when_no_raw_attribute(self):
        stub = make_stub()
        class Plain:
            def __str__(self):
                return "plain string result"
        stub["crewai"].Crew.kickoff = lambda self: Plain()
        with mock.patch.dict(sys.modules, stub):
            self.assertEqual(run_crew("RAG", False, self.config), "plain string result")


class TestRunCrewWithoutCrewAI(unittest.TestCase):
    def test_missing_crewai_raises_import_error(self):
        with mock.patch.dict(sys.modules, {"crewai": None, "crewai.tools": None}):
            with self.assertRaises(ImportError):
                run_crew("RAG", False, CrewConfig(verbose=False))


if __name__ == "__main__":
    unittest.main()
