"""LangGraph orchestration and MLflow tracking for the research crew."""
from pathlib import Path
from typing import TypedDict
import mlflow
from langgraph.graph import StateGraph, END
from src.crew import generate_report


class ResearchState(TypedDict, total=False):
    topic: str
    report: str
    report_path: str
    run_id: str


def research_node(state: ResearchState) -> ResearchState:
    """Generate the report through the CrewAI crew."""
    topic = state["topic"]
    report = generate_report(topic)
    return {"report": report}


def save_report_node(state: ResearchState) -> ResearchState:
    """Persist the generated Markdown report locally."""
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_topic = "".join(c.lower() if c.isalnum() else "_" for c in state["topic"])
    safe_topic = "_".join(part for part in safe_topic.split("_") if part)[:60] or "research"
    report_path = output_dir / f"{safe_topic}_report.md"
    report_path.write_text(state["report"], encoding="utf-8")
    return {"report_path": str(report_path)}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("research", research_node)
    graph.add_node("save_report", save_report_node)
    graph.set_entry_point("research")
    graph.add_edge("research", "save_report")
    graph.add_edge("save_report", END)
    return graph.compile()


def run_research_workflow(topic: str) -> dict:
    """Run the graph and record basic run metadata in MLflow."""
    mlflow.set_experiment("W9D5-Automated-Research-Report-Agent")
    with mlflow.start_run() as run:
        mlflow.log_param("topic", topic)
        app = build_graph()
        result = app.invoke({"topic": topic})
        mlflow.log_param("workflow", "CrewAI + LangGraph")
        mlflow.log_metric("report_characters", len(result["report"]))
        mlflow.log_artifact(result["report_path"])
        result["run_id"] = run.info.run_id
        return result
