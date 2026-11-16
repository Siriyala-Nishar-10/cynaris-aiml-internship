"""Command line.

  python -m research_crew --topic "..." --label baseline                 # no web access
  python -m research_crew --topic "..." --label web --web tool           # agent calls web_search
  python -m research_crew --topic "..." --label prefetch --web prefetch  # search done in advance
  python -m research_crew compare baseline web                           # compare two outputs
"""
import argparse
import csv
import sys
import time

from research_crew.config import load_settings
from research_crew.metrics import analyse, slugify
from research_crew.search import run_search

DEFAULT_TOPIC = ("Running AI assistants locally with Ollama: "
                 "benefits and limits for small businesses")


def output_text(obj) -> str:
    """CrewAI outputs expose the final text as .raw; fall back to str() for other versions."""
    return getattr(obj, "raw", None) or str(obj)


def run_crew(topic, web, settings):
    """Build and run the crew. Returns (final_text, [per-task texts])."""
    from crewai import LLM, Crew, Process

    from research_crew.agents import build_agents
    from research_crew.tasks import build_tasks

    llm = LLM(model=f"ollama/{settings.llm_model}", base_url=settings.ollama_host)
    tools, web_results = [], ""
    if web == "tool":
        from research_crew.tools import WebSearchTool
        tools = [WebSearchTool()]
    elif web == "prefetch":
        web_results = run_search(topic, settings.max_results)
    agents = build_agents(llm, tools, max_iter=settings.max_iter)
    tasks = build_tasks(*agents, use_tool=(web == "tool"), web_results=web_results)
    crew = Crew(agents=list(agents), tasks=tasks, process=Process.sequential, verbose=True)
    result = crew.kickoff(inputs={"topic": topic})
    return output_text(result), [output_text(getattr(t, "output", "")) for t in tasks]


def cmd_run(args, settings):
    start = time.perf_counter()
    final, steps = run_crew(args.topic, args.web, settings)
    seconds = round(time.perf_counter() - start, 1)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    header = f"<!-- topic: {args.topic} | web: {args.web} | model: {settings.llm_model} -->\n\n"
    (settings.outputs_dir / f"{args.label}.md").write_text(header + final, encoding="utf-8")
    names = ["research notes", "draft briefing", "reviewed briefing"]
    steps_md = "\n\n".join(f"## {n}\n\n{t}" for n, t in zip(names, steps))
    (settings.outputs_dir / f"{args.label}_steps.md").write_text(steps_md, encoding="utf-8")
    print(f"\nDone in {seconds}s. Saved outputs/{args.label}.md")
    print(analyse(final))


def cmd_compare(args, settings):
    rows = []
    for label in args.labels:
        text = (settings.outputs_dir / f"{label}.md").read_text(encoding="utf-8")
        rows.append({"label": label, **analyse(text)})
    csv_path = settings.outputs_dir / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'label':<12}{'words':>7}{'urls':>6}{'domains':>9}{'figures':>9}")
    for r in rows:
        print(f"{r['label']:<12}{r['words']:>7}{r['urls']:>6}{r['domains']:>9}{r['figures']:>9}")
    print(f"Saved {csv_path}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    settings = load_settings()
    if argv and argv[0] == "compare":
        parser = argparse.ArgumentParser(prog="research_crew compare")
        parser.add_argument("labels", nargs="+", help="output labels, e.g. baseline web")
        cmd_compare(parser.parse_args(argv[1:]), settings)
        return
    parser = argparse.ArgumentParser(prog="research_crew", description="Multi-agent research crew")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--label", default="baseline", help="output file name (no extension)")
    parser.add_argument("--web", choices=["off", "tool", "prefetch"], default="off",
                        help="off: no web; tool: agent calls web_search; prefetch: search first")
    parser.add_argument("--slug", action="store_true", help="use the topic as the label")
    args = parser.parse_args(argv)
    if args.slug:
        args.label = slugify(args.topic)
    cmd_run(args, settings)
