"""
Run the crew in a given mode, save each stage's output, analyse it, and build the
report that documents the difference between the no-search and with-search runs.

`run_mode` takes the crew runner as an argument (`run_fn`) so it can be tested with
a fake runner, no CrewAI or model needed.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from .analysis import extract_urls, output_metrics, provenance, quality_checks
from .config import CrewConfig
from .crew import run_crew
from .tools import SearchLog


def run_mode(topic: str, with_search: bool, config: CrewConfig, runs_dir,
             run_fn: Callable = run_crew) -> dict:
    """Run one mode. Returns a summary dict (and writes files under runs_dir/<mode>/).
    A model/tool/network failure is recorded in the summary instead of raised, so one
    flaky run does not lose the other. A missing CrewAI install still raises ImportError."""
    mode = "with_search" if with_search else "no_search"
    out_dir = Path(runs_dir) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    log = SearchLog() if with_search else None
    summary = {"mode": mode, "topic": topic, "model": config.model,
               "run_at": datetime.now().isoformat(timespec="seconds")}

    try:
        run = run_fn(topic, with_search, config, search_log=log)
    except ImportError:
        raise
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _write_summary(out_dir, summary)
        return summary

    files = []
    for name, text in run.stages.items():
        path = out_dir / f"{name}.md"
        path.write_text(text, encoding="utf-8")
        files.append(path.name)

    research = run.stages.get("research")
    summary.update({
        "duration_s": round(run.duration_s, 1),
        "metrics": output_metrics(run.final_text),
        "quality": quality_checks(run.final_text, research),
        "files": files,
        "final_text": run.final_text,
    })
    if with_search:
        summary["search"] = log.to_dict()
        summary["provenance"] = provenance(research if research is not None else run.final_text, log.urls)
        (out_dir / "search_log.json").write_text(json.dumps(log.to_dict(), indent=2), encoding="utf-8")
        files.append("search_log.json")
    else:
        summary["urls_in_no_search_output"] = extract_urls(run.final_text)
    _write_summary(out_dir, summary)
    return summary


def _write_summary(out_dir: Path, summary: dict):
    on_disk = {k: v for k, v in summary.items() if k != "final_text"}
    (out_dir / "summary.json").write_text(json.dumps(on_disk, indent=2), encoding="utf-8")


def build_pipeline_report(topic: str, model: str, no_search: dict, with_search: dict) -> str:
    def cell(d, getter, fmt=str):
        if d.get("error"):
            return "n/a (run failed)"
        try:
            value = getter(d)
        except (KeyError, TypeError):
            return "n/a"
        return "n/a" if value is None else fmt(value)

    yes_no = lambda b: "yes" if b else "NO"
    rows = [
        ("Run time (s)", lambda d: d["duration_s"], str),
        ("Final word count", lambda d: d["metrics"]["word_count"], str),
        ("Headings", lambda d: d["quality"]["heading_count"], str),
        ("Has Sources section", lambda d: d["quality"]["has_sources_section"], yes_no),
        ("URLs in final output", lambda d: d["metrics"]["url_count"], str),
        ("Final URLs the research never mentioned", lambda d: None if d["quality"]["unsupported_urls"] is None
         else len(d["quality"]["unsupported_urls"]), str),
        ("Quality rubric passed", lambda d: d["quality"]["passed"], yes_no),
    ]
    prov_rows = [
        ("URLs returned by the search tool", lambda d: d["provenance"]["urls_returned_by_search"], str),
        ("URLs cited by the Researcher", lambda d: d["provenance"]["urls_cited"], str),
        ("Cited but never returned by search", lambda d: len(d["provenance"]["cited_but_never_returned"]), str),
        ("All citations traceable to a search result", lambda d: d["provenance"]["all_citations_traceable"], yes_no),
    ]

    lines = [
        "# W9D3: Research Crew Pipeline - Without vs With Web Search", "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Topic: {topic}  ", f"Model: {model}", "",
        "> Automatic checks are crude. They cannot tell whether a claim is true or whether a real",
        "> URL supports the sentence beside it. Complete the manual checks at the end.", "",
        "## Output quality", "", "| Check | Without search | With search |", "|---|---|---|",
        *[f"| {label} | {cell(no_search, g, f)} | {cell(with_search, g, f)} |" for label, g, f in rows], "",
        "## Citation provenance (with-search run)", "",
        "Every URL the Researcher cites should have been returned by the search tool. A cited URL that was",
        "never returned is a likely fabrication, even though search was available.", "",
        "| Check | With search |", "|---|---|",
        *[f"| {label} | {cell(with_search, g, f)} |" for label, g, f in prov_rows], "",
    ]
    cited_bad = (with_search.get("provenance") or {}).get("cited_but_never_returned") or []
    if cited_bad:
        lines += ["**Cited but never returned by search:**", *[f"- {u}" for u in cited_bad], ""]
    invented = no_search.get("urls_in_no_search_output") or []
    if invented:
        lines += ["**URLs in the no-search output (the agents had no way to look these up):**",
                  *[f"- {u}" for u in invented], ""]

    for title, d in (("Without web search", no_search), ("With web search", with_search)):
        lines += [f"## Final output: {title}", ""]
        lines += [f"**Run failed:** {d['error']}", ""] if d.get("error") else [d["final_text"], ""]

    lines += [
        "## Manual checks (fill in after reading both outputs and the stage files in runs/)", "",
        "- [ ] Opened at least 3 URLs from the with-search output; they exist and support the nearby claims:",
        "- [ ] Compared runs/*/research.md -> draft.md -> final.md: what did the Writer add or drop, and what did the Reviewer change?",
        "- [ ] Specific facts (numbers, names, dates) present with search but missing without:",
        "- [ ] Anything in the final briefing that is wrong or unsupported despite passing the rubric:",
        "- [ ] Overall improvement from adding search, in my own words:", "",
    ]
    return "\n".join(lines)
