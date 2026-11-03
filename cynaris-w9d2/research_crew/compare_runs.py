#!/usr/bin/env python
"""
Task 3: run the crew without and with web search, then write a comparison report.

Usage:  python -m research_crew.compare_runs [--topic "..."]
Writes: output_no_search.md, output_with_search.md, comparison_metrics.json, comparison_report.md
A failed run is recorded in the report instead of aborting, so one flaky run
(common with small local models) does not lose the other.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from research_crew.compare import build_report, output_metrics
from research_crew.config import DEFAULT_TOPIC, CrewConfig
from research_crew.crew import run_crew


def run_one(topic, with_search, config, path):
    label = "WITH" if with_search else "WITHOUT"
    print(f"\n=== Running crew {label} web search ===")
    try:
        text = run_crew(topic, with_search, config)
    except ImportError:
        raise
    except Exception as exc:                      # model/tool/network failure: record and continue
        print(f"Run failed: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}", "metrics": output_metrics("")}
    Path(path).write_text(text, encoding="utf-8")
    return {"text": text, "metrics": output_metrics(text)}


def main() -> int:
    # Fix for Windows asyncio event loop issues with CrewAI and local models
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--model", default=CrewConfig.model)
    args = p.parse_args()
    config = CrewConfig(model=args.model)
    try:
        without = run_one(args.topic, False, config, "output_no_search.md")
        with_ = run_one(args.topic, True, config, "output_with_search.md")
    except ImportError as exc:
        print(f"ERROR: {exc}\nInstall dependencies: pip install -r requirements.txt")
        return 1

    Path("comparison_metrics.json").write_text(json.dumps(
        {"topic": args.topic, "model": config.model,
         "without_search": {k: v for k, v in without.items() if k != "text"},
         "with_search": {k: v for k, v in with_.items() if k != "text"}}, indent=2))
    Path("comparison_report.md").write_text(build_report(args.topic, config.model, without, with_), encoding="utf-8")
    print("\nSaved comparison_report.md - now complete its 'Manual checks' section.")
    return 0


if __name__ == "__main__":
    sys.exit(main())