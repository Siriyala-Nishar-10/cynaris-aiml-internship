#!/usr/bin/env python
"""
W9D3: run the Researcher -> Writer -> Reviewer pipeline without and with web search,
save every stage's output, and write a report documenting the difference.

Usage:
    python run_pipeline.py
    python run_pipeline.py --topic "your topic" --model ollama/llama3.2:3b
    python run_pipeline.py --only with_search

Writes:
    runs/no_search/   research.md, draft.md, final.md, summary.json
    runs/with_search/ research.md, draft.md, final.md, summary.json, search_log.json
    pipeline_report.md

Prerequisites: Ollama running with the model pulled; `pip install -r requirements.txt`.
Each run takes several minutes on a small local model. Do not interrupt it.
"""

import argparse
import sys
from pathlib import Path

from research_crew.config import DEFAULT_TOPIC, CrewConfig
from research_crew.pipeline import build_pipeline_report, run_mode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--model", default=CrewConfig.model)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--report", default="pipeline_report.md")
    p.add_argument("--only", choices=["no_search", "with_search"], default=None,
                   help="run just one mode (the report then needs both, so it is skipped)")
    args = p.parse_args()
    config = CrewConfig(model=args.model)

    summaries = {}
    try:
        for with_search in (False, True):
            mode = "with_search" if with_search else "no_search"
            if args.only and args.only != mode:
                continue
            print(f"\n=== Running pipeline: {mode} ===")
            summaries[mode] = run_mode(args.topic, with_search, config, args.runs_dir)
            s = summaries[mode]
            print("FAILED: " + s["error"] if s.get("error") else
                  f"done in {s['duration_s']}s | rubric passed: {s['quality']['passed']}")
    except ImportError as exc:
        print(f"ERROR: {exc}\nInstall dependencies: pip install -r requirements.txt")
        return 1

    if len(summaries) == 2:
        Path(args.report).write_text(build_pipeline_report(
            args.topic, config.model, summaries["no_search"], summaries["with_search"]), encoding="utf-8")
        print(f"\nSaved {args.report} - now complete its 'Manual checks' section.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
