#!/usr/bin/env python
"""
Run the Researcher -> Writer -> Reviewer crew once.

Usage:
    python run_crew.py                              # without search (baseline)
    python run_crew.py --search                     # with web search
    python run_crew.py --search --topic "your topic" --output result.md

Prerequisites: Ollama running with the model pulled (ollama pull llama3.2:3b),
and `pip install -r requirements.txt`.
"""

import argparse
import sys
from pathlib import Path

from research_crew.config import DEFAULT_TOPIC, CrewConfig
from research_crew.crew import run_crew


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--search", action="store_true", help="give the Researcher the web search tool")
    p.add_argument("--model", default=CrewConfig.model, help="e.g. ollama/llama3.2:3b")
    p.add_argument("--output", default=None, help="where to save the final text")
    args = p.parse_args()

    config = CrewConfig(model=args.model)
    print(f"Topic: {args.topic}\nModel: {config.model} | web search: {'ON' if args.search else 'OFF'}\n")
    try:
        text = run_crew(args.topic, args.search, config)
    except ImportError as exc:
        print(f"ERROR: {exc}\nInstall dependencies: pip install -r requirements.txt")
        return 1

    out = Path(args.output or ("output_with_search.md" if args.search else "output_no_search.md"))
    out.write_text(text, encoding="utf-8")
    print(f"\n{'=' * 60}\nFINAL OUTPUT\n{'=' * 60}\n{text}\n\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
