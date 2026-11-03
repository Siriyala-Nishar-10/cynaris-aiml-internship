"""CLI + MLflow tracking. Every `ask` is one logged, reproducible run."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .memory import VectorMemory, ingest_folder


def ask(question: str, cfg: Settings, evaluate: bool) -> str:
    from .service import answer_question

    return answer_question(question, cfg, VectorMemory(cfg.store_path), evaluate)["answer"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="research-assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("folder", type=Path)
    q = sub.add_parser("ask")
    q.add_argument("question")
    q.add_argument("--no-eval", action="store_true", help="skip Ragas scoring")
    args = parser.parse_args()

    cfg = Settings()
    if args.cmd == "ingest":
        print(f"Indexed {ingest_folder(args.folder, VectorMemory(cfg.store_path), cfg)} chunks.")
    else:
        print(ask(args.question, cfg, evaluate=not args.no_eval))


if __name__ == "__main__":
    main()
