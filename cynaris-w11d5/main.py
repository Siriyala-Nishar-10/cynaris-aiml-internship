
import argparse
from src.workflow import run_tracked_question


def main():
    parser = argparse.ArgumentParser(
        description="W11D5 tracked and evaluated RAG"
    )
    parser.add_argument(
        "--question",
        help="Ask one question through the RAG workflow"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the Ragas evaluation set"
    )

    args = parser.parse_args()

    if args.evaluate:
        # Import Ragas only when evaluation is requested.
        from src.evaluate_ragas import run_evaluation
        run_evaluation()

    elif args.question:
        result = run_tracked_question(args.question)

        print("\nQUESTION:", result["question"])
        print("\nANSWER:", result["answer"])
        print("\nCREWAI REVIEW:", result["crew_review"])
        print("\nRETRIEVED CONTEXTS:")

        for context in result["contexts"]:
            print("-", context)

    else:
        parser.error("Provide --question or --evaluate")


if __name__ == "__main__":
    main()
