"""Entry point for the W9D5 Automated Research Report Agent."""
from src.workflow import run_research_workflow


def main():
    topic = input("Research topic: ").strip()
    if not topic:
        raise SystemExit("Please enter a research topic.")
    result = run_research_workflow(topic)
    print("\n=== RESEARCH REPORT ===\n")
    print(result["report"])
    print(f"\nMLflow run ID: {result.get('run_id', 'not recorded')}")
    print(f"Saved report: {result['report_path']}")


if __name__ == "__main__":
    main()
