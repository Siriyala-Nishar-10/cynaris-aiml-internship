
import mlflow

from .config import (
    MLFLOW_TRACKING_URI,
    CHAT_MODEL,
    EMBED_MODEL,
)
from .rag_pipeline import TrackedRAG
from .crew_review import review_answer


def add_context_warnings(answer: str, contexts: list[str]) -> str:
    """
    Add important security warnings only when they are explicitly
    present in retrieved context and missing from the answer.
    """

    updated_answer = answer.strip()

    for context in contexts:
        sentences = context.replace("\n", " ").split(". ")

        for sentence in sentences:
            warning = sentence.strip().rstrip(".")

            lower_warning = warning.lower()

            # Only consider explicit security-related warnings.
            is_security_warning = (
                any(word in lower_warning for word in (
                    "never share",
                    "do not share",
                    "don't share",
                    "never provide",
                    "do not provide",
                ))
                and any(word in lower_warning for word in (
                    "password",
                    "verification code",
                    "one-time code",
                    "recovery link",
                ))
            )

            if not is_security_warning:
                continue

            # Avoid adding a warning that is already covered.
            warning_words = [
                word for word in (
                    "password",
                    "verification code",
                    "one-time code",
                    "recovery link",
                )
                if word in lower_warning
            ]

            already_covered = all(
                word in updated_answer.lower()
                for word in warning_words
            )

            if not already_covered:
                updated_answer += " " + warning + "."

    return updated_answer


def run_tracked_question(question: str) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("W11D5-Tracked-Evaluated-RAG")

    rag = TrackedRAG()

    with mlflow.start_run(run_name="langgraph-rag-question"):
        mlflow.log_param("chat_model", CHAT_MODEL)
        mlflow.log_param("embedding_model", EMBED_MODEL)
        mlflow.log_param("retrieval_k", 3)

        # Step 1: Retrieve context and generate the initial answer.
        result = rag.ask(question)

        # Step 2: Ensure explicit security warnings are not omitted.
        result["answer"] = add_context_warnings(
            result["answer"],
            result["contexts"],
        )

        # Step 3: Review the completed answer.
        review = review_answer(
            result["question"],
            result["answer"],
            result["contexts"],
        )

        result["crew_review"] = review

        # Step 4: Track evaluation evidence in MLflow.
        mlflow.log_metric(
            "retrieved_context_count",
            len(result["contexts"]),
        )

        mlflow.log_text(result["answer"], "answer.txt")
        mlflow.log_text(review, "crew_review.txt")
        mlflow.log_text(
            "\n---\n".join(result["contexts"]),
            "retrieved_contexts.txt",
        )

        return result
