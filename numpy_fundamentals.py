"""
W1D1: Python for ML — NumPy Fundamentals
Task: array creation, broadcasting, vectorised ops, matrix multiplication,
and descriptive statistics on a real CSV dataset.

Author: Siriyala Nishar
"""

from pathlib import Path
import numpy as np
import pandas as pd


def create_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create and return a 1D, 2D, and 3D NumPy array, printing their shapes.

    Why: demonstrates np.array() construction and the .shape attribute,
    which is the first thing you check when debugging shape-mismatch errors.
    """
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    print("1D shape:", arr_1d.shape)  # (5,)
    print("2D shape:", arr_2d.shape)  # (2, 3)
    print("3D shape:", arr_3d.shape)  # (2, 2, 2)

    return arr_1d, arr_2d, arr_3d


def broadcasting_and_vectorised_ops(arr_2d: np.ndarray) -> np.ndarray:
    """Demonstrate broadcasting (no loops) and vectorised arithmetic.

    Why: broadcasting lets NumPy apply an operation across arrays of
    different shapes without writing an explicit Python loop — this is
    both faster and more readable than manual iteration.
    """
    # Broadcasting: a (1,3) row vector is "stretched" to match (2,3)
    row_vector = np.array([10, 20, 30])
    broadcasted = arr_2d + row_vector
    print("Broadcasted result:\n", broadcasted)

    # Vectorised operation: applied to every element at once, no loop
    squared = arr_2d ** 2
    print("Element-wise square:\n", squared)

    return broadcasted


def matrix_multiplication(arr_2d: np.ndarray) -> np.ndarray:
    """Perform true matrix multiplication using @ (or np.matmul).

    Why: `*` does element-wise multiplication; `@` does actual linear-algebra
    matrix multiplication. Mixing these up is a classic ML bug.
    """
    # arr_2d is (2,3); we need a (3,2) matrix to multiply validly
    other = np.array([[1, 2], [3, 4], [5, 6]])
    result = arr_2d @ other  # (2,3) @ (3,2) -> (2,2)
    print("Matrix multiplication result:\n", result)
    return result


def dataset_statistics(csv_path: Path) -> dict[str, float]:
    """Load a CSV and compute mean, std, and correlation using NumPy.

    Why: real ML work rarely uses toy arrays — you load a dataset,
    check its distribution (mean/std), and check relationships
    between variables (correlation) before doing anything else.
    """
    df = pd.read_csv(csv_path)

    study_hours = df["study_hours"].to_numpy()
    exam_score = df["exam_score"].to_numpy()

    stats = {
        "study_hours_mean": float(np.mean(study_hours)),
        "study_hours_std": float(np.std(study_hours)),
        "exam_score_mean": float(np.mean(exam_score)),
        "exam_score_std": float(np.std(exam_score)),
        # np.corrcoef returns a 2x2 matrix; [0,1] is the correlation
        # between the two variables
        "study_vs_score_correlation": float(
            np.corrcoef(study_hours, exam_score)[0, 1]
        ),
    }

    for key, value in stats.items():
        print(f"{key}: {value:.3f}")

    return stats


def main() -> None:
    arr_1d, arr_2d, arr_3d = create_arrays()
    broadcasting_and_vectorised_ops(arr_2d)
    matrix_multiplication(arr_2d)

    csv_path = Path("student_performance.csv")
    dataset_statistics(csv_path)


if __name__ == "__main__":
    main()
