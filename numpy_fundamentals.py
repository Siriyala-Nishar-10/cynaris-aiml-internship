"""
W1D1: Python for ML - NumPy Fundamentals
Task: array creation, broadcasting, vectorised ops, matrix multiplication,
and descriptive statistics on a real CSV dataset.

Author: Siriyala Nishar
"""

from pathlib import Path
import numpy as np
import pandas as pd


def create_arrays():
    """Create and return a 1D, 2D, and 3D NumPy array, printing their shapes."""
    arr_1d = np.array([1, 2, 3, 4, 5])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    print("1D shape:", arr_1d.shape)
    print("2D shape:", arr_2d.shape)
    print("3D shape:", arr_3d.shape)

    return arr_1d, arr_2d, arr_3d


def broadcasting_and_vectorised_ops(arr_2d):
    """Demonstrate broadcasting (no loops) and vectorised arithmetic."""
    row_vector = np.array([10, 20, 30])
    broadcasted = arr_2d + row_vector
    print("Broadcasted result:\n", broadcasted)

    squared = arr_2d ** 2
    print("Element-wise square:\n", squared)

    return broadcasted


def matrix_multiplication(arr_2d):
    """Perform true matrix multiplication using @ (not element-wise *)."""
    other = np.array([[1, 2], [3, 4], [5, 6]])
    result = arr_2d @ other
    print("Matrix multiplication result:\n", result)
    return result


def dataset_statistics(csv_path):
    """Load a CSV and compute mean, std, and correlation using NumPy."""
    df = pd.read_csv(csv_path)

    study_hours = df["study_hours"].to_numpy()
    exam_score = df["exam_score"].to_numpy()

    stats = {
        "study_hours_mean": float(np.mean(study_hours)),
        "study_hours_std": float(np.std(study_hours)),
        "exam_score_mean": float(np.mean(exam_score)),
        "exam_score_std": float(np.std(exam_score)),
        "study_vs_score_correlation": float(
            np.corrcoef(study_hours, exam_score)[0, 1]
        ),
    }

    for key, value in stats.items():
        print(f"{key}: {value:.3f}")

    return stats


def main():
    arr_1d, arr_2d, arr_3d = create_arrays()
    broadcasting_and_vectorised_ops(arr_2d)
    matrix_multiplication(arr_2d)

    csv_path = Path("student_performance.csv")
    dataset_statistics(csv_path)


if __name__ == "__main__":
    main()