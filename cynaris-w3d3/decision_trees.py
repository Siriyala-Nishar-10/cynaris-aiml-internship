"""
W3D3: Decision Trees & Random Forests
----------------------------------------
Covers: how trees split using Gini impurity / information gain (entropy),
visualizing a decision tree, overfitting from unrestricted depth, tuning
to control it, and comparing a single tree against a Random Forest.

Run: python w3d3_decision_trees.py
Outputs: prints results to console, saves PNG plots to this folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42


def load_data():
    X_arr, y = make_classification(
        n_samples=800, n_features=20, n_informative=6, n_redundant=6,
        n_classes=3, n_clusters_per_class=1, flip_y=0.08,
        class_sep=0.9, random_state=RANDOM_STATE
    )
    feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
    X = pd.DataFrame(X_arr, columns=feature_names)
    class_names = ["Class 0", "Class 1", "Class 2"]
    return X, y, class_names


def splitting_criteria_demo(X_train, y_train):
    print("=== Splitting Criteria: Gini vs Entropy ===")
    for criterion in ["gini", "entropy"]:
        tree = DecisionTreeClassifier(criterion=criterion, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        print(f"{criterion:10s} -> tree depth: {tree.get_depth()}, "
              f"leaves: {tree.get_n_leaves()}, "
              f"train accuracy: {tree.score(X_train, y_train):.4f}")


def overfitting_demo(X_train, X_test, y_train, y_test):
    print("\n=== Overfitting: Unrestricted Depth vs Limited Depth ===")

    unrestricted = DecisionTreeClassifier(random_state=RANDOM_STATE)
    unrestricted.fit(X_train, y_train)
    print(f"Unrestricted tree (depth={unrestricted.get_depth()}): "
          f"train acc={unrestricted.score(X_train, y_train):.4f}, "
          f"test acc={unrestricted.score(X_test, y_test):.4f}")

    limited = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    limited.fit(X_train, y_train)
    print(f"Limited tree (max_depth=3):        "
          f"train acc={limited.score(X_train, y_train):.4f}, "
          f"test acc={limited.score(X_test, y_test):.4f}")

    return limited


def tune_tree(X_train, y_train):
    print("\n=== Hyperparameter Tuning (GridSearchCV, 5-fold) ===")
    param_grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def visualize_tree(tree, feature_names, class_names):
    plt.figure(figsize=(18, 10))
    plot_tree(
        tree, feature_names=feature_names, class_names=list(class_names),
        filled=True, rounded=True, fontsize=8
    )
    plt.title("Decision Tree (tuned)")
    plt.tight_layout()
    plt.savefig("decision_tree_visualization.png", dpi=120)
    plt.close()
    print("Saved decision_tree_visualization.png")


def feature_importance_demo(tree, feature_names, title, filename):
    importances = pd.Series(tree.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(8)

    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="seagreen")
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Saved {filename}")
    return importances


def random_forest_comparison(X_train, X_test, y_train, y_test, tuned_tree):
   
    print("\n=== Single Tree vs Random Forest ===")

    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    tree_pred = tuned_tree.predict(X_test)
    rf_pred = rf.predict(X_test)

    results = pd.DataFrame([
        {"Model": "Tuned Decision Tree", "Test Accuracy": accuracy_score(y_test, tree_pred)},
        {"Model": "Random Forest (200 trees)", "Test Accuracy": accuracy_score(y_test, rf_pred)},
    ])
    print(results.to_string(index=False))

    print("\nRandom Forest classification report:")
    print(classification_report(y_test, rf_pred))

    return rf, results


def main():
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    splitting_criteria_demo(X_train, y_train)
    overfitting_demo(X_train, X_test, y_train, y_test)
    tuned_tree = tune_tree(X_train, y_train)

    visualize_tree(tuned_tree, X.columns, class_names)
    feature_importance_demo(
        tuned_tree, X.columns,
        "Decision Tree - Feature Importance", "tree_feature_importance.png"
    )

    rf, results = random_forest_comparison(X_train, X_test, y_train, y_test, tuned_tree)
    feature_importance_demo(
        rf, X.columns,
        "Random Forest - Feature Importance", "rf_feature_importance.png"
    )

    print("\n=== Takeaway ===")
    print("Trees split greedily on the feature/threshold that most reduces")
    print("Gini impurity (or increases information gain via entropy) at each")
    print("node. Left unrestricted, a tree memorizes training data (overfits) —")
    print("limiting depth/min_samples trades some train accuracy for better")
    print("generalization. Random Forest reduces overfitting further by")
    print("averaging many trees trained on random data/feature subsets,")
    print("usually improving test accuracy at the cost of easy interpretability.")


if __name__ == "__main__":
    main()
