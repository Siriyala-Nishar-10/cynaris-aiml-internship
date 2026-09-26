"""
W4D6: Responsible AI — Real UCI Adult Dataset Edition
Tasks covered:
1. Fetch and process the real UCI Adult Income dataset
2. Compute Baseline Disparate Impact by gender using AIF360
3. Apply Reweighing pre-processing mitigation and measure the improvement
4. Train a LogisticRegression classifier and generate a SHAP summary plot
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import shap
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

print("Fetching real UCI Adult Income dataset from OpenML (this may take a few seconds)...")
adult_raw = fetch_openml('adult', version=2, as_frame=True, parser='auto')
df = adult_raw.frame

print("Preprocessing features to align with AIF360 standards...")
features = ['age', 'education-num', 'sex', 'class']
df_subset = df[features].dropna().copy()

df_subset['income_bin'] = df_subset['class'].astype(str).str.contains('>50K').astype(int)
df_subset['gender_bin'] = df_subset['sex'].astype(str).str.strip().map({'Male': 1, 'Female': 0})
df_subset = df_subset.dropna()

data_clean = pd.DataFrame({
    'age': df_subset['age'].astype(float),
    'education_num': df_subset['education-num'].astype(float),
    'gender': df_subset['gender_bin'].astype(float),
    'income': df_subset['income_bin'].astype(float)
})

unprivileged_groups = [{'gender': 0.0}]
privileged_groups = [{'gender': 1.0}]

aif_dataset = BinaryLabelDataset(
    df=data_clean,
    label_names=['income'],
    protected_attribute_names=['gender'],
    favorable_label=1.0,
    unfavorable_label=0.0
)

metric_orig = BinaryLabelDatasetMetric(
    aif_dataset, 
    unprivileged_groups=unprivileged_groups, 
    privileged_groups=privileged_groups
)
print(f"\nBaseline Disparate Impact: {metric_orig.disparate_impact():.4f}")
print("*(Values below 0.8 indicate substantial systemic bias under the 80% rule)*\n")

print("Applying Reweighing pre-processing mitigation strategy...")
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_transformed = RW.fit_transform(aif_dataset)

metric_trans = BinaryLabelDatasetMetric(
    dataset_transformed, 
    unprivileged_groups=unprivileged_groups, 
    privileged_groups=privileged_groups
)
print(f"Mitigated Disparate Impact (After Reweighing): {metric_trans.disparate_impact():.4f}\n")

X = data_clean[['age', 'education_num', 'gender']]
y = data_clean['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression Classifier...")
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

print("Computing SHAP feature attribution explanations...")
explainer = shap.LinearExplainer(model, X_train)
shap_values_obj = explainer(X_test)
shap_values = shap_values_obj.values

print("\nDisplaying SHAP Summary Plot...")
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot: Feature Attributions on Real UCI Adult Dataset", fontsize=11, pad=15)
plt.tight_layout()
plt.show()

print("\nComplete assignment pipeline executed cleanly with real data!")

