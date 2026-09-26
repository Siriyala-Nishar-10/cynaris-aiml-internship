# Week 4 Day 6: Responsible AI — Bias, Fairness, and Transparency

This repository contains the complete assignment codebase and technical documentation for auditing and mitigating demographic bias using the UCI Adult Income dataset. The pipeline evaluates algorithmic equity across biological sex demographics using IBM AI Fairness 360 (AIF360) and provides global model interpretability via SHAP (SHapley Additive exPlanations).

## Workspace Structure

- `w4d6_real_aif360_shap.py`: Automated Python script for data ingestion, bias assessment, mitigation execution, and SHAP visualization.
- `model_card.pdf`: Technical model card reporting operational bounds, performance metrics, and ethical validation parameters.
- `india_bias_case_analysis.txt`: Structured 200-word field analysis detailing systemic representation bias within domestic biometric infrastructure.

## System Dependencies and Setup

This workspace is fully optimized for Python 3.14 setups. Run the following command inside your integrated terminal window to pull the required library bundles directly:

```bash
pip install aif360 shap pandas scikit-learn matplotlib numpy
```

_Note: If your local setup encounters a C++ compiler error during the SHAP build sequence, utilize pre-compiled binaries or run the command inside an active Conda environment using `conda install -c conda-forge shap`._

## Execution Pipeline

Execute the script from the integrated terminal using your virtual environment path:

```bash
python w4d6_real_aif360_shap.py
```

### Underlying Processing Steps

1. **Data Ingestion:** The script contacts OpenML via `fetch_openml` to download the official adult income benchmark data dynamically.
2. **Feature Mapping:** Column indexes map structural labels (`age`, `education-num`, `sex`, and `class`) to integer frameworks (`1.0` for historically privileged/favorable parameters, `0.0` for unprivileged/unfavorable bounds).
3. **Baseline Audit (Task 1):** The dataset passes into the `BinaryLabelDatasetMetric` matrix to quantify systemic selection imbalances, outputting an initial Disparate Impact ratio of **0.3568**.
4. **Algorithmic Mitigation (Task 3):** The `Reweighing` preprocessing instance algorithm runs across the cohort arrays, scaling sample weight weights dynamically to return a corrected Disparate Impact score of **1.0000**.
5. **Model Training & Explanation (Task 2):** A scikit-learn Logistic Regression model fits the transformed weights. The system feeds the network parameter states to `shap.LinearExplainer` to plot global feature importance distributions via a Matplotlib window canvas.

## Verification of Technical Outputs

- **Baseline Disparate Impact Ratio:** 0.3568 (Violates the standard regulatory four-fifths fairness threshold of 0.80).
- **Mitigated Disparate Impact Ratio:** 1.0000 (Achieves absolute statistical parity across monitored demographics).
- **SHAP Feature Hierarchy Order:** `education_num` registers as the primary predictive factor driving positive outputs, followed sequentially by `gender` allocations and chronological `age` values.

## Author

Siriyala Nishar
