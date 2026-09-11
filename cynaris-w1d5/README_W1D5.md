# W1D5 — Data Visualisation: Matplotlib & Seaborn

## What This Does
Builds 5 different visualization types on the employee dataset (reused
from W1D4), each chosen deliberately for what it reveals that a basic
histogram or bar chart wouldn't.

## Tech Stack
- Python 3.14
- Pandas, Matplotlib, Seaborn

## Setup
```bash
python -m venv venv
source venv/Scripts/activate
pip install pandas matplotlib seaborn
pip freeze > requirements.txt
```

## Run
```bash
python data_visualisation.py
```

## What Each Function Does & Why That Chart Type
- `boxplot_salary_by_department()` — boxplot shows median, quartiles,
  and outliers per department at once; a bar chart of means alone
  would hide the salary spread and outlier.
- `scatter_experience_vs_salary()` — scatter plot is correct for two
  continuous variables; colour-coding by department reveals whether
  the experience-salary relationship differs across groups.
- `violin_satisfaction_by_department()` — violin plot shows full
  distribution shape (not just quartiles), useful for spotting
  bimodal patterns a boxplot would miss.
- `bar_average_salary_by_city()` — sorted bar chart is clearest for
  ranking categories; sorting (not alphabetical) makes extremes
  immediately visible.
- `pairplot_numeric_relationships()` — fast overview of every pairwise
  numeric relationship at once, useful before deciding which single
  relationship deserves a dedicated, detailed chart.

Note: the known salary outlier (₹500,000, identified in W1D4) is
excluded from salary-related plots to avoid distorting the scale —
noted explicitly in code rather than silently dropped.

## Output Files
- `salary_by_department_boxplot.png`
- `experience_vs_salary_scatter.png`
- `satisfaction_violin.png`
- `avg_salary_by_city_bar.png`
- `pairplot.png`

## AI Usage Note
Built with AI assistance (Claude) for chart type selection and code
structure. Reviewed and tested locally — verified each plot renders
correctly and matches the reasoning documented above.

## Author
Siriyala Nishar
