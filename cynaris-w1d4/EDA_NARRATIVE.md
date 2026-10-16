# EDA Narrative — Employee Dataset

The dataset contains 300 employee records across 7 columns: employee ID,
age, department, city, experience, monthly salary, and a satisfaction
score. Overall the data looks reasonably clean and realistic — age
ranges from 21 to 47 with a mean around 32, and department distribution
is led by Sales (25%) followed by a fairly even spread across Marketing,
Finance, and Operations.

Two issues stand out. First, **satisfaction_score has 15 missing values
(5% of rows)** — likely unanswered survey responses rather than a
systemic data problem, since the missingness doesn't concentrate in any
one department or city. Second, and more concerning, is a **clear
outlier in monthly_salary**: one employee is recorded at ₹500,000/month,
nearly 5x the 75th percentile (₹97,625) and far outside the rest of the
distribution. This is almost certainly a data entry error (e.g. an extra
zero) rather than a legitimate salary, and should be corrected or
removed before any downstream analysis, since it would badly skew mean
salary calculations and any model trained on this feature.

The correlation heatmap confirms an expected strong positive
relationship between age and experience_years, which makes sense since
experience is generally derived from age. Monthly salary also correlates
positively with experience, as expected in most compensation structures.
Satisfaction_score shows weak correlation with the other numeric
variables, suggesting it's driven by factors not captured in this
dataset (e.g. management quality, work-life balance) rather than by
salary or tenure alone — worth flagging as a limitation if this dataset
were used for a "what drives satisfaction" analysis.

**What needs fixing before further use:** correct or remove the salary
outlier, and decide on an imputation strategy for the missing
satisfaction scores (median imputation, given no strong distributional
skew, would be reasonable here).
