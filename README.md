# HR Workforce Attrition Analytics

## Business problem

People analytics helps HR understand where attrition is concentrated — by department and tenure — and whether it can be predicted, so retention action can be targeted.

## Questions

1. What is the overall attrition rate?
2. Which departments and tenure bands have the highest attrition?
3. How well does a simple predictive model perform?

## Dataset

**Synthetic data only** — generated in `analysis.py` with a fixed seed (42): 2,000 employees with department, tenure, salary, job satisfaction and overtime. Saved to `data/workforce_data_synthetic.csv`.

## Method

- Descriptive attrition rates by department and tenure band
- Logistic Regression (standardised features, 75/25 stratified split), evaluated with accuracy and ROC-AUC

## Key findings (actual run results)

From `results/results_summary.txt`:

- Overall attrition rate: **0.2445 (24.45%)**
- Highest attrition by department: **Sales, 0.2722**; lowest: Finance, 0.2171
- Attrition by tenure band: 0–2 years 0.3188, 3–5 years 0.3459, 6–10 years 0.2370, 10+ years 0.1894 — early-tenure employees leave most in this synthetic dataset
- Logistic Regression: **accuracy = 0.7620, ROC-AUC = 0.6605**

## Tech stack

Python · pandas · NumPy · scikit-learn

## How to run

```bash
pip install -r requirements.txt
python3 analysis.py
```

## Repository structure

```
├── analysis.py
├── data/
├── results/results_summary.txt
├── requirements.txt
└── LICENSE
```

## License

MIT — see `LICENSE`. Copyright (c) 2026 Varshitha Konakala.
