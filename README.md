# Online Retail Analysis

End-to-end machine learning project predicting future customer spend
using the UCI Online Retail dataset (13 months, ~400k transactions).

## What's inside
- Data cleaning and exploratory analysis
- RFM feature engineering (Recency, Frequency, Monetary)
- Temporal train/test split to avoid data leakage
- Baseline vs Full Linear Regression vs Random Forest
- Final R² = 0.52 on genuine future spend prediction

## Models compared
| Model | R² |
|---|---|
| Baseline LR (Recency + Frequency) | 0.30 |
| Full Linear Regression (7 features) | 0.52 |
| Random Forest (7 features) | 0.50 |



### 2. `online_retail_classification.ipynb` — Classification
Classifying customers into Low / Medium / High spenders.

- 13 features including RFM, tenure, shopping patterns, product diversity
- Logistic Regression vs SVM with 7 and 13 features
- Temporal split — features from past, target from future
- Per-class performance analysis

| Model | Accuracy | F1 |
|---|---|---|
| LR — 7 features | 0.567 | 0.561 |
| LR — 13 features | 0.562 | 0.555 |
| SVM — 7 features | 0.573 | 0.573 |
| SVM — 13 features | 0.583 | 0.581 |

---

## Key Lessons Learned
- **Data leakage** — early models hit R²=0.97 using same-period features.
  Temporal split brought it to a real 0.52
- **Correlation ≠ importance** — features with high raw correlation
  can be useless inside a model due to overlap with other features
- **Simpler models generalise better** — Linear Regression matched
  Random Forest because log transformation resolved most non-linearity
- **Medium segment is hardest to classify** — soft boundaries between
  segments make it inherently ambiguous

## Stack
Python, pandas, scikit-learn, matplotlib, seaborn

## Dataset
UCI Online Retail Dataset:
https://archive.ics.uci.edu/dataset/352/online+retail
