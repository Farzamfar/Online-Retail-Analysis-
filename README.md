# Online Retail Analysis

End-to-end machine learning project using the UCI Online Retail dataset
(13 months, ~400k transactions from a UK-based e-commerce store).

---

## Notebooks

### 1. `online_retail_analysis_final.ipynb` — Regression
Predicting future customer spend using past transaction behavior.

- Data cleaning and exploratory analysis
- RFM feature engineering (Recency, Frequency, Monetary)
- Temporal train/test split to avoid data leakage
- Baseline vs Full Linear Regression vs Random Forest
- Key lesson: data leakage and how to detect and fix it

| Model | R² |
|---|---|
| Baseline LR (Recency + Frequency) | 0.30 |
| Full Linear Regression (7 features) | 0.52 |
| Random Forest (7 features) | 0.50 |

---

### 2. `online_retail_classification.ipynb` — Classification + Calibration
Classifying customers into Low / Medium / High spenders.

- 13 features: RFM, tenure, weekend/morning shopping, product diversity, active weeks
- Logistic Regression vs SVM (7 and 13 features)
- SVM probability calibration — Sigmoid vs Isotonic
- Precision@K analysis — business-ready targeting metric
- Temporal split throughout — no data leakage

| Model | Accuracy | F1 |
|---|---|---|
| LR — 7 features | 0.567 | 0.561 |
| SVM — 7 features | 0.573 | 0.573 |
| SVM — 13 features | 0.583 | 0.581 |
| SVM — Sigmoid calibrated | **0.586** | 0.577 |
| SVM — Isotonic calibrated | 0.583 | **0.582** |

#### Precision@K — headline business result
If you rank customers by predicted High spender probability and target the top K:

| Target top K | % actually High spenders |
|---|---|
| Top 10 | 90% |
| Top 20 | 85% |
| Top 50 | 82% |
| Top 100 | 70% |
| Random baseline | 33% |

Targeting the top 20 customers is **2.6x more precise than random**.

---

## Key Lessons Learned

- **Data leakage** — early regression models hit R²=0.97 using same-period features.
  Temporal split brought it to a real 0.52
- **Correlation ≠ importance** — features with high raw correlation can be useless
  inside a model due to overlap with other features
- **Calibration matters** — SVM outputs decision scores, not real probabilities.
  Sigmoid calibration makes probabilities trustworthy for ranking and targeting
- **Precision@K beats accuracy** — for business decisions like marketing targeting,
  how accurately you identify the top K customers matters more than overall accuracy
- **Medium segment is hardest** — soft segment boundaries make it inherently ambiguous

## Stack
Python, pandas, scikit-learn, matplotlib, seaborn

## Dataset
UCI Online Retail Dataset:
https://archive.ics.uci.edu/dataset/352/online+retail
