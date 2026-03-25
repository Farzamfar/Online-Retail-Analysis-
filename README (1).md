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
- Key lesson: data leakage — how to detect and fix it

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
Rank customers by predicted High spender probability and target the top K:

| Target top K | % actually High spenders |
|---|---|
| Top 10 | 90% |
| Top 20 | 85% |
| Top 50 | 82% |
| Top 100 | 70% |
| Random baseline | 33% |

Targeting the top 20 customers is **2.6x more precise than random**.

---

### 3. `online_retail_unsupervised.ipynb` — Unsupervised Learning
Discovering natural customer groups without predefined labels.

- Full transaction history used (no temporal split needed)
- 14 features: RFM, order patterns, behavioral, product diversity
- PCA → K-Means → Hierarchical Clustering → DBSCAN

#### PCA
- 5 components explain 80% of variance (14 features → 5)
- PC1 driven by engagement (Frequency, Active Weeks, Monetary)
- PC2 driven by order size (Avg Order Value, Max Order)
- Weekend and morning shopping carry almost no signal

#### K-Means — natural clusters found (K=2)

| Cluster | Customers | Recency | Frequency | Avg Spend |
|---|---|---|---|---|
| Active | 2,131 | 39 days | 7.2 orders | £3,710 |
| Lapsed | 2,207 | 143 days | 1.4 orders | £455 |

#### DBSCAN — outlier detection
- Confirmed the 2 main clusters
- Identified **82 VIP customers (1.9%)** averaging **£18,709 spend** — 10x the normal average
- These VIP customers were invisible in the supervised classification

---

## Key Lessons Learned

- **Data leakage** — early regression models hit R²=0.97 using same-period features.
  Temporal split brought it to a real 0.52
- **Correlation ≠ importance** — features with high raw correlation can be useless
  inside a model due to overlap with other features
- **Calibration matters** — SVM outputs decision scores not real probabilities.
  Sigmoid calibration makes probabilities trustworthy for ranking and targeting
- **Precision@K beats accuracy** — for business targeting, how accurately you
  identify the top K customers matters more than overall accuracy
- **Unsupervised reveals what supervised misses** — equal-thirds classification
  forced three segments. Unsupervised found the data naturally has two groups
  plus a small VIP outlier group worth treating separately
- **PCA as a feature selector** — weekend and morning shopping features looked
  useful but PCA revealed they carry almost no real signal

---

## Stack
Python, pandas, scikit-learn, scipy, matplotlib, seaborn

## Dataset
UCI Online Retail Dataset:
https://archive.ics.uci.edu/dataset/352/online+retail
