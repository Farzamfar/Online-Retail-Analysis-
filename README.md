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

### 3. `online_retail_unsupervised.ipynb` — Unsupervised Learning + Autoencoder
Discovering natural customer groups without predefined labels.
- Full transaction history used (no temporal split needed)
- 14 features: RFM, order patterns, behavioral, product diversity
- Pipeline: PCA → K-Means → Hierarchical Clustering → DBSCAN → Autoencoder

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

#### Autoencoder — deep learning for clustering
A neural network trained to compress 14 features into a 3-dimensional bottleneck
and reconstruct them, forcing it to learn the most meaningful non-linear
representation of each customer.

Architecture: 14 → 8 → **3 (bottleneck)** → 8 → 14

KMeans was then run on the learned 3D embeddings instead of raw RFM features.

| Method | Silhouette Score |
|---|---|
| KMeans on PCA embeddings | 0.282 |
| KMeans on Autoencoder embeddings | **0.424** |

The autoencoder improved cluster quality by **50%** over PCA. Only 41% of customers
were assigned to the same cluster by both methods, confirming the autoencoder
captured non-linear feature interactions that PCA could not represent.

#### Autoencoder cluster profiles

| | Cluster 0 — Bulk Buyers | Cluster 1 — Loyal Browsers |
|---|---|---|
| Recency | 135 days | 80 days |
| Frequency | 5 orders | 4 orders |
| Monetary | £4,561 | £1,398 |
| Unique Products | 25 | 70 |
| Avg Order Value | £266 | £17 |
| Max Order | £636 | £72 |
| Tenure | 83 days | 143 days |

**Cluster 0 — Bulk Buyers:** high spend per order, fewer unique products.
Likely wholesale or business accounts placing large repeat orders.

**Cluster 1 — Loyal Browsers:** longer tenure, more recent, buy many different
products in small quantities. Engaged retail shoppers worth retaining.

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
- **Autoencoders outperform PCA for clustering** — PCA is limited to linear
  relationships. The autoencoder's non-linear compression produced a 50% better
  silhouette score and surfaced a genuinely different customer segmentation,
  demonstrating that deep learning adds value even in unsupervised tasks

---

## Stack
Python, pandas, scikit-learn, scipy, TensorFlow/Keras, matplotlib, seaborn

## Dataset
UCI Online Retail Dataset:
https://archive.ics.uci.edu/dataset/352/online+retail
