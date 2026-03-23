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

## Key lesson — data leakage
Early versions achieved R²=0.97 using same-period features.
The correct temporal approach gives R²=0.52 — a real,
trustworthy result for predicting future customer behavior.

## Stack
Python, pandas, scikit-learn, matplotlib, seaborn

## Dataset
UCI Online Retail Dataset:
https://archive.ics.uci.edu/dataset/352/online+retail
