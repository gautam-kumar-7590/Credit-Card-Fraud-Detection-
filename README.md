# Credit Card Fraud Detection

> End-to-End Machine Learning Project | Python · XGBoost · Isolation Forest · Power BI

**Author:** Gautam Kumar Kanojia | **GitHub:** [gautam-kumar-7590](https://github.com/gautam-kumar-7590) | **LinkedIn:** [GautamKumarKanojia](https://www.linkedin.com/in/GautamKumarKanojia) | progautam54@gmail.com

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat&logo=python&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-Supervised-orange?style=flat) ![IsolationForest](https://img.shields.io/badge/Isolation%20Forest-Unsupervised-purple?style=flat) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Manipulation-150458?style=flat) ![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat) ![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c8cbf?style=flat) ![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=flat) ![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-green?style=flat) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat) ![VS Code](https://img.shields.io/badge/VS%20Code-IDE-007ACC?style=flat) ![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9998-brightgreen?style=flat) ![Fraud Recall](https://img.shields.io/badge/Fraud%20Recall-1.00-brightgreen?style=flat) ![Dataset](https://img.shields.io/badge/Dataset-6.3M%20Transactions-red?style=flat) ![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

## 1. Project Overview

A complete end-to-end ML pipeline to detect fraudulent financial transactions on a massively imbalanced dataset of 6.3 million synthetic transactions. Built on the PaySim dataset, this project covers the full data science workflow — EDA, feature engineering, supervised and unsupervised modeling, and a Power BI dark-theme fintech dashboard.

**Final deliverables:** a trained XGBoost model (`.pkl`), an Isolation Forest baseline (`.pkl`), a fraud analytics Power BI dashboard, and a full engineered dataset exported to CSV.

---
![Dashboard](Screenshot%20(486).png)

## 2. Business Problem

> *"With 6.3 million daily transactions and a fraud rate under 0.2%, how do we catch fraudsters before the money is gone — without drowning analysts in false alarms?"*

In fintech and digital payments, fraud detection operates under extreme class imbalance. Missing a fraud case is far more costly than a false positive. This model is calibrated to **maximize recall for fraud (Class 1)**, accepting precision trade-offs to flag as many genuine fraud cases as possible before funds clear.

---
![Imbalance Data](imblanace%20data%20plot.png)

## 3. Dataset

| Attribute | Detail |
|---|---|
| Source | PaySim Synthetic Financial Dataset (Kaggle) |
| Total Transactions | 6,362,620 |
| Fraud Cases | 8,213 (0.13% — severe imbalance) |
| File Size | ~470MB CSV |
| Target Variable | `isFraud` — 0 = Legitimate, 1 = Fraud |
| Fraud Transaction Types | CASH_OUT and TRANSFER only |
| Features | 10 raw columns (step, type, amount, balances, flags) |

Download the dataset from Kaggle: 👉 https://www.kaggle.com/datasets/ealaxi/paysim1

---
![Target Distribution](Target%20plot.png)
## 4. Technology Stack

| Category | Tools & Libraries |
|---|---|
| Language | Python 3.13 |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn |
| Unsupervised Baseline | Isolation Forest (sklearn) |
| Model Evaluation | classification_report, roc_auc_score, ConfusionMatrixDisplay |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Power BI Desktop |
| Model Persistence | Joblib (.pkl) |
| Environment | VS Code + Jupyter Notebook |

---
![Screenshot 440](Screenshot%20(440).png)
## 5. Data Preprocessing & Feature Engineering

- **Windows Path Fix:** Loaded 470MB CSV using raw string path to avoid backslash errors
- **Column Drops:** Removed `isFlaggedFraud`, `nameOrig`, `nameDest` via safe list comprehension
- **Categorical Encoding:** `type` encoded with `pd.get_dummies(drop_first=True)` — no scaling needed for tree-based models
- **Balance Features:** `balance_diff_orig` and `balance_diff_dest` — captures how much money moved relative to starting balance
- **Zero Balance Flags:** `orig_zero_after` and `dest_zero_before` — binary flags for accounts drained to zero or funded from zero
- **Time Feature:** `hour` derived from `step % 24` — captures intraday fraud patterns
- **No Scaling Applied:** Both models are tree-based — scaling adds no value

---

## 6. Models Trained & Comparison

| Model | Type | ROC-AUC | Fraud Recall | Fraud Precision | Notes |
|---|---|---|---|---|---|
| XGBoost | Supervised | **0.9998** | **1.00** | 0.21 | Final model — scale_pos_weight for imbalance |
| Isolation Forest | Unsupervised | — | 0.03 | — | Baseline — contamination=0.0013 |

XGBoost selected as final model. Isolation Forest's 3% recall intentionally demonstrates why labeled fraud data is worth collecting.

---
![XGBoost Feature Importance](XGBOOST%20plot%20acc.png)
![Isolation Forest](iso_forest%20plot.png)
![Screenshot 444](Screenshot%20(444).png)
## 7. Key Challenges & Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| Severe class imbalance | 0.13% fraud rate | `scale_pos_weight` in XGBoost + `stratify=y` in split |
| 470MB file load on Windows | Backslash path issue | Raw string path `r"C:\..."` |
| High false positives | Precision 0.21 — 6,022 FPs | Accepted tradeoff — recall is primary objective |
| Unsupervised baseline failure | No labels → no signal | Isolation Forest at 3% recall proves supervised approach is essential |
| Perfectly correlated features | `oldbalanceOrg` & `newbalanceOrig` at 1.0 | Retained — tree models handle collinearity; engineered deltas add new signal |

---

## 8. Final Model Performance

**XGBoost Configuration:**
```
Algorithm:         XGBoost Classifier
scale_pos_weight:  ~770 (ratio of negatives to positives)
stratify:          y (preserves fraud ratio in train/test split)
Train/Test Split:  80/20
```

**Classification Report (Test Set):**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Legitimate (Class 0) | 1.00 | 1.00 | 1.00 |
| Fraud (Class 1) | 0.21 | 1.00 | 0.35 |
| **ROC-AUC** | | **0.9998** | |

**Key Observations:**
- Model identifies **100% of actual fraud cases** — only 8 of 1,643 fraud cases missed on test set
- ROC-AUC of **0.9998** on a 0.13% imbalanced dataset validates `scale_pos_weight` as the right imbalance strategy over SMOTE
- 6,022 false positives — acceptable where missing fraud is far costlier than a false alert
- `balance_diff_orig` ranked #1 in feature importance, `orig_zero_after` ranked #2 — both engineered features

---
![XGBoost Feature Importance](XGBOOST%20plot%20acc.png)
## 9. Key Insights

- **Fraud is type-specific:** CASH_OUT and TRANSFER account for 100% of all fraud — DEBIT, PAYMENT, CASH_IN have zero fraud cases
- **Fraud peaks at specific hours:** Actionable for real-time alert systems and transaction monitoring
- **Feature engineering beat raw data:** Both top features were engineered — domain knowledge adds real signal
- **Labeled data is worth it:** XGBoost 100% recall vs Isolation Forest 3% recall — direct quantified argument for investing in fraud labeling pipelines

---

## 10. Project Deliverables

| File | Description |
|---|---|
| `CREDIT_CARD_FRAUD_DETECTION_Project_file.ipynb` | Complete Jupyter Notebook with all code, outputs, and analysis |
| `xgb_fraud_model.pkl` | Saved XGBoost model via Joblib |
| `iso_forest_model.pkl` | Saved Isolation Forest baseline model |
| `fraud_dashboard_data.csv` | Full engineered dataset for Power BI |
| `model_comparison.csv` | Precision, recall, F1 for both models |
| `feature_importance.csv` | XGBoost feature importance scores |

---

## 11. How to Run

**Prerequisites:**
```
pip install pandas numpy xgboost scikit-learn matplotlib seaborn joblib
```

1. Clone the repository and navigate to the project folder
2. Download the PaySim dataset from Kaggle and place the CSV in the project folder
3. Open `CREDIT_CARD_FRAUD_DETECTION_Project_file.ipynb` in VS Code or Jupyter and run all cells sequentially
4. Open the Power BI `.pbix` file in Power BI Desktop to explore the dashboard

---

*Gautam Kumar Kanojia* | [GitHub: gautam-kumar-7590](https://github.com/gautam-kumar-7590) | [LinkedIn: GautamKumarKanojia](https://www.linkedin.com/in/GautamKumarKanojia) | progautam54@gmail.com
