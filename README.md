# 💳 Credit Card Fraud Detection: End-to-End Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2C2C2C?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=power-bi&logoColor=black)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Blue?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 1. Project Overview
This project addresses high-frequency financial fraud detection using the **PaySim synthetic dataset**. With over **6.3 million transactions**, the challenge was to navigate an extreme class imbalance (0.13%) to build a model capable of identifying fraudulent transfers and cash-outs with near-perfect recall.
## 📊 Dashboard Preview
![Financial Fraud Dashboard](Screenshot%20(487).jpg)
## 2. Dataset Description
* **Source:** PaySim Synthetic Financial Datasets
* **Scale:** 6,362,620 transactions / 11 Columns
* **Imbalance:** 8,213 Fraudulent cases (0.13%) vs. 6,354,407 Valid transactions.
* **Key Features:** Transaction type, amount, origin/destination balances, and temporal steps.

## 3. Data Cleaning & Preprocessing
* **Path Resolution:** Utilized raw string literals to handle Windows directory structures.
* **Feature Dropping:** Systematically removed `isFlaggedFraud`, `nameOrig`, and `nameDest` to prevent model overfitting on non-predictive identifiers.
* **Temporal Mapping:** Preserved the `step` column to derive cyclical patterns.

## 4. Advanced Feature Engineering
To move beyond raw data, I engineered domain-specific features that significantly increased model signal:
* **`balance_diff_orig`**: Captured the exact delta in sender accounts.
* **`balance_diff_dest`**: Monitored recipient account influx.
* **`orig_zero_after`**: A binary indicator for accounts completely liquidated (Top Predictor).
* **`hour`**: Derived from `step % 24` to identify high-risk time windows.

## 5. Exploratory Data Analysis (EDA)
* **Correlation Analysis:** Identified perfect multi-collinearity between `oldbalanceOrg` and `newbalanceOrig`, guiding feature selection.
* **Categorical Insights:** Discovered that **100% of fraud** is concentrated in `TRANSFER` and `CASH_OUT` types.
* **Distribution:** Leveraged Histplots to visualize extreme outliers in transaction amounts.

## 6. Machine Learning Strategy
I implemented a dual-modeling approach to benchmark supervised vs. unsupervised performance:
* **XGBoost (Supervised):** Configured with `scale_pos_weight` to mathematically prioritize the minority class.
* **Isolation Forest (Unsupervised):** Utilized with a `contamination` parameter of 0.0013 as a baseline for anomaly detection.

## 7. Model Results
The performance metrics highlight the superiority of cost-sensitive supervised learning for this domain:

| Metric | XGBoost (Supervised) | Isolation Forest (Unsupervised) |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9998** | 0.51 |
| **Fraud Recall** | **1.00 (99.5%)** | 0.03 |
| **Precision** | 0.21 (Trade-off for Security) | 0.01 |
| **False Negatives** | **Only 8 cases missed** | 1,590 cases missed |

> **Technical Note:** In fraud detection, **Recall is King**. The XGBoost model successfully flagged nearly all fraud, accepting a higher false-positive rate as a necessary trade-off for financial security.

## 8. Power BI Decision Support Dashboard
The model outputs were integrated into a professional "Executive Slate" dashboard for stakeholder monitoring.
* **KPIs:** Total Transactions, Fraud Rate %, Total Fraud Impact ($12.06bn).
* **Visuals:** Fraud by Hour (Line), Transaction Volume by Type (Treemap), and Balance Drain Analysis.
* **Aesthetics:** High-contrast Dark Mode (#0D1117) for SOC (Security Operations Center) environments.

## 9. Project Deliverables
* **`CREDIT CARD FRAUD DETECTION Project file.ipynb`**: Full Python Pipeline.
* **`xgb_fraud_model.pkl`**: Optimized production-ready model.
* **`fraud_dashboard_data.csv`**: Engineered dataset for BI reporting.
* **`model_comparison.csv`**: Comparative statistical performance logs.
* **`feature_importance.csv`**: Ranking of predictive variables.

## 10. Key Findings
1. **Engineered Signal:** Engineered features (`balance_diff_orig`) outperformed all raw dataset columns in predictive weight.
2. **Operational Efficiency:** By filtering for `CASH_OUT` and `TRANSFER` types, the computational load for real-time monitoring can be reduced by 50%.
3. **Imbalance Mitigation:** `scale_pos_weight` proved more effective than oversampling (SMOTE) for maintaining feature distribution integrity in this high-dimensional dataset.
