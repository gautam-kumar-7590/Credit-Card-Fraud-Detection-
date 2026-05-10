# Credit Card Fraud Detection: End-to-End Financial Intelligence Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2C2C2C?style=for-the-badge&logo=xgboost&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=power-bi&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Project Overview
This project addresses the critical challenge of identifying fraudulent activities within massive synthetic financial datasets (**PaySim**). Spanning **6.3 million transactions**, the objective was to build a robust detection pipeline that minimizes false negatives while maintaining high computational efficiency.

The solution integrates a **supervised XGBoost model** for high-precision detection and an **unsupervised Isolation Forest model** to demonstrate the necessity of labeled data in complex fraud environments.

---

## 📂 Repository Structure & Document List
The following files represent the full technical lifecycle of the project:

1.  **`CREDIT CARD FRAUD DETECTION Project file.ipynb`**: The primary Python notebook containing data cleaning, feature engineering, and model training.
2.  **`xgb_fraud_model.pkl`**: Serialized XGBoost model optimized for extreme class imbalance.
3.  **`iso_forest_model.pkl`**: Serialized Isolation Forest model used for anomaly benchmarking.
4.  **`fraud_dashboard_data.csv`**: The refined dataset containing all engineered features, exported for Power BI.
5.  **`model_comparison.csv`**: A comparative analysis of Precision, Recall, and F1-Scores.
6.  **`feature_importance.csv`**: Exported weights of variables that contributed most to the model's decisions.
7.  **`Screenshot (486).png` & `Screenshot (487).jpg`**: Visual documentation of the local development environment and final dashboard layout.

---

## 📊 Power BI Decision Support Dashboard
The final phase of the project translates model outputs into actionable business intelligence.

* **Design Theme**: "Executive Slate" / Clean Corporate Dark (#0D1117).
* **Key Metrics (KPIs)**: Total Transactions (6.36M), Total Fraud Cases (~8K), Total Fraud Amount ($12.06bn), and a verified **ROC-AUC of 0.9998**.
* **Visual Logic**:
    * **Line Chart**: Temporal analysis of fraud by "Hour of Day," revealing specific high-risk windows.
    * **Clustered Bar Chart**: Fraud distribution by transaction type (identifying **CASH_OUT** and **TRANSFER** as the high-risk zones).
    * **DAX Implementation**: Custom measures for Fraud Rate %, Transaction Volume, and Total Impact.

---

## ⚙️ Technical Methodology & Findings

### **Advanced Feature Engineering**
The model's high accuracy is attributed to domain-specific feature engineering rather than raw data:
* **`balance_diff_orig`**: Quantifies the delta in the origin account.
* **`orig_zero_after`**: A binary flag for accounts completely emptied (a primary fraud indicator).
* **`hour`**: Derived from the `step` column to capture time-based fraud patterns.

### **Model Performance & Insights**
* **XGBoost (Supervised)**: Utilizing `scale_pos_weight` to handle the 0.13% class imbalance, the model achieved a **1.00 Recall for Fraud**, missing only 8 cases out of 1,643 in the test set.
* **Isolation Forest (Unsupervised)**: With a recall of only 0.03, this comparison validates that in financial fraud, labeled supervised learning is significantly more effective than pure anomaly detection.
* **Key Discovery**: 100% of fraud occurred within `TRANSFER` and `CASH_OUT` types. `DEBIT`, `PAYMENT`, and `CASH_IN` showed zero fraudulent activity, allowing for streamlined monitoring rules.

---

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Libraries**: XGBoost, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
* **Analytics**: Power BI Desktop (DAX, Power Query)
* **Environment**: Jupyter Notebook / Windows File System

---

## 🚀 How to Use
1.  **Clone the Repo**: Download all project files.
2.  **Run the Notebook**: Execute the `.ipynb` file to see the data transformation and model evaluation.
3.  **Open Dashboard**: Use Power BI Desktop to open the provided dataset and view the visual report.
